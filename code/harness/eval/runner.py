"""Run every golden case through a system under test, score it, and build the report.

What it does
    1. Answer pass: asks the `SystemUnderTest` for an `Answer` to every `Case`
       and scores every expected label by exact match.
    2. Judge pass: scores every answer with four model-graded metrics
       (harness/eval/metrics.py): faithfulness, response relevancy, context
       recall and action quality.
    3. Aggregates overall and per group with seeded bootstrap 95% confidence
       intervals, merges the app's own aggregates from `summarise`, and returns
       the report dict. Writing it and gating it are report.py's job.

Why answer everything first, then judge everything
    A local model server on a small machine keeps one model in memory at a
    time. Measured on the 16 GB machine (2026-11-09): loading the judge
    (qwen2.5:7b, 4.6 GB) unloads the answering model (qwen3:4b-instruct,
    2.9 GB) and the reverse, at 2 to 3 s per load. Answering and judging case
    by case swapped models twice per case, about 60 loads on 30 cases. Two
    passes load each model about once. The cost: no case is scored until every
    case is answered, so a viewer sees the first score half way through a run,
    and a run stopped in the judge pass has spent its answering time for
    nothing. A judge served from its own machine removes the swap and with it
    the reason for the two passes.

    What the judge sees does not change: the same answers, the same contexts,
    the same prompts, in the same case and metric order. Only when it sees them
    does. `timing` records each pass (`answer_s`, `judge_s`) next to `wall_s`.

The record schema (one dict per case, kept stable because saved reports and
the baseline depend on it)
    The core keys in `stats.CORE_RECORD_KEYS` (including `labels`, the label
    names), plus for each expected label `L`: `L` (what was answered),
    `L_match` and `expected_L`, plus the case's and the answer's `meta`,
    merged flat. Every key goes in through one collision check, so a meta key
    can never silently overwrite another: that would put a wrong number in a
    report that looks right.

Validate before scoring
    An hour of judge calls must never be thrown away by a mistake that was
    visible before the first case ran. So `run_eval` checks up front: every
    case's record builds without a key collision, no group is named
    "overall", and the app's `summarise` keys (probed on an empty list) do not
    clash with harness aggregate keys. What can only show up later is
    contained: an answer whose meta collides becomes that case's run error,
    and a `summarise` key that clashes only on real records is dropped with a
    warning on the progress sink, so the harness counts always win.

Failure model
    - An answer with `error` set, or any exception from `answer`, is recorded
      for that case as `run_error` ("TypeName: message"). The run continues.
      The gate fails on any run error.
    - A failed metric (judge error, zero claims) has `score None`. It is
      excluded from the mean and counted. It is never scored 0. The gate
      fails when the failure rate is above the baseline cap, because a judge
      that fails on hard cases would otherwise raise the mean.
    - Cases with a run error have no answer, so they are left out of every
      mean and are reported as `n_run_errors`.

Live events
    `run_eval` takes an `on_event` callback (harness/eval/events.py) and
    reports each case as it starts and is answered in the answer pass, then
    each metric as soon as it is computed and each case as it is scored in the
    judge pass. A case with a run error is scored (with no metric) as soon as
    it is answered, because there is nothing to judge. The default reports nothing, so the command line
    and every saved report are unchanged. The callback only observes: nothing
    it does can change a score or a record.

Stopping
    `should_stop` is asked before each case in each pass: before it is
    answered and before it is judged. When it says yes, the run raises
    `EvalStopped` and returns no report: a run cut short is not a smaller
    report, because its cases were chosen by when someone pressed stop, not by
    design. The case in flight finishes first; a judge call is never cut off
    halfway.

Tracing and scores (optional)
    With `tracing`, each case is one trace: a top span "eval case <id>" made
    current while the system answers, so an app that traces its own steps
    (FindyFixy: app/trace_adapter.py) nests them under it with no span handed
    over; then, in the judge pass, a "judge" span in the same trace with one
    child per metric. The case span ends when the answer is done, so its
    latency is the answer's. With `scores`, each label match and each metric
    score goes to the backend on that trace (harness/trace/scores.py); a
    failed metric is an error on its span, never a score of 0. The record
    gains `trace_id`. Neither changes an answer, a score or the report's
    numbers: tracing only observes. Without them nothing is imported.

What changes at scale
    One local model server on one GPU caps parallelism at a few workers
    (`MAX_WORKERS`). Cost is about ten judge calls per case. With more cases,
    run them against a judge server, cache by (prompt, schema) and use paired
    comparison (same cases, both systems) for a tighter test than two CIs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import TYPE_CHECKING, Any, TypeVar

from harness.eval.case import Case
from harness.eval.events import EvalEmit, no_eval_emit
from harness.eval.judge import Judge
from harness.eval.metrics import (
    Embed,
    MetricResult,
    action_quality,
    context_recall,
    faithfulness,
    response_relevancy,
)
from harness.eval.stats import (
    BOOTSTRAP_SEED,
    CORE_AGGREGATE_KEYS,
    CORE_RECORD_KEYS,
    MetricName,
    Report,
    aggregate,
    groups_in,
    match_name,
    require_schema,
)
from harness.eval.sut import Answer, SystemUnderTest

if TYPE_CHECKING:
    from harness.trace.scores import ScoreSink
    from harness.trace.spans import SpanTree
    from harness.trace.tracing import Tracing

# A hard cap, not a suggestion. Each worker holds one more conversation in the
# model server's memory (its KV cache), and every extra request queues on the
# same GPU. Past a few workers the machine swaps and the run gets slower.
MAX_WORKERS = 4

_In = TypeVar("_In")
_Out = TypeVar("_Out")


class EvalStopped(Exception):
    """The caller asked the run to stop. No report is built for a stopped run."""


def _safe_metric(compute: Callable[[], MetricResult]) -> MetricResult:
    """Run one metric. An expected fault becomes a failed result with its reason.

    The metrics already map `JudgeError` to failed. This also covers an
    embedder or network fault (OSError, ValueError, RuntimeError). Anything else,
    such as a KeyError or TypeError, is a bug in our code, so it propagates and
    stops the run with a traceback. Scoring a bug as a "judge failure" would
    hide it behind a failure-rate number.
    """
    try:
        return compute()
    except (OSError, ValueError, RuntimeError) as err:
        return MetricResult(None, True, (), f"{type(err).__name__}: {str(err)[:200]}")


def _merge(record: dict[str, Any], extra: Mapping[str, Any], source: str) -> None:
    """Add `extra` to `record`, refusing any key the record already has."""
    clash = sorted(set(extra) & set(record))
    if clash:
        raise ValueError(f"{source} reuses record keys: {clash}")
    record.update(extra)


def new_record(case: Case) -> dict[str, Any]:
    """The record before the system answers: every key present, nothing scored yet.

    Built up front so a case that fails still has the full schema. The
    question, reference and expected labels are copied from the case so a saved
    report is self-contained: the golden file can be rebuilt later without
    changing what the report shows.
    """
    record: dict[str, Any] = {
        "id": case.id,
        "group": case.group,
        "latency_s": 0.0,
        "question": case.question,
        "reference": case.reference,
        # Stored, not inferred: a reader must not guess labels from key shapes.
        "labels": list(case.expected),
        "answer": None,
        "contexts": [],
        "metrics": {},
        "run_error": None,
    }
    require_schema(record, CORE_RECORD_KEYS, "the core case record")
    for label, expected in case.expected.items():
        _merge(
            record,
            {label: None, match_name(label): None, f"expected_{label}": expected},
            f"case {case.id} label {label!r}",
        )
    _merge(record, case.meta, f"case {case.id} meta")
    return record


def _answered(record: Mapping[str, Any], case: Case, on_event: EvalEmit) -> None:
    """Report what the system answered for one case: its labels, or its run error.

    Plain dicts, not the case's read-only mappings, so a viewer can serialise
    the event as it is.
    """
    on_event(
        "case_answered",
        id=case.id,
        labels={label: record[label] for label in case.expected},
        expected=dict(case.expected),
        run_error=record["run_error"],
    )


def _unscored(record: dict[str, Any], case: Case, on_event: EvalEmit) -> dict[str, Any]:
    """End a case that has a run error: report it answered and scored, return the record."""
    _answered(record, case, on_event)
    on_event("case_scored", id=case.id)
    return record


@dataclass(frozen=True)
class _Answered:
    """One case after the answer pass: its record, and the answer to judge.

    `answer` is None when the case ended with a run error. Its record is then
    final, and the judge pass skips it.
    """

    case: Case
    record: dict[str, Any]
    answer: Answer | None
    tree: SpanTree | None = None  # the case's trace, when tracing is on


@dataclass(frozen=True)
class _ToJudge:
    """A case the judge pass scores. `answer` is never None, so no check is needed."""

    case: Case
    record: dict[str, Any]
    answer: Answer
    tree: SpanTree | None = None


def answer_case(
    case: Case,
    *,
    sut: SystemUnderTest,
    timer: Callable[[], float],
    on_event: EvalEmit = no_eval_emit,
    tracing: Tracing | None = None,
    run_attributes: Mapping[str, Any] | None = None,
) -> _Answered:
    """Answer one case and score its labels, inside the case's trace when tracing is on.

    The case span is current while the system answers, so the app's own spans
    nest under it. It ends here, with the labels and the answer (or the run
    error) on it, and the record gains the trace id. `run_attributes` mark
    which eval run the case belongs to (its session, the control flag), so
    two runs, or a control run and a real one, never mix in the backend.
    The span ends even when something inside raises, so no child is orphaned.
    """
    if tracing is None:
        return _answer(case, sut=sut, timer=timer, on_event=on_event)
    from harness.trace.spans import SpanTree

    tree = SpanTree(tracing.tracer(), f"eval case {case.id}", input=case.question,
                    attributes={**(run_attributes or {}), "eval.case": case.id, "eval.group": case.group})
    try:
        with tree.activate():
            result = _answer(case, sut=sut, timer=timer, on_event=on_event)
    except BaseException as err:
        tree.finish(error=f"{type(err).__name__}: {err}")
        raise
    record = result.record
    tree.set({f"eval.{key}": record[key] for label in case.expected for key in (label, match_name(label))})
    if record["run_error"]:
        tree.finish(error=record["run_error"])
    else:
        tree.finish(output=result.answer.text if result.answer else None)
    try:
        _merge(record, {"trace_id": tree.trace_id}, f"tracing for case {case.id}")
    except ValueError as err:  # the app's meta already uses the name: a run error, like any meta clash
        record["run_error"] = f"{type(err).__name__}: {err}"
        return _Answered(case, record, None, tree)
    return _Answered(case, record, result.answer, tree)


def _answer(
    case: Case,
    *,
    sut: SystemUnderTest,
    timer: Callable[[], float],
    on_event: EvalEmit,
) -> _Answered:
    """Answer one case and score its labels. No judge call, no tracing.

    Emits `case_answered`. A case that ends in a run error also emits
    `case_scored` here, so every case still closes on the stream.
    `latency_s` is the answer alone, as it was when answering and judging
    were one step, so latencies stay comparable with older reports.
    """
    record = new_record(case)
    start = timer()
    try:
        answer: Answer = sut.answer(case)
    except Exception as err:  # the system's fault, not ours: record it, keep going
        record["latency_s"] = timer() - start
        record["run_error"] = f"{type(err).__name__}: {err}"
        return _Answered(case, _unscored(record, case, on_event), None)
    record["latency_s"] = timer() - start
    try:
        _merge(record, answer.meta, f"answer meta for case {case.id}")
    except ValueError as err:
        # Known only once the system has answered, so it cannot be checked up
        # front. A run error keeps the other cases, and the gate fails on it.
        record["run_error"] = f"{type(err).__name__}: {err}"
        return _Answered(case, _unscored(record, case, on_event), None)
    if answer.error is not None:
        record["run_error"] = answer.error
        return _Answered(case, _unscored(record, case, on_event), None)

    for label, expected in case.expected.items():
        got = answer.labels.get(label)
        record[label] = got
        record[match_name(label)] = got == expected
    _answered(record, case, on_event)
    return _Answered(case, record, answer)


def score_case(
    case: Case,
    record: dict[str, Any],
    answer: Answer,
    *,
    judge: Judge,
    embed: Embed,
    on_event: EvalEmit = no_eval_emit,
    tree: SpanTree | None = None,
    scores: ScoreSink | None = None,
) -> None:
    """Judge one answered case with the four metrics. Fills `record` in place.

    Emits one `metric_scored` per metric, then `case_scored`. Takes an
    `Answer`, not an optional one, so a case with a run error cannot reach it.
    Returns nothing on purpose: the record is the one the answer pass made,
    and the report reads it from there, so there is one copy to keep right.

    With `tree` (the case's trace), a "judge" span joins that trace with one
    child per metric. With `scores`, every label match and every metric that
    did not fail is sent as a score on the trace.
    """
    judge_tree = tree.follow("judge") if tree is not None else None
    try:
        results = _judge(case, record, answer, judge=judge, embed=embed, on_event=on_event, judge_tree=judge_tree)
    except BaseException as err:
        if judge_tree is not None:  # end the span even on a bug, so its children are not orphaned
            judge_tree.finish(error=f"{type(err).__name__}: {err}")
        raise
    if judge_tree is not None:
        judge_tree.finish()
    if scores is not None and tree is not None:
        _send_scores(scores, tree.trace_id, case, record, results)
    on_event("case_scored", id=case.id)


def _judge(
    case: Case,
    record: dict[str, Any],
    answer: Answer,
    *,
    judge: Judge,
    embed: Embed,
    on_event: EvalEmit,
    judge_tree: SpanTree | None,
) -> dict[str, MetricResult]:
    """The four metrics, in order, each in its own span when tracing. Fills `record`, returns the results."""
    text, contexts = answer.text, list(answer.contexts)
    # In the fixed metric order, one at a time, so the judge sees the same call
    # sequence as before events existed and a viewer sees each score as it lands.
    metrics: tuple[tuple[MetricName, Callable[[], MetricResult]], ...] = (
        ("faithfulness", lambda: faithfulness(judge, text, contexts)),
        ("response_relevancy", lambda: response_relevancy(judge, embed, case.question, text)),
        ("context_recall", lambda: context_recall(judge, case.reference, contexts)),
        # No action is a failed metric (the metric refuses an empty action),
        # never a zero: the system may simply not propose actions.
        ("action_quality", lambda: action_quality(judge, answer.action or "", case.reference)),
    )
    results: dict[str, MetricResult] = {}
    for name, compute in metrics:
        if judge_tree is not None:
            judge_tree.open("metric", name, attributes={"eval.metric": name})
        result = _safe_metric(compute)
        if judge_tree is not None:
            judge_tree.close("metric", output=result.score, error=result.error if result.failed else None)
        results[name] = result
        on_event(
            "metric_scored",
            id=case.id,
            name=name,
            score=result.score,
            failed=result.failed,
            error=result.error,
        )
    # The answer, what the judge was shown, and the judge's own decisions are
    # kept so a score can be audited and hand-checked (`--show-case`).
    record["answer"] = text
    record["contexts"] = contexts
    record["metrics"] = {
        n: {
            "score": r.score,
            "failed": r.failed,
            "error": r.error,
            "trace": [dict(row) for row in r.trace],
        }
        for n, r in results.items()
    }
    return results


def _send_scores(
    scores: ScoreSink,
    trace_id: str,
    case: Case,
    record: Mapping[str, Any],
    results: Mapping[str, MetricResult],
) -> None:
    """Each label match as a boolean score, each metric that did not fail as a number.

    A failed metric has no score: 0 would claim a bad answer when the judge
    never answered. Its error is on the metric's span instead.
    """
    for label in case.expected:
        name = match_name(label)
        if record[name] is not None:
            scores.score(trace_id, name, 1.0 if record[name] else 0.0, "BOOLEAN")
    for name, result in results.items():
        if not result.failed and result.score is not None:
            scores.score(trace_id, name, float(result.score), "NUMERIC")


def _status(record: Mapping[str, Any], case: Case) -> str:
    """One progress word per case: the run error, or each label's match."""
    if record["run_error"]:
        return record["run_error"]
    matches = [f"{match_name(label)}={record[match_name(label)]}" for label in case.expected]
    return " ".join(matches) or "ok"


def validate(cases: Sequence[Case], sut: SystemUnderTest) -> None:
    """Raise ValueError for any mistake visible before a case runs. Cheap: no model call.

    Builds every case's empty record (which checks its labels and meta for key
    collisions), refuses a group named "overall" (it would replace the overall
    aggregate), and probes `summarise([])` for keys that clash with the
    harness aggregate keys.
    """
    for case in cases:
        new_record(case)
        if case.group == "overall":
            raise ValueError(f"case {case.id}: a case group may not be named 'overall'")
    clash = sorted(set(sut.summarise([])) & set(CORE_AGGREGATE_KEYS))
    if clash:
        raise ValueError(f"summarise() reuses harness aggregate keys: {clash}")


def _aggregates(
    records: Sequence[Mapping[str, Any]],
    sut: SystemUnderTest,
    labels: Sequence[str],
    seed: int,
    warn: Callable[[str], None] = lambda _msg: None,
) -> dict[str, Any]:
    """Overall, then one aggregate per group, each with the app's `summarise` merged in.

    `validate` already probed `summarise` on an empty list. A clash that
    appears only on real records is dropped with a warning, because raising
    here would throw away a run whose cases are already scored.
    """

    def one(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The harness aggregate plus the app's extras. Harness keys always win."""
        core = aggregate(subset, labels=labels, seed=seed)
        extra = dict(sut.summarise(subset))
        for key in sorted(set(extra) & set(core)):
            warn(f"warning: summarise() key {key!r} clashes with a harness aggregate key and was dropped")
            del extra[key]
        return {**core, **extra}

    result = {"overall": one(records)}
    for group in groups_in(records):
        result[group] = one([r for r in records if r["group"] == group])
    return result


def run_eval(
    cases: Sequence[Case],
    *,
    sut: SystemUnderTest,
    judge: Judge,
    embed: Embed,
    provenance: Mapping[str, Any],
    seed: int = BOOTSTRAP_SEED,
    timer: Callable[[], float] = time.perf_counter,
    progress: Callable[[str], None] = lambda _msg: None,
    workers: int = 1,
    on_event: EvalEmit = no_eval_emit,
    should_stop: Callable[[], bool] = lambda: False,
    tracing: Tracing | None = None,
    scores: ScoreSink | None = None,
) -> Report:
    """Run every case and return the report dict. No I/O apart from optional tracing.

    Everything that touches a model, the database or the clock is injected,
    so the whole pipeline runs in tests with fakes.

    `workers` > 1 runs cases in a thread pool. Threads are enough: nearly all
    the time is spent waiting on the model server, not on Python. The caller
    must pass a thread-safe `sut` and `embed`. Records keep the order of
    `cases`, so a parallel report has the same shape as a serial one. Per-case
    latency under `workers` > 1 includes queueing, so it is not comparable with
    a serial run's.

    `on_event` receives the live events (harness/eval/events.py). It is called
    from the worker threads, so under `workers` > 1 it must be thread-safe.
    `eval_started` goes out only after validation, so a run refused up front
    reports nothing.

    `tracing` and `scores` turn on the case traces and scores ("Tracing and
    scores" above). The caller owns both and closes them.

    `should_stop` is asked before each case in each pass. Yes raises
    `EvalStopped` once the cases already started have finished (see
    "Stopping" above).
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}, got {workers}")
    validate(cases, sut)
    labels = sorted({label for case in cases for label in case.expected})
    progress_lock = threading.Lock()
    n = len(cases)
    answered_count = 0
    judged_count = 0
    on_event("eval_started", n=n, cases=[c.id for c in cases], system=dict(sut.describe()))
    # Every case trace of this run shares one session (Langfuse groups traces
    # by `session.id`), so a run's traces are one view and two runs never mix.
    run_attributes = {
        "session.id": f"eval {provenance.get('timestamp_utc', 'unstamped')}",
        "eval.negative_control": bool(provenance.get("negative_control", False)),
    }

    def answer_one(index: int, case: Case) -> _Answered:
        """Answer one case and report progress. The counter is shared, so it is locked."""
        nonlocal answered_count
        if should_stop():
            with progress_lock:
                done = answered_count
            raise EvalStopped(f"stopped after {done} of {n} cases")
        on_event("case_started", id=case.id, group=case.group, index=index + 1, n=n)
        result = answer_case(case, sut=sut, timer=timer, on_event=on_event, tracing=tracing,
                             run_attributes=run_attributes)
        with progress_lock:
            answered_count += 1
            progress(
                f"[{answered_count}/{n}] {case.id} {_status(result.record, case)} "
                f"({result.record['latency_s']:.1f}s)"
            )
        return result

    def judge_one(item: _ToJudge) -> None:
        """Judge one answered case and report progress."""
        nonlocal judged_count
        if should_stop():
            with progress_lock:
                done = judged_count
            raise EvalStopped(f"stopped after judging {done} of {len(to_judge)} answered cases")
        score_case(item.case, item.record, item.answer, judge=judge, embed=embed, on_event=on_event,
                   tree=item.tree, scores=scores)
        with progress_lock:
            judged_count += 1
            progress(f"[judged {judged_count}/{len(to_judge)}] {item.case.id}")

    def each(fn: Callable[[_In], _Out], items: Sequence[_In]) -> list[_Out]:
        """Map `fn` over `items`, serially or in the pool. Results keep input order."""
        if workers == 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    start = timer()
    answered = each(lambda pair: answer_one(*pair), list(enumerate(cases)))
    answered_at = timer()
    to_judge = [_ToJudge(a.case, a.record, a.answer, a.tree) for a in answered if a.answer is not None]
    each(judge_one, to_judge)  # score_case fills each record in place
    finished_at = timer()
    # `answered` keeps the order of `cases`, and holds the records the judge pass filled.
    records = [item.record for item in answered]
    wall_s = finished_at - start
    timing = {
        "workers": workers,
        "wall_s": round(wall_s, 1),
        "answer_s": round(answered_at - start, 1),
        "judge_s": round(finished_at - answered_at, 1),
        "cases_per_min": round(60 * n / wall_s, 2) if wall_s > 0 else None,
    }
    return {
        "provenance": dict(provenance),
        "timing": timing,
        "aggregates": _aggregates(records, sut, labels, seed, warn=progress),
        "cases": records,
    }
