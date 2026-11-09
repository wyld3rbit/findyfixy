"""Run every golden case through a system under test, score it, and build the report.

What it does
    1. Asks the `SystemUnderTest` for an `Answer` to each `Case`.
    2. Scores each answer: every expected label by exact match, then four
       model-graded metrics (harness/eval/metrics.py): faithfulness, response
       relevancy, context recall and action quality.
    3. Aggregates overall and per group with seeded bootstrap 95% confidence
       intervals, merges the app's own aggregates from `summarise`, and returns
       the report dict. Writing it and gating it are report.py's job.

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
    reports each case as it starts, is answered and is scored, and each metric
    as soon as it is computed. The default reports nothing, so the command line
    and every saved report are unchanged. The callback only observes: nothing
    it does can change a score or a record.

Stopping
    `should_stop` is asked before each case starts. When it says yes, the run
    raises `EvalStopped` and returns no report: a run cut short is not a
    smaller report, because its cases were chosen by when someone pressed
    stop, not by design. The case in flight finishes first; a judge call is
    never cut off halfway.

What changes at scale
    One local model server on one GPU caps parallelism at a few workers
    (`MAX_WORKERS`). Cost is about ten judge calls per case. With more cases,
    run them against a judge server, cache by (prompt, schema) and use paired
    comparison (same cases, both systems) for a tighter test than two CIs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Any

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

# A hard cap, not a suggestion. Each worker holds one more conversation in the
# model server's memory (its KV cache), and every extra request queues on the
# same GPU. Past a few workers the machine swaps and the run gets slower.
MAX_WORKERS = 4


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


def run_case(
    case: Case,
    *,
    sut: SystemUnderTest,
    judge: Judge,
    embed: Embed,
    timer: Callable[[], float],
    on_event: EvalEmit = no_eval_emit,
) -> dict[str, Any]:
    """Run and score one case. Returns the per-case record.

    Every path ends with `case_answered` then `case_scored` on `on_event`, so a
    viewer always sees a case close, even one that failed.
    """
    record = new_record(case)
    start = timer()
    try:
        answer: Answer = sut.answer(case)
    except Exception as err:  # the system's fault, not ours: record it, keep going
        record["latency_s"] = timer() - start
        record["run_error"] = f"{type(err).__name__}: {err}"
        return _unscored(record, case, on_event)
    record["latency_s"] = timer() - start
    try:
        _merge(record, answer.meta, f"answer meta for case {case.id}")
    except ValueError as err:
        # Known only once the system has answered, so it cannot be checked up
        # front. A run error keeps the other cases, and the gate fails on it.
        record["run_error"] = f"{type(err).__name__}: {err}"
        return _unscored(record, case, on_event)
    if answer.error is not None:
        record["run_error"] = answer.error
        return _unscored(record, case, on_event)

    for label, expected in case.expected.items():
        got = answer.labels.get(label)
        record[label] = got
        record[match_name(label)] = got == expected
    _answered(record, case, on_event)
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
        result = _safe_metric(compute)
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
    on_event("case_scored", id=case.id)
    return record


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
) -> Report:
    """Run every case and return the report dict. No I/O.

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

    `should_stop` is asked before each case. Yes raises `EvalStopped` once the
    cases already started have finished (see "Stopping" above).
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}, got {workers}")
    validate(cases, sut)
    labels = sorted({label for case in cases for label in case.expected})
    progress_lock = threading.Lock()
    finished = 0
    on_event("eval_started", n=len(cases), cases=[c.id for c in cases], system=dict(sut.describe()))

    def work(index: int, case: Case) -> dict[str, Any]:
        """Run one case and report progress. The counter is shared, so it is locked."""
        nonlocal finished
        if should_stop():
            with progress_lock:
                done = finished
            raise EvalStopped(f"stopped after {done} of {len(cases)} cases")
        on_event("case_started", id=case.id, group=case.group, index=index + 1, n=len(cases))
        record = run_case(case, sut=sut, judge=judge, embed=embed, timer=timer, on_event=on_event)
        with progress_lock:
            finished += 1
            progress(
                f"[{finished}/{len(cases)}] {case.id} {_status(record, case)} "
                f"({record['latency_s']:.1f}s)"
            )
        return record

    start = timer()
    if workers == 1:
        records = [work(i, case) for i, case in enumerate(cases)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(work, range(len(cases)), cases))
    wall_s = timer() - start
    timing = {
        "workers": workers,
        "wall_s": round(wall_s, 1),
        "cases_per_min": round(60 * len(cases) / wall_s, 2) if wall_s > 0 else None,
    }
    return {
        "provenance": dict(provenance),
        "timing": timing,
        "aggregates": _aggregates(records, sut, labels, seed, warn=progress),
        "cases": records,
    }
