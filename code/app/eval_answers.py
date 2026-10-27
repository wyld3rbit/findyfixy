"""Day 5: score the triage agent's answers and gate a release on the scores.

What it does
    1. Runs the triage graph once per golden case, with that case's own
       incident held out of recall (the holdout).
    2. Scores each answer: severity match (exact), faithfulness, response
       relevancy and context recall (app/metrics.py, model-graded).
    3. Aggregates overall and per group with seeded bootstrap 95% confidence
       intervals, stamps provenance, and writes `reports/eval-<UTC>.json`.
    4. `gate` compares a report to a committed baseline
       (`evals/baseline.json`) and lists every problem. Empty list = pass.

Run it:  uv run python -m app.eval_answers [--gate | --write-baseline]

Why holdout
    The golden question is built from one incident. If recall returns that
    incident, the model reads the answer. The score then measures lookup of
    the exact record, not triage. Recall fetches k + 1 candidates, drops the
    source id, keeps k. The twin (same incident, other voice) stays eligible:
    that is what the twin group measures.

What the judge sees
    `contexts` are the prior incidents the model saw, one string per
    incident, formatted and capped by `format_prior`, the same function the
    prompt uses. The judge never sees more than the model saw. Otherwise
    faithfulness would reward facts the model could not have used.

Failure model
    - A graph error (for example `TriageError`) is recorded for that case as
      `run_error`. The run continues. The gate fails on any run error.
    - A failed metric (judge error, zero claims) has `score None`. It is
      excluded from the mean and counted. It is never scored 0. The gate
      fails when the failure rate is above the baseline cap, because a judge
      that fails on hard cases would otherwise raise the mean.
    - Cases with a run error have no answer, so they are left out of every
      mean and are reported as `n_run_errors`.
    - A baseline is only comparable when judge digest, prompt hashes and
      golden hash match. If not, means are not compared. The gate says to
      re-baseline deliberately. Silent comparison across a changed judge
      would blame the agent for a change in the ruler.

Statistics, so nobody over-reads a number
    30 cases: one case moves a rate by 3 points. The percentile bootstrap
    (1000 resamples, fixed seed) shows how much a mean moves under
    resampling. It does not cover judge bias, templated data or temperature
    noise. A regression is declared only when the baseline mean lies above
    the current interval's upper bound, so a dip inside the noise is not a
    failure. With n = 0 or 1 there is no interval (None).

What changes at scale
    Sequential for now: one Ollama on one Apple GPU, with the models loaded
    in shared 16 GB memory. Cost is about ten judge calls per case. Parallel
    cases are possible (Ollama batches concurrent requests on the GPU) but
    need a thread-safe database connection, memory for one more KV cache per
    slot, and they distort the latency numbers. With more cases, run them in
    parallel against a judge server, cache by (prompt, schema) and use paired comparison
    (same cases, both systems) for a tighter test than two separate CIs.

Where things live
    eval_stats.py       bootstrap intervals, aggregation, score and group names
    eval_provenance.py  the stamp that says what produced a report
    eval_report.py      report and baseline files, the gate, the case viewer
    eval_answers.py     this file: holdout recall, running the cases, live wiring
                        and the command line
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
import time
from typing import Any, Protocol

from app.eval_provenance import build_provenance, file_sha256, git_info, ollama_digest
from app.eval_report import (
    BASELINE_PATH,
    BaselineMissing,
    BaselineRefused,
    REPORTS_DIR,
    format_table,
    freeze_baseline,
    gate,
    load_baseline,
    show_case,
    write_baseline,
    write_report,
)
from app.eval_stats import BOOTSTRAP_SEED, GROUPS, Report, aggregate
from app.golden import DEFAULT_PATH as GOLDEN_PATH, GoldenCase, load_golden
from app.graph import Recall
from app.judge import Judge
from app.metrics import Embed, MetricResult, context_recall, faithfulness, response_relevancy
from app.paths import ROOT
from app.settings import MissingEnvironment, load_env, read_models
from app.triage import RECALL_K, Prior, Triage, format_prior
from app.wiring import build_retriever, connect

# (question, n) -> up to n prior incidents, best first. The live one wraps
# `Retriever.hybrid_rerank` and `Retriever.brief`.
Candidates = Callable[[str, int], Sequence[Prior]]


class Invokable(Protocol):
    """What the runner needs from a compiled graph: `invoke(state) -> state`."""

    def invoke(self, state: dict[str, Any]) -> Mapping[str, Any]:
        """Run the graph on `{"question": ...}` and return the final state."""
        ...



# ---- Holdout recall ---------------------------------------------------------


def holdout(candidates: Sequence[Prior], source_id: str, k: int) -> list[Prior]:
    """Drop the case's own incident, keep the first `k` of the rest. Pure.

    The caller must fetch `k + 1` candidates, so that after the drop `k`
    remain. The twin is not special: it stays if the retriever returned it.
    """
    return [p for p in candidates if p.incident_id != source_id][:k]


def make_holdout_recall(
    candidates: Candidates, source_id: str, k: int, sink: list[Prior]
) -> Recall:
    """Build the recall step for one case, bound to that case's source id.

    Every call asks for `k + 1` candidates and filters. The kept priors are
    also written to `sink`, so the runner knows what the model saw. Clearing
    the sink first means a graph that calls recall twice reports the last call,
    which is the one whose prompt the model answered.
    """

    def recall(question: str) -> list[Prior]:
        """Recall minus the source incident."""
        kept = holdout(candidates(question, k + 1), source_id, k)
        sink.clear()
        sink.extend(kept)
        return kept

    return recall


# ---- What the judge sees ----------------------------------------------------


def render_answer(triage: Triage) -> str:
    """Plain-text answer for the judge. Confidence is left out: it is not a claim."""
    return (
        f"Severity {triage.severity}. Component {triage.component}. "
        f"Next action: {triage.next_action}"
    )


def render_contexts(priors: Sequence[Prior]) -> list[str]:
    """One string per recalled incident, formatted and capped as in the prompt."""
    return [format_prior([p]) for p in priors]



# ---- The runner -------------------------------------------------------------


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


def run_case(
    case: GoldenCase,
    *,
    make_graph: Callable[[Recall], Invokable],
    candidates: Candidates,
    judge: Judge,
    embed: Embed,
    k: int,
    timer: Callable[[], float],
) -> dict[str, Any]:
    """Run and score one case. Returns the per-case record.

    The graph is compiled per case so recall can be bound to this case's
    source id. Compiling is cheap; the model calls are not.
    """
    seen: list[Prior] = []
    record: dict[str, Any] = {
        "id": case.id,
        "group": case.group,
        "source_id": case.source_id,
        "twin_id": case.twin_id,
        "recalled_ids": [],
        "twin_recalled": False,
        "model_used": None,
        "latency_s": 0.0,
        "severity": None,
        "severity_match": None,
        # Copied from the golden set so a saved report is self-contained: the
        # golden file can be rebuilt later without changing what this report shows.
        "question": case.question,
        "reference": case.reference,
        "expected_severity": case.expected_severity,
        "answer": None,
        "contexts": [],
        "metrics": {},
        "run_error": None,
    }
    recall = make_holdout_recall(candidates, case.source_id, k, seen)
    start = timer()
    try:
        state = make_graph(recall).invoke({"question": case.question})
        triage = state.get("triage")
        if triage is None:
            raise ValueError("graph returned no triage")
    except Exception as err:
        record["latency_s"] = timer() - start
        record["recalled_ids"] = [p.incident_id for p in seen]
        record["run_error"] = f"{type(err).__name__}: {err}"
        return record
    record["latency_s"] = timer() - start

    contexts = render_contexts(seen)
    answer = render_answer(triage)
    record["recalled_ids"] = [p.incident_id for p in seen]
    record["twin_recalled"] = case.twin_id is not None and case.twin_id in record["recalled_ids"]
    record["model_used"] = state.get("model_used")
    record["severity"] = triage.severity
    record["severity_match"] = triage.severity == case.expected_severity
    results = {
        "faithfulness": _safe_metric(lambda: faithfulness(judge, answer, contexts)),
        "response_relevancy": _safe_metric(
            lambda: response_relevancy(judge, embed, case.question, answer)
        ),
        "context_recall": _safe_metric(lambda: context_recall(judge, case.reference, contexts)),
    }
    # The answer, what the judge was shown, and the judge's own decisions are
    # kept so a score can be audited and hand-checked (`--show-case`).
    record["answer"] = answer
    record["contexts"] = list(contexts)
    record["metrics"] = {
        n: {
            "score": r.score,
            "failed": r.failed,
            "error": r.error,
            "trace": [dict(row) for row in r.trace],
        }
        for n, r in results.items()
    }
    return record


# A hard cap, not a suggestion. Each worker holds one more conversation in the
# model server's memory (its KV cache), and every extra request queues on the
# same GPU. Past a few workers the machine swaps and the run gets slower.
MAX_WORKERS = 4


def run_eval(
    cases: Sequence[GoldenCase],
    *,
    make_graph: Callable[[Recall], Invokable],
    candidates: Candidates,
    judge: Judge,
    embed: Embed,
    provenance: Mapping[str, Any],
    k: int = RECALL_K,
    seed: int = BOOTSTRAP_SEED,
    timer: Callable[[], float] = time.perf_counter,
    progress: Callable[[str], None] = lambda _msg: None,
    workers: int = 1,
) -> Report:
    """Run every case and return the report dict. No I/O.

    Everything that touches a model, the database or the clock is injected,
    so the whole pipeline runs in tests with fakes.

    `workers` > 1 runs cases in a thread pool. Threads are enough: nearly all
    the time is spent waiting on the model server, not on Python. The caller
    must pass thread-safe `candidates` and `embed` (the live wiring puts both
    behind a lock). Records keep the order of `cases`, so a parallel report
    has the same shape as a serial one. Per-case latency under `workers` > 1
    includes queueing, so it is not comparable with a serial run's.
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}, got {workers}")
    progress_lock = threading.Lock()
    finished = 0

    def work(case: GoldenCase) -> dict[str, Any]:
        nonlocal finished
        record = run_case(
            case,
            make_graph=make_graph,
            candidates=candidates,
            judge=judge,
            embed=embed,
            k=k,
            timer=timer,
        )
        status = record["run_error"] or f"severity_match={record['severity_match']}"
        with progress_lock:
            finished += 1
            progress(f"[{finished}/{len(cases)}] {case.id} {status} ({record['latency_s']:.1f}s)")
        return record

    start = timer()
    if workers == 1:
        records = [work(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(work, cases))
    wall_s = timer() - start
    aggregates = {"overall": aggregate(records, seed=seed)}
    for group in GROUPS:
        aggregates[group] = aggregate([r for r in records if r["group"] == group], seed=seed)
    timing = {
        "workers": workers,
        "wall_s": round(wall_s, 1),
        "cases_per_min": round(60 * len(cases) / wall_s, 2) if wall_s > 0 else None,
    }
    return {
        "provenance": dict(provenance),
        "timing": timing,
        "aggregates": aggregates,
        "cases": records,
    }



# ---- Live wiring (only main calls these) ------------------------------------


def live_run(
    progress: Callable[[str], None], *, recall_enabled: bool = True, workers: int = 1
) -> Report:
    """Build the real dependencies and run the eval. Needs Ollama and the database.

    Environment: SMALL_MODEL and LARGE_MODEL (required), JUDGE_MODEL (default
    qwen2.5:7b). Everything heavy is imported here, so importing this module
    stays cheap. The retriever is warmed once before timing so model loading
    does not land in case 1's latency. The Ollama models still load on first
    use, so case 1 may be slower.

    `recall_enabled=False` builds the graph without the recall step. It is the
    negative control: scores must fall and the gate must fail. The report says
    so in its provenance and cannot be saved as a baseline.

    `workers` > 1 runs cases in parallel. Start the model server with
    OLLAMA_NUM_PARALLEL set to match, or its requests queue and nothing speeds
    up. The retriever, the database connection and the embedder are shared,
    and neither is safe to call from two threads at once, so each sits behind
    a lock. The lock covers only retrieval and embedding (well under a second
    per case), never the model calls, which are where the time goes.
    """
    from app.graph import build_graph
    from app.llm import ollama_ask
    from app.ingest import make_embedder
    from app.judge import make_ollama_judge

    load_env()
    models = read_models()
    small, large, judge_model = models.small, models.large, models.judge
    cases = load_golden()
    embedder = make_embedder()
    with connect() as conn:
        retriever = build_retriever(conn, embedder)

        shared = threading.Lock()

        def candidates(question: str, n: int) -> list[Prior]:
            """Top n incidents by hybrid + rerank, as prompt-ready briefs. Thread-safe."""
            with shared:
                found = retriever.hybrid_rerank(question, n)
                return [retriever.brief(f.incident_id) for f in found]

        def embed_locked(texts: Sequence[str]) -> list[list[float]]:
            """The embedder behind a lock: a loaded model is not safe to share across threads."""
            with shared:
                return embedder(texts)

        candidates(cases[0].question, 1)  # warm-up, not measured

        def make_graph(recall: Recall) -> Invokable:
            """The production graph with recall bound to one case."""
            return build_graph(
                ollama_ask,
                small_model=small,
                large_model=large,
                recall=recall if recall_enabled else None,
            )

        provenance = build_provenance(
            judge_model=judge_model,
            digest=ollama_digest,
            golden_sha256=file_sha256(GOLDEN_PATH),
            small_model=small,
            large_model=large,
            git=git_info,
            now=lambda: datetime.now(timezone.utc),
            recall_enabled=recall_enabled,
        )
        return run_eval(
            cases,
            make_graph=make_graph,
            candidates=candidates,
            judge=make_ollama_judge(judge_model),
            embed=embed_locked,
            provenance=provenance,
            progress=progress,
            workers=workers,
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., Report] = live_run,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    reports_dir: Path = REPORTS_DIR,
    baseline_path: Path = BASELINE_PATH,
) -> int:
    """Run the eval, print the table and the report path. Returns the exit code.

    0 pass, 1 gate problems or baseline refused, 2 baseline missing or a
    missing environment variable. `runner`, `now` and the paths are injected
    so tests drive this without a model. Progress goes to stderr.
    """
    parser = argparse.ArgumentParser(prog="python -m app.eval_answers")
    parser.add_argument("--gate", action="store_true", help="compare to the baseline, exit 1 on problems")
    parser.add_argument("--write-baseline", action="store_true", help="freeze this run as the baseline")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=range(1, MAX_WORKERS + 1),
        metavar=f"1..{MAX_WORKERS}",
        help="cases to run in parallel. Set OLLAMA_NUM_PARALLEL on the server to match.",
    )
    parser.add_argument(
        "--no-recall",
        action="store_true",
        help="negative control: run without the recall step. Never saved as a baseline.",
    )
    parser.add_argument("--show-case", metavar="ID", help="print one case from a saved report, with every judge decision")
    parser.add_argument("--report", type=Path, help="the saved report for --show-case (default: the newest)")
    parser.add_argument(
        "--baseline-from",
        metavar="REPORT",
        type=Path,
        help="freeze a saved report as the baseline, without running anything",
    )
    args = parser.parse_args(argv)

    if args.show_case:
        return show_case(args.show_case, args.report, reports_dir)
    if args.baseline_from:
        return freeze_baseline(args.baseline_from, baseline_path)

    baseline = None
    if args.gate:  # fail on a missing baseline before spending an hour on the run
        try:
            baseline = load_baseline(baseline_path)
        except BaselineMissing as err:
            print(f"error: {err}", file=sys.stderr)
            return 2

    try:
        extra: dict[str, Any] = {"recall_enabled": False} if args.no_recall else {}
        if args.workers != 1:
            extra["workers"] = args.workers
        report = runner(lambda msg: print(msg, file=sys.stderr, flush=True), **extra)
    except MissingEnvironment as err:
        print(f"error: set the environment variable {err}", file=sys.stderr)
        return 2
    path = write_report(report, now(), reports_dir)
    print(format_table(report))
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"report: {shown}")
    timing = report.get("timing")
    if timing:
        print(
            f"timing: {timing['wall_s']} s wall, {timing['workers']} worker(s), "
            f"{timing['cases_per_min']} cases/min"
        )

    code = 0
    if baseline is not None:
        problems = gate(report, baseline)
        for problem in problems:
            print(f"GATE FAIL: {problem}")
        print("gate: FAIL" if problems else "gate: pass")
        code = 1 if problems else 0
    if args.write_baseline and code:
        # The gate just failed. Writing this run as the baseline would turn the
        # regression into the new standard, and the next gate would pass.
        print("baseline not written: the gate failed in this same run", file=sys.stderr)
    elif args.write_baseline:
        try:
            write_baseline(report, baseline_path)
            print(f"baseline written: {baseline_path.name}")
        except BaselineRefused as err:
            print(f"baseline refused: {err}", file=sys.stderr)
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
