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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from app.golden import DEFAULT_PATH as GOLDEN_PATH
from app.golden import GoldenCase, load_golden
from app.graph import THRESHOLD, Recall
from app.judge import Judge
from app.metrics import (
    PROMPT_VERSIONS,
    Embed,
    MetricResult,
    context_recall,
    faithfulness,
    response_relevancy,
)
from app.triage import Prior, Triage, format_prior

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
BASELINE_PATH = ROOT / "evals" / "baseline.json"

# Same value the console uses for the prompt (app/console.py RECALL_K). Kept
# here as a literal so this module does not import the HTTP server; a test
# pins the two together.
RECALL_K = 3

BOOTSTRAP_SEED = 17
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE = 0.95

MetricName = Literal["faithfulness", "response_relevancy", "context_recall"]
METRIC_NAMES: tuple[MetricName, ...] = ("faithfulness", "response_relevancy", "context_recall")
# severity_match is a rate scored the same way, so it shares the layout.
SCORE_NAMES: tuple[str, ...] = ("severity_match", *METRIC_NAMES)
GROUPS: tuple[Literal["twin", "no_twin"], ...] = ("twin", "no_twin")

# Provenance keys that must match for a baseline to be comparable.
COMPARABLE_KEYS: tuple[str, ...] = ("judge_digest", "prompt_versions", "golden_sha256")
# Provenance keys copied into the baseline file.
BASELINE_PROVENANCE_KEYS: tuple[str, ...] = (
    "judge_model",
    "judge_digest",
    "prompt_versions",
    "golden_sha256",
    "small_model",
    "large_model",
    "threshold",
    "recall_k",
    "git_sha",
    "timestamp_utc",
)

# (question, n) -> up to n prior incidents, best first. The live one wraps
# `Retriever.hybrid_rerank` and `Retriever.brief`.
Candidates = Callable[[str, int], Sequence[Prior]]
Report = dict[str, Any]


class Invokable(Protocol):
    """What the runner needs from a compiled graph: `invoke(state) -> state`."""

    def invoke(self, state: dict[str, Any]) -> Mapping[str, Any]:
        """Run the graph on `{"question": ...}` and return the final state."""
        ...


class BaselineMissing(Exception):
    """The baseline file does not exist. Distinct from a failed gate."""


class BaselineRefused(Exception):
    """A baseline must not be written from this run. The message says why."""


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


# ---- Statistics -------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for the mean. None when n < 2.

    Resample with replacement, take each resample's mean, read the
    (1-c)/2 and (1+c)/2 quantiles. A fresh seeded generator per call makes the
    result a pure function of the values. The percentile method is crude at
    n = 30 (it under-covers a little); it is shown to size the noise, not to
    prove significance.
    """
    n = len(values)
    if n < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    low = means[int(math.floor(tail * resamples))]
    high = means[min(resamples - 1, int(math.ceil((1 - tail) * resamples)) - 1)]
    return (low, high)


def _summary(values: Sequence[float], n_failed: int, n_total: int, seed: int) -> dict[str, Any]:
    """Mean, CI and failure counts for one score. Mean and CI are None when empty."""
    ci = bootstrap_ci(values, seed=seed)
    return {
        "mean": (sum(values) / len(values)) if values else None,
        "ci": list(ci) if ci else None,
        "n_scored": len(values),
        "n_failed": n_failed,
        "failure_rate": (n_failed / n_total) if n_total else None,
    }


def aggregate(cases: Sequence[Mapping[str, Any]], *, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Summarise a list of case records. Pure.

    Failed metrics are excluded from the mean and counted. Cases with a
    run error are excluded from everything except `n_run_errors`.
    """
    ran = [c for c in cases if c["run_error"] is None]
    scores: dict[str, Any] = {
        "severity_match": _summary(
            [1.0 if c["severity_match"] else 0.0 for c in ran], 0, len(ran), seed
        )
    }
    for name in METRIC_NAMES:
        results = [c["metrics"][name] for c in ran]
        good = [r["score"] for r in results if not r["failed"]]
        scores[name] = _summary(good, len(results) - len(good), len(results), seed)
    twin_cases = [c for c in ran if c["twin_id"] is not None]
    hits = sum(1 for c in twin_cases if c["twin_recalled"])
    return {
        "n": len(cases),
        "n_run_errors": len(cases) - len(ran),
        "scores": scores,
        "twin_recall_rate": (hits / len(twin_cases)) if twin_cases else None,
        "median_latency_s": statistics.median(c["latency_s"] for c in ran) if ran else None,
    }


# ---- The runner -------------------------------------------------------------


def _safe_metric(compute: Callable[[], MetricResult]) -> MetricResult:
    """Run one metric. Any exception becomes a failed result, never an abort.

    The metrics already map `JudgeError` to failed. This also covers an
    embedder fault. The cost: a programming error here is counted as a
    failure too, but it then shows up in the failure rate and fails the gate.
    """
    try:
        return compute()
    except Exception:
        return MetricResult(None, True)


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
        n: {"score": r.score, "failed": r.failed, "trace": [dict(row) for row in r.trace]}
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


# ---- Provenance -------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """SHA-256 hex of a file's bytes. Pins the golden set into the report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_provenance(
    *,
    judge_model: str,
    digest: Callable[[str], str],
    golden_sha256: str,
    small_model: str,
    large_model: str,
    git: Callable[[], tuple[str, bool | None]],
    now: Callable[[], datetime],
    k: int = RECALL_K,
    threshold: float = THRESHOLD,
    seed: int = BOOTSTRAP_SEED,
    recall_enabled: bool = True,
) -> dict[str, Any]:
    """Everything needed to say what produced these numbers. Clock, git and Ollama injected.

    `recall_enabled` is False only for the negative-control run (`--no-recall`).
    Recording it means such a run can never be mistaken for a real one.
    """
    sha, dirty = git()
    return {
        "judge_model": judge_model,
        "judge_digest": digest(judge_model),
        "prompt_versions": dict(PROMPT_VERSIONS),
        "golden_sha256": golden_sha256,
        "small_model": small_model,
        "large_model": large_model,
        "threshold": threshold,
        "recall_k": k,
        "recall_enabled": recall_enabled,
        "git_sha": sha,
        "git_dirty": dirty,
        "timestamp_utc": now().astimezone(timezone.utc).isoformat(timespec="seconds"),
        "bootstrap_seed": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def ollama_digest(model: str) -> str:
    """The digest Ollama reports for a local model, or "unknown".

    The digest is what makes a silent model swap visible: the tag stays the
    same while the weights change. Any failure (server off, model absent)
    returns "unknown" rather than stopping the eval; the gate then treats two
    "unknown" digests as equal, which is a known weak spot.
    """
    try:
        import ollama

        wanted = model if ":" in model else f"{model}:latest"
        for entry in ollama.list().models:
            if entry.model == wanted or entry.model == model:
                return entry.digest or "unknown"
    except Exception:
        return "unknown"
    return "unknown"


def git_info() -> tuple[str, bool | None]:
    """Short commit SHA and whether the tree has uncommitted changes.

    Falls back to ("unknown", None) outside a git checkout or without git.
    A dirty tree means the SHA does not fully describe the code that ran.
    """

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain"))
    except Exception:
        return "unknown", None


# ---- Files ------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON to a temp file in the same directory, then rename over `path`.

    The rename is atomic on one filesystem, so a reader or a crash never sees
    a half-written report or baseline. The temp file is removed on failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def report_path(now: datetime, directory: Path = REPORTS_DIR) -> Path:
    """`reports/eval-<UTC timestamp>.json`. Second resolution is enough for a sequential run."""
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"eval-{stamp}.json"


def write_report(report: Mapping[str, Any], now: datetime, directory: Path = REPORTS_DIR) -> Path:
    """Write the report atomically and return its path."""
    path = report_path(now, directory)
    _atomic_write_json(path, report)
    return path


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    """Read the baseline. Raises `BaselineMissing` with the fix, never a bare crash."""
    if not path.exists():
        raise BaselineMissing(
            f"no baseline at {path.name}. Run `python -m app.eval_answers --write-baseline` "
            "once, review the numbers, and commit the file."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---- Baseline and gate ------------------------------------------------------


def baseline_refusals(report: Mapping[str, Any], max_failure_rate: float) -> list[str]:
    """Reasons not to write a baseline from this run. Empty means fine."""
    overall = report["aggregates"]["overall"]
    reasons = []
    if report["provenance"].get("recall_enabled") is False:
        reasons.append("recall was disabled for this run (negative control)")
    if report["provenance"].get("git_dirty") is True:
        reasons.append("the working tree had uncommitted changes, so the code is not traceable")
    if overall["n_run_errors"]:
        reasons.append(f"{overall['n_run_errors']} case(s) had a run error")
    for name in METRIC_NAMES:
        rate = overall["scores"][name]["failure_rate"]
        if rate is not None and rate > max_failure_rate:
            reasons.append(f"{name} failure rate {rate:.2f} is above {max_failure_rate:.2f}")
    return reasons


def write_baseline(
    report: Mapping[str, Any],
    path: Path,
    *,
    floor_margin: float = 0.05,
    max_failure_rate: float = 0.15,
) -> None:
    """Freeze this run as the baseline. Never called automatically.

    Floors are the overall mean minus a margin, clamped at 0. They are
    absolute minimums: a release below them fails even if it matches the
    baseline. Refuses a run with run errors or a failure rate above the cap,
    because the numbers would not be trustworthy. A score with no mean gets
    no floor.
    """
    reasons = baseline_refusals(report, max_failure_rate)
    if reasons:
        raise BaselineRefused("; ".join(reasons))
    overall = report["aggregates"]["overall"]["scores"]
    provenance = report["provenance"]
    payload = {
        "provenance": {k: provenance[k] for k in BASELINE_PROVENANCE_KEYS if k in provenance},
        "scores": {n: {"mean": overall[n]["mean"], "ci": overall[n]["ci"]} for n in SCORE_NAMES},
        "floors": {
            n: round(max(0.0, overall[n]["mean"] - floor_margin), 4)
            for n in SCORE_NAMES
            if overall[n]["mean"] is not None
        },
        "max_failure_rate": max_failure_rate,
    }
    _atomic_write_json(path, payload)


def comparability_problems(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Which provenance fields differ, as messages. Empty means comparable."""
    problems = []
    for key in COMPARABLE_KEYS:
        now, then = report["provenance"].get(key), baseline["provenance"].get(key)
        if now != then:
            problems.append(f"{key} differs from the baseline")
    if problems:
        problems.append(
            "baseline is not comparable: re-baseline deliberately "
            "(--write-baseline) after reviewing the change; means were not compared"
        )
    return problems


def gate(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Return every reason to block the release. Empty list means pass.

    (a) a mean below its absolute floor
    (b) regression: mean below the baseline mean AND the baseline mean above
        the current 95% CI upper bound (so noise alone does not fail it)
    (c) a metric failure rate above `max_failure_rate`
    (d) any run error
    (e) the baseline was made with a different judge, prompts or golden set
    When (e) fires, (a) and (b) are skipped: the comparison is meaningless.
    """
    problems: list[str] = []
    overall = report["aggregates"]["overall"]
    scores = overall["scores"]

    if overall["n_run_errors"]:
        errors = [f"{c['id']}: {c['run_error']}" for c in report["cases"] if c["run_error"]]
        problems.append(f"{len(errors)} run error(s): " + "; ".join(errors))

    cap = baseline["max_failure_rate"]
    for name in METRIC_NAMES:
        rate = scores[name]["failure_rate"]
        if rate is not None and rate > cap:
            problems.append(f"{name}: failure rate {rate:.2f} is above the cap {cap:.2f}")

    incomparable = comparability_problems(report, baseline)
    if incomparable:
        return problems + incomparable

    for name, floor in baseline["floors"].items():
        mean = scores[name]["mean"]
        if mean is None:
            problems.append(f"{name}: no scored cases, cannot check the floor {floor:.2f}")
        elif mean < floor:
            problems.append(f"{name}: mean {mean:.3f} is below the floor {floor:.2f}")

    for name, then in baseline["scores"].items():
        mean, ci, base = scores[name]["mean"], scores[name]["ci"], then["mean"]
        if mean is None or base is None or ci is None:
            continue  # no interval at n < 2: a regression cannot be shown
        if mean < base and base > ci[1]:
            problems.append(
                f"{name}: regression. mean {mean:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], "
                f"baseline {base:.3f} is above the interval"
            )
    return problems


# ---- Output -----------------------------------------------------------------


def _cell(stat: Mapping[str, Any]) -> str:
    """`0.83 [0.70,0.93] f2`, or `n/a` with no scored cases. CI may be None."""
    if stat["mean"] is None:
        return f"n/a f{stat['n_failed']}"
    ci = stat["ci"]
    span = f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else " [n/a]"
    return f"{stat['mean']:.2f}{span} f{stat['n_failed']}"


# ---- Saved reports: load, show one case, freeze a baseline ------------------

# A report is a few hundred KB. A file far larger than that is not one of ours,
# and reading it whole would be an easy way to exhaust memory.
MAX_REPORT_BYTES = 20_000_000


class ReportInvalid(ValueError):
    """The file is not a report this module wrote."""


def latest_report(directory: Path = REPORTS_DIR) -> Path:
    """The newest report file by name (names carry a UTC timestamp)."""
    found = sorted(directory.glob("eval-*.json"))
    if not found:
        raise ReportInvalid(f"no reports in {directory}")
    return found[-1]


def load_report(path: Path) -> dict[str, Any]:
    """Read a saved report. Raises `ReportInvalid` for anything that is not one."""
    try:
        if path.stat().st_size > MAX_REPORT_BYTES:
            raise ReportInvalid(f"{path.name} is too large to be a report")
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise ReportInvalid(f"cannot read {path.name}: {err}") from err
    if not isinstance(data, dict) or not {"provenance", "aggregates", "cases"} <= data.keys():
        raise ReportInvalid(f"{path.name} is missing provenance, aggregates or cases")
    return data


def _verdict_rows(trace: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    """One line per judge decision: [x] supported, [ ] not, then the text and reason."""
    return [
        f"    [{'x' if row.get('supported') else ' '}] {row.get(key, '')}"
        + (f"  -- {row['reason']}" if row.get("reason") else "")
        for row in trace
    ]


def format_case(record: Mapping[str, Any], golden: GoldenCase | None) -> str:
    """One case end to end, as plain text, for a person to check by eye.

    Everything the model and the judge saw, and every decision the judge made.
    Tolerates a run error and reports saved before traces existed.
    """
    verdict = {True: "MATCH", False: "MISS", None: "-"}[record.get("severity_match")]
    head = (
        f"{record['id']}  {record['group']}  source {record['source_id']}  "
        f"expected {golden.expected_severity if golden else '?'}  "
        f"answered {record.get('severity')}  {verdict}  "
        f"model {record.get('model_used')}  {record.get('latency_s', 0):.1f}s"
    )
    twin = record.get("twin_id")
    lines = [head, f"ALERT      {golden.question if golden else '?'}"]
    if golden:
        lines.append("REFERENCE  " + golden.reference.replace("\n", "\n           "))
    recalled = ", ".join(
        f"{i} (twin)" if i == twin else i for i in record.get("recalled_ids", [])
    )
    lines.append(f"RECALLED   {recalled or 'nothing'}")
    if record.get("run_error"):
        lines.append(f"RUN ERROR  {record['run_error']}")
        return "\n".join(lines)
    lines.append("SEEN BY THE MODEL AND THE JUDGE")
    lines.extend(f"    {c}" for c in record.get("contexts", []) or ["(nothing was recalled)"])
    lines.append(f"ANSWER     {record.get('answer')}")
    for name, key in (
        ("faithfulness", "claim"),
        ("response_relevancy", "question"),
        ("context_recall", "sentence"),
    ):
        m = record.get("metrics", {}).get(name, {})
        if not m:
            lines.append(f"{name}: not recorded")
            continue
        if m.get("failed") or m.get("score") is None:
            lines.append(f"{name}: FAILED (the judge gave no usable result)")
            continue
        lines.append(f"{name}: {m['score']:.3f}")
        trace = m.get("trace", [])
        if name == "response_relevancy":
            lines.extend(f"    {row['similarity']:.2f}  {row['question']}" for row in trace)
        else:
            lines.extend(_verdict_rows(trace, key))
    return "\n".join(lines)


def show_case(case_id: str, report_path: Path | None, reports_dir: Path) -> int:
    """Print one case from a saved report. Returns the exit code."""
    try:
        report = load_report(report_path or latest_report(reports_dir))
    except ReportInvalid as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    record = next((r for r in report["cases"] if r.get("id") == case_id), None)
    if record is None:
        known = ", ".join(r.get("id", "?") for r in report["cases"])
        print(f"error: no case {case_id!r} in that report. Cases: {known}", file=sys.stderr)
        return 2
    golden = {c.id: c for c in load_golden()}.get(case_id)
    print(format_case(record, golden))
    return 0


def freeze_baseline(report_path: Path, baseline_path: Path) -> int:
    """Freeze a saved report as the baseline, with the same refusals as a live run.

    The baseline is then exactly the numbers that were reviewed, not a fresh
    run that happens to follow them. Returns the exit code.
    """
    try:
        report = load_report(report_path)
        write_baseline(report, baseline_path)
    except ReportInvalid as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except BaselineRefused as err:
        print(f"baseline refused: {err}", file=sys.stderr)
        return 1
    print(f"baseline written: {baseline_path.name} (from {report_path.name})")
    return 0


def format_table(report: Mapping[str, Any]) -> str:
    """Markdown table: one row per group, mean with 95% CI and failed count (f).

    A cell is `mean [low,high] fN`: N metric calls failed and are not in the mean.
    """
    header = ["group", "n", "severity_match", *METRIC_NAMES, "twin_recall", "median_s", "run_err"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for group in ("overall", *GROUPS):
        agg = report["aggregates"][group]
        recall = agg["twin_recall_rate"]
        latency = agg["median_latency_s"]
        row = [
            group,
            str(agg["n"]),
            *(_cell(agg["scores"][name]) for name in SCORE_NAMES),
            "n/a" if recall is None else f"{recall:.2f}",
            "n/a" if latency is None else f"{latency:.1f}",
            str(agg["n_run_errors"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


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
    import psycopg
    from dotenv import load_dotenv
    from pgvector.psycopg import register_vector

    from app.graph import build_graph, ollama_ask
    from app.ingest import database_url, make_embedder
    from app.judge import make_ollama_judge
    from app.retrieve import Retriever, load_passages, make_reranker, postgres_vector_search

    load_dotenv()
    small, large = os.environ["SMALL_MODEL"], os.environ["LARGE_MODEL"]
    judge_model = os.environ.get("JUDGE_MODEL", "qwen2.5:7b")
    cases = load_golden()
    embedder = make_embedder()
    with psycopg.connect(database_url(), autocommit=True) as conn:
        register_vector(conn)
        retriever = Retriever(
            load_passages(conn), postgres_vector_search(conn, embedder), make_reranker()
        )

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
    except KeyError as err:
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
    if args.write_baseline:
        try:
            write_baseline(report, baseline_path)
            print(f"baseline written: {baseline_path.name}")
        except BaselineRefused as err:
            print(f"baseline refused: {err}", file=sys.stderr)
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
