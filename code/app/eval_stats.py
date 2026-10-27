"""Statistics and shared names for the answer evaluation: bootstrap intervals and aggregation.

Why a module of its own: these are pure functions of numbers and dicts. They need no
model, database, file or clock, so they are the easiest part of the evaluation to
test and the part everything else depends on. The names of the scores and groups
live here too, so the runner, the report and the gate agree on them.

Honest limits: with 30 cases one case moves a rate by 3 points. The percentile
bootstrap (seeded, so a report is reproducible) shows how much a mean moves under
resampling. It does not cover judge bias, templated data or sampling noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import random
import statistics
from typing import Any, Literal

BOOTSTRAP_SEED = 17
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE = 0.95

MetricName = Literal["faithfulness", "response_relevancy", "context_recall"]
METRIC_NAMES: tuple[MetricName, ...] = ("faithfulness", "response_relevancy", "context_recall")
# severity_match is a rate scored the same way, so it shares the layout.
SCORE_NAMES: tuple[str, ...] = ("severity_match", *METRIC_NAMES)
GROUPS: tuple[Literal["twin", "no_twin"], ...] = ("twin", "no_twin")

# A saved evaluation report: provenance, aggregates, cases and timing.
Report = dict[str, Any]


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
