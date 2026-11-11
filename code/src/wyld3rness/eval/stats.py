"""Statistics and shared names for the answer evaluation: bootstrap intervals and aggregation.

Why a module of its own: these are pure functions of numbers and dicts. They need no
model, database, file or clock, so they are the easiest part of the evaluation to
test and the part everything else depends on. The names of the scores live here
too, so the runner, the report and the gate agree on them.

Labels, generalised: a case may expect any number of labels. Each label `L`
becomes the score `L_match`, an exact-match rate scored with the same layout as
a metric. Label scores come first (sorted by name), then the four metrics. A
saved report does not list its labels, so the order is rebuilt from the score
names alone (`ordered_score_names`), and a report read back from disk prints the
same table as the one in memory.

Honest limits: with 30 cases one case moves a rate by 3 points. The percentile
bootstrap (seeded, so a report is reproducible) shows how much a mean moves under
resampling. It does not cover judge bias, templated data or sampling noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import random
import statistics
from typing import Any, Literal

BOOTSTRAP_SEED = 17
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE = 0.95

MetricName = Literal["faithfulness", "response_relevancy", "context_recall", "action_quality"]
METRIC_NAMES: tuple[MetricName, ...] = (
    "faithfulness",
    "response_relevancy",
    "context_recall",
    "action_quality",
)
# The aggregate keys the harness owns. An app's `summarise` may add others, but
# not these: an app number must never overwrite a harness count.
CORE_AGGREGATE_KEYS: tuple[str, ...] = ("n", "n_run_errors", "scores", "median_latency_s")

# The record keys the harness writes for every case: the one definition the
# writer (runner.new_record) checks itself against and the reader
# (report.format_case) uses to tell harness keys from app meta. Anything else
# in a record is a label (`L`, `L_match`, `expected_L`) or the app's meta.
# `labels` lists the record's label names; records written before 2026-11-09
# do not have it.
CORE_RECORD_KEYS: tuple[str, ...] = (
    "id",
    "group",
    "latency_s",
    "question",
    "reference",
    "labels",
    "answer",
    "contexts",
    "metrics",
    "run_error",
)


def require_schema(built: Mapping[str, Any], schema: Sequence[str], what: str) -> None:
    """Raise RuntimeError unless `built` has exactly the keys of `schema`.

    The writer calls this on what it just built, so the writer and the schema
    constant cannot drift apart: the first test run after an edit to one side
    fails, instead of a reader silently mistaking a harness key for app meta.
    """
    if set(built) != set(schema):
        extra, missing = sorted(set(built) - set(schema)), sorted(set(schema) - set(built))
        raise RuntimeError(f"{what} drifted from its schema: extra {extra}, missing {missing}")


# A saved evaluation report: provenance, aggregates, cases and timing.
Report = dict[str, Any]


# ---- Names ------------------------------------------------------------------


def match_name(label: str) -> str:
    """The score name for one label: `severity` -> `severity_match`."""
    return f"{label}_match"


def ordered_score_names(names: Iterable[str]) -> tuple[str, ...]:
    """Label scores sorted by name, then the four metrics in their fixed order.

    Works from names alone, so a report loaded from disk (whose keys were
    sorted by the JSON writer) gets the same order as one fresh from a run.
    """
    label_scores = sorted(set(names) - set(METRIC_NAMES))
    return (*label_scores, *METRIC_NAMES)


def groups_in(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The groups of these records, in the order each first appears. Pure."""
    seen: dict[str, None] = {}
    for record in records:
        group = record.get("group")
        if isinstance(group, str):
            seen.setdefault(group, None)
    return list(seen)


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
    low = means[math.floor(tail * resamples)]
    high = means[min(resamples - 1, math.ceil((1 - tail) * resamples) - 1)]
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


def aggregate(
    cases: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarise a list of case records. Pure.

    Failed metrics are excluded from the mean and counted. Cases with a
    run error are excluded from everything except `n_run_errors`. A label is
    scored only over the cases that expected it, so a mixed golden set (some
    cases with a label, some without) does not count the absent ones as misses.
    """
    ran = [c for c in cases if c["run_error"] is None]
    scores: dict[str, Any] = {}
    for label in sorted(set(labels)):
        key = match_name(label)
        matches = [c[key] for c in ran if c.get(key) is not None]
        scores[key] = _summary([1.0 if m else 0.0 for m in matches], 0, len(matches), seed)
    for name in METRIC_NAMES:
        results = [c["metrics"][name] for c in ran]
        good = [r["score"] for r in results if not r["failed"]]
        scores[name] = _summary(good, len(results) - len(good), len(results), seed)
    return {
        "n": len(cases),
        "n_run_errors": len(cases) - len(ran),
        "scores": scores,
        "median_latency_s": statistics.median(c["latency_s"] for c in ran) if ran else None,
    }
