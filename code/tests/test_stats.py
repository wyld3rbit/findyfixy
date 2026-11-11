"""Tests for the statistics (wyld3rness/eval/stats.py). Plain values, no fakes needed.

What these tests cover
    The seeded bootstrap, aggregation by hand (failures excluded and counted,
    never scored 0), label scores over only the cases that expect the label,
    and the score and group order a saved report is read back in.
"""

from typing import Any

import pytest

from wyld3rness.eval.stats import (
    aggregate,
    bootstrap_ci,
    groups_in,
    match_name,
    ordered_score_names,
)


def rec(
    group: str, ok: bool | None, faith: float | None, latency: float = 1.0, error: str | None = None
) -> dict[str, Any]:
    """A case record with label `kind` (absent when `ok` is None)."""

    def m(score: float | None) -> dict[str, Any]:
        """One metric result with this score, failed when None."""
        return {"score": score, "failed": score is None}

    record: dict[str, Any] = {
        "id": "x",
        "group": group,
        "latency_s": latency,
        "run_error": error,
        "metrics": {}
        if error
        else {
            "faithfulness": m(faith),
            "response_relevancy": m(0.8),
            "context_recall": m(0.4),
            "action_quality": m(0.5),
        },
    }
    if ok is not None:
        record.update(kind="x", kind_match=ok, expected_kind="x")
    return record


def test_aggregate_by_hand_excludes_failures_and_never_scores_them_zero() -> None:
    """Hand-computed means: a failed metric is out of the mean and counted."""
    records = [rec("a", True, 1.0), rec("a", False, 0.5), rec("b", True, None), rec("b", True, 0.0)]
    agg = aggregate(records, labels=["kind"])
    faith = agg["scores"]["faithfulness"]
    assert faith["mean"] == pytest.approx(0.5)  # (1.0 + 0.5 + 0.0) / 3, the failure is out
    assert faith["n_scored"] == 3 and faith["n_failed"] == 1
    assert faith["failure_rate"] == pytest.approx(0.25)
    assert agg["scores"]["kind_match"]["mean"] == pytest.approx(0.75)
    assert agg["scores"]["context_recall"]["mean"] == pytest.approx(0.4)
    assert set(agg) == {"n", "n_run_errors", "scores", "median_latency_s"}


def test_run_errors_are_counted_and_left_out_of_every_mean() -> None:
    """A run error counts in `n_run_errors` and nowhere else."""
    records = [rec("a", True, 1.0, latency=2.0), rec("a", None, None, error="boom")]
    agg = aggregate(records, labels=["kind"])
    assert agg["n"] == 2 and agg["n_run_errors"] == 1
    assert agg["scores"]["kind_match"]["n_scored"] == 1
    assert agg["median_latency_s"] == 2.0


def test_a_label_is_scored_only_over_the_cases_that_expect_it() -> None:
    """An unlabelled case is not a miss."""
    records = [rec("a", True, 1.0), rec("a", None, 1.0)]
    stat = aggregate(records, labels=["kind"])["scores"]["kind_match"]
    assert stat["mean"] == 1.0 and stat["n_scored"] == 1  # the unlabelled case is not a miss


def test_label_scores_come_first_sorted_then_the_metrics_in_fixed_order() -> None:
    """The score order is rebuilt from names alone."""
    names = ordered_score_names(["action_quality", "zeta_match", "faithfulness", "alpha_match"])
    assert names == (
        "alpha_match",
        "zeta_match",
        "faithfulness",
        "response_relevancy",
        "context_recall",
        "action_quality",
    )
    assert match_name("severity") == "severity_match"


def test_groups_keep_their_first_appearance_order() -> None:
    """Groups are listed in the order their first record appears."""
    assert groups_in([{"group": "twin"}, {"group": "no_twin"}, {"group": "twin"}, {}]) == ["twin", "no_twin"]


# ---- Bootstrap --------------------------------------------------------------


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    """Same values, same interval, and the mean lies inside it."""
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first, second = bootstrap_ci(values), bootstrap_ci(values)
    assert first == second
    low, high = first  # type: ignore[misc]
    assert low <= sum(values) / len(values) <= high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_depends_on_the_seed() -> None:
    """The seed is really used."""
    values = [0.1, 0.9, 0.4, 0.7, 0.3, 0.8]
    assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)


def test_n0_and_n1_have_no_interval_and_do_not_crash() -> None:
    """No interval below two values, and no crash."""
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([0.5]) is None
    assert aggregate([], labels=["kind"])["scores"]["faithfulness"]["mean"] is None
    one = aggregate([rec("a", True, 1.0)], labels=["kind"])["scores"]["faithfulness"]
    assert one["mean"] == 1.0 and one["ci"] is None
