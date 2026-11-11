"""FindyFixy reports written before the harness (before 2026-11-09) still read, gate and refreeze.

What these tests pin, without the git-ignored reports on disk
    A legacy report is built by hand with the provenance keys and the numbers
    of the committed baseline. The tests pin the exact baseline provenance keys
    and floors, that the final run passes, the 8 gate messages of the no-recall
    control word for word, the table rows in their old order, and that the old
    control flag `recall_enabled: False` (through the app's `LEGACY_CONTROL`)
    still refuses a baseline.

Why app-side: `recall_enabled` is FindyFixy's word. The harness only knows the
hook (`legacy_control`), tested generically in tests/test_harness_eval_report.py.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from findyfixy.evals.answers import LEGACY_CONTROL
from wyld3rness.eval.report import BaselineRefused, format_table, gate, load_baseline, write_baseline


def stat(mean: float, ci: list[float]) -> dict[str, Any]:
    """One score's summary, shaped as `aggregate` writes it, with no failures."""
    return {"mean": mean, "ci": ci, "n_scored": 30, "n_failed": 0, "failure_rate": 0.0}


# The committed baseline's numbers and provenance, copied here so this test does
# not depend on any file. Score means and CIs are those of the final Day 5 run.
LEGACY_PROVENANCE = {
    "bootstrap_resamples": 1000,
    "bootstrap_seed": 17,
    "git_dirty": False,
    "git_sha": "9a4d12c",
    "golden_sha256": "2d00",
    "judge_digest": "845d",
    "judge_model": "qwen2.5:7b",
    "large_model": "qwen3:4b-instruct",
    "prompt_versions": {"faithfulness": "799f"},
    "recall_enabled": True,
    "recall_k": 3,
    "small_model": "qwen3:4b-instruct",
    "threshold": 0.7,
    "timestamp_utc": "2026-10-27T04:27:44+00:00",
}
# The provenance keys the baseline writer kept before the harness existed.
LEGACY_BASELINE_KEYS = {
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
}
FINAL = {
    "action_quality": (0.9166666666666666, [0.8166666666666667, 0.9833333333333333]),
    "context_recall": (0.9277777777777777, [0.8555555555555556, 0.9833333333333333]),
    "faithfulness": (0.7855555555555556, [0.7022222222222222, 0.8688888888888889]),
    "response_relevancy": (0.6677625016951377, [0.6442724935937852, 0.6896826113555243]),
    "severity_match": (0.7666666666666667, [0.6, 0.9]),
}
CONTROL = {
    "action_quality": (0.1, [0.03333333333333333, 0.16666666666666666]),
    "context_recall": (0.0, [0.0, 0.0]),
    "faithfulness": (0.0, [0.0, 0.0]),
    "response_relevancy": (0.72, [0.69, 0.74]),
    "severity_match": (0.16666666666666666, [0.03333333333333333, 0.3333333333333333]),
}
# Word for word what the gate printed for the no-recall control on 2026-10-26.
CONTROL_PROBLEMS = [
    "action_quality: mean 0.100 is below the floor 0.87",
    "context_recall: mean 0.000 is below the floor 0.88",
    "faithfulness: mean 0.000 is below the floor 0.74",
    "severity_match: mean 0.167 is below the floor 0.72",
    "action_quality: regression. mean 0.100, 95% CI [0.033, 0.167], baseline 0.917 is above the interval",
    "context_recall: regression. mean 0.000, 95% CI [0.000, 0.000], baseline 0.928 is above the interval",
    "faithfulness: regression. mean 0.000, 95% CI [0.000, 0.000], baseline 0.786 is above the interval",
    "severity_match: regression. mean 0.167, 95% CI [0.033, 0.333], baseline 0.767 is above the interval",
]


def legacy_report(means: dict[str, tuple[float, list[float]]], **prov: Any) -> dict[str, Any]:
    """A FindyFixy report as written before 2026-11-09: sorted keys, `recall_enabled`, twin data."""

    def agg(n: int, recall: float | None) -> dict[str, Any]:
        """One aggregate with these means, `n` cases and this twin recall rate."""
        return {
            "median_latency_s": 4.8,
            "n": n,
            "n_run_errors": 0,
            "twin_recall_rate": recall,
            "scores": {k: stat(m, ci) for k, (m, ci) in sorted(means.items())},
        }

    cases = [{"id": "g01", "group": "twin", "run_error": None}, {"id": "g21", "group": "no_twin", "run_error": None}]
    # Keys in JSON order, as a report reads back from disk: no_twin before twin.
    aggregates = {"no_twin": agg(10, None), "overall": agg(30, 0.95), "twin": agg(20, 0.95)}
    return {"provenance": {**LEGACY_PROVENANCE, **prov}, "aggregates": aggregates, "cases": cases}


def test_a_legacy_report_refreezes_to_the_same_baseline_provenance(tmp_path: Path) -> None:
    """The same 10 provenance keys and the same floors as the committed baseline."""
    write_baseline(legacy_report(FINAL), tmp_path / "b.json", legacy_control=LEGACY_CONTROL)
    data = json.loads((tmp_path / "b.json").read_text())
    assert set(data["provenance"]) == LEGACY_BASELINE_KEYS
    assert data["floors"] == {
        "action_quality": 0.8667,
        "context_recall": 0.8778,
        "faithfulness": 0.7356,
        "response_relevancy": 0.6178,
        "severity_match": 0.7167,
    }


def test_the_legacy_final_run_passes_and_the_control_fails_with_the_same_eight_problems(tmp_path: Path) -> None:
    """The final run passes; the control prints the 8 messages of 2026-10-26, in order."""
    path = tmp_path / "b.json"
    write_baseline(legacy_report(FINAL), path, legacy_control=LEGACY_CONTROL)
    baseline = load_baseline(path)
    assert gate(legacy_report(FINAL), baseline) == []
    assert gate(legacy_report(CONTROL, recall_enabled=False, git_sha="70decf4"), baseline) == CONTROL_PROBLEMS


def test_a_legacy_report_prints_the_same_rows_in_the_same_order() -> None:
    """Same row values, rows in overall, twin, no_twin order."""
    lines = format_table(legacy_report(FINAL)).splitlines()
    assert lines[0] == (
        "| group | n | severity_match | faithfulness | response_relevancy | context_recall "
        "| action_quality | twin_recall_rate | median_s | run_err |"
    )
    assert lines[2] == (
        "| overall | 30 | 0.77 [0.60,0.90] f0 | 0.79 [0.70,0.87] f0 | 0.67 [0.64,0.69] f0 "
        "| 0.93 [0.86,0.98] f0 | 0.92 [0.82,0.98] f0 | 0.95 | 4.8 | 0 |"
    )
    assert [line.split(" | ")[0] for line in lines[2:]] == ["| overall", "| twin", "| no_twin"]


def test_a_legacy_control_report_is_refused_as_a_baseline(tmp_path: Path) -> None:
    """`recall_enabled: False` still marks a control through the app's hook."""
    with pytest.raises(BaselineRefused, match="negative control"):
        write_baseline(legacy_report(FINAL, recall_enabled=False), tmp_path / "b.json", legacy_control=LEGACY_CONTROL)
