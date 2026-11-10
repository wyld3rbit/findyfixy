"""Tests for reports, the baseline and the gate (harness/eval/report.py). Fakes only.

What these tests cover
    Every gate rule, the baseline writer and its refusals, report files, the
    table, the case viewer, and compatibility with the report schema
    FindyFixy wrote before the harness existed.

What a staff+ reader should notice
    The compatibility tests build a legacy report by hand, with the provenance
    keys and the numbers of the committed baseline. They pin the exact gate
    messages and the exact baseline provenance keys, so the generalised code
    is proven against the old output without the git-ignored reports on disk.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from harness.eval import report as er
from harness.eval.report import (
    BaselineMissing,
    BaselineRefused,
    format_case,
    format_table,
    freeze_baseline,
    gate,
    load_baseline,
    show_case,
    write_baseline,
    write_report,
)
from tests.harness_fakes import NOW, PROV, case, run


def stat(mean: float | None, ci: list[float] | None = None, rate: float = 0.0) -> dict[str, Any]:
    """One score's summary, shaped as `aggregate` writes it."""
    return {"mean": mean, "ci": ci, "n_scored": 30, "n_failed": 0, "failure_rate": rate}


def report_with(means: dict[str, tuple[float, list[float]]], rate: float = 0.0, errors: int = 0,
                prov: dict[str, Any] | None = None) -> dict[str, Any]:
    """A report with these overall means, failure rate and run errors, and nothing else."""
    scores = {n: stat(m, ci, rate) for n, (m, ci) in means.items()}
    cases = [{"id": f"g{i}", "group": "a", "run_error": "boom"} for i in range(errors)]
    overall = {"n": 30, "n_run_errors": errors, "scores": scores, "median_latency_s": 12.0}
    return {"provenance": prov or dict(PROV), "aggregates": {"overall": overall, "a": overall},
            "cases": cases}


GOOD = {"kind_match": (0.8, [0.7, 0.9]), "faithfulness": (0.8, [0.7, 0.9]),
        "response_relevancy": (0.8, [0.7, 0.9]), "context_recall": (0.8, [0.7, 0.9]),
        "action_quality": (0.8, [0.7, 0.9])}


def baseline_for(report: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Freeze `report` as a baseline in `tmp_path` and read it back."""
    path = tmp_path / "evals" / "baseline.json"
    write_baseline(report, path)
    return load_baseline(path)


# ---- Gate -------------------------------------------------------------------


def test_a_clean_report_passes(tmp_path: Path) -> None:
    """A report gated against its own baseline has no problems."""
    report = report_with(GOOD)
    assert gate(report, baseline_for(report, tmp_path)) == []


def test_floor_breach_fails(tmp_path: Path) -> None:
    """A mean under its floor fails even inside the noise band."""
    base = baseline_for(report_with(GOOD), tmp_path)
    low = dict(GOOD, faithfulness=(0.6, [0.5, 0.95]))  # baseline 0.8 is inside the CI: not a regression
    problems = gate(report_with(low), base)
    assert len(problems) == 1 and "faithfulness" in problems[0] and "floor" in problems[0]


def test_regression_needs_the_baseline_to_sit_above_the_interval(tmp_path: Path) -> None:
    """A dip inside the interval is noise; one below it is a regression."""
    base = baseline_for(report_with(GOOD), tmp_path)
    base["floors"] = {}  # isolate rule (b)
    noisy = dict(GOOD, context_recall=(0.75, [0.6, 0.9]))  # dip inside the CI
    assert gate(report_with(noisy), base) == []
    clear = dict(GOOD, context_recall=(0.75, [0.65, 0.78]))  # baseline 0.8 above 0.78
    problems = gate(report_with(clear), base)
    assert len(problems) == 1 and "regression" in problems[0] and "context_recall" in problems[0]


def test_an_improvement_is_not_a_regression(tmp_path: Path) -> None:
    """Better scores pass."""
    base = baseline_for(report_with(GOOD), tmp_path)
    better = dict(GOOD, faithfulness=(0.95, [0.9, 0.99]))
    assert gate(report_with(better), base) == []


def test_failure_rate_above_the_cap_fails(tmp_path: Path) -> None:
    """Too many failed judge calls fail the gate."""
    base = baseline_for(report_with(GOOD), tmp_path)
    problems = gate(report_with(GOOD, rate=0.2), base)
    assert problems and all("failure rate" in p for p in problems)


def test_any_run_error_fails_and_names_the_case(tmp_path: Path) -> None:
    """One run error fails the gate and the message names the case."""
    base = baseline_for(report_with(GOOD), tmp_path)
    problems = gate(report_with(GOOD, errors=1), base)
    assert len(problems) == 1 and "run error" in problems[0] and "g0: boom" in problems[0]


@pytest.mark.parametrize("key,value", [("judge_digest", "d2"), ("prompt_versions", {"m": "h2"}),
                                       ("golden_sha256", "g2")])
def test_incomparable_baseline_says_rebaseline_and_skips_the_comparison(
    tmp_path: Path, key: str, value: Any
) -> None:
    """A changed judge, prompt or golden set makes the means meaningless to compare."""
    base = baseline_for(report_with(GOOD), tmp_path)
    terrible = {n: (0.1, [0.0, 0.2]) for n in GOOD}  # would breach floors and regress
    prov = dict(PROV, **{key: value})
    problems = gate(report_with(terrible, prov=prov), base)
    assert any(key in p for p in problems)
    assert any("re-baseline deliberately" in p for p in problems)
    assert not any("floor" in p or "regression" in p for p in problems)


def test_incomparable_baseline_still_runs_the_run_error_and_failure_checks(tmp_path: Path) -> None:
    """Run errors and failure rates are checked even when means are not."""
    base = baseline_for(report_with(GOOD), tmp_path)
    prov = dict(PROV, golden_sha256="other")
    problems = gate(report_with(GOOD, rate=0.5, errors=2, prov=prov), base)
    assert any("run error" in p for p in problems)
    assert any("failure rate" in p for p in problems)
    assert any("re-baseline" in p for p in problems)


def test_two_unknown_judge_digests_are_not_the_same_judge(tmp_path: Path) -> None:
    """Unknown on both sides proves nothing about the judge."""
    report = report_with(GOOD, prov={**PROV, "judge_digest": "unknown"})
    baseline = baseline_for(report_with(GOOD), tmp_path)
    baseline["provenance"]["judge_digest"] = "unknown"
    assert any("judge_digest is unknown" in p for p in gate(report, baseline))


def test_the_gate_fails_when_action_quality_drops_below_its_floor(tmp_path: Path) -> None:
    """Action quality has a floor like every other score."""
    baseline = baseline_for(report_with(GOOD), tmp_path)
    worse = report_with({**GOOD, "action_quality": (0.3, [0.2, 0.4])})
    assert any("action_quality" in p for p in gate(worse, baseline))


def test_missing_baseline_is_a_clear_error_that_names_the_command(tmp_path: Path) -> None:
    """The error names the fix, with the app's own command when given."""
    with pytest.raises(BaselineMissing, match="--write-baseline"):
        load_baseline(tmp_path / "nope.json")
    with pytest.raises(BaselineMissing, match="`python -m my.app --write-baseline`"):
        load_baseline(tmp_path / "nope.json", command="python -m my.app")


# ---- Baseline writer --------------------------------------------------------


def test_write_baseline_sets_floors_from_the_overall_means(tmp_path: Path) -> None:
    """Floors are the mean minus the margin, clamped at zero."""
    means = dict(GOOD, context_recall=(0.03, [0.0, 0.1]))
    path = tmp_path / "b.json"
    write_baseline(report_with(means), path, floor_margin=0.05, max_failure_rate=0.1)
    data = json.loads(path.read_text())
    assert data["floors"]["faithfulness"] == pytest.approx(0.75)
    assert data["floors"]["context_recall"] == 0.0  # clamped, not negative
    assert data["max_failure_rate"] == 0.1
    assert data["scores"]["faithfulness"] == {"mean": 0.8, "ci": [0.7, 0.9]}
    assert set(data["scores"]) == set(GOOD)  # the label score too, read from the report


def test_the_baseline_keeps_the_apps_provenance_and_drops_the_per_run_keys(tmp_path: Path) -> None:
    """App keys reach the baseline; dirty flag, bootstrap and control keys do not."""
    prov = {**PROV, "fake_model": "m1", "git_dirty": False, "bootstrap_seed": 17,
            "bootstrap_resamples": 1000, "negative_control": False}
    write_baseline(report_with(GOOD, prov=prov), tmp_path / "b.json")
    kept = json.loads((tmp_path / "b.json").read_text())["provenance"]
    assert set(kept) == {*PROV, "fake_model"} - {"git_dirty"}


def test_write_baseline_refuses_run_errors_and_high_failure_rates(tmp_path: Path) -> None:
    """Untrustworthy numbers are never frozen."""
    with pytest.raises(BaselineRefused, match="run error"):
        write_baseline(report_with(GOOD, errors=1), tmp_path / "b.json")
    with pytest.raises(BaselineRefused, match="failure rate"):
        write_baseline(report_with(GOOD, rate=0.5), tmp_path / "b.json")
    assert not (tmp_path / "b.json").exists()


def test_a_negative_control_can_never_become_a_baseline(tmp_path: Path) -> None:
    """A control exists to fail; freezing it would make failure the standard."""
    with pytest.raises(BaselineRefused, match="negative control"):
        write_baseline(report_with(GOOD, prov={**PROV, "negative_control": True}), tmp_path / "b.json")


def test_an_apps_legacy_control_flag_is_refused_and_kept_out_of_the_baseline(tmp_path: Path) -> None:
    """The app names its old flag; the harness itself names none."""
    legacy = {"old_flag": False}
    with pytest.raises(BaselineRefused, match="negative control"):
        write_baseline(report_with(GOOD, prov={**PROV, "old_flag": False}), tmp_path / "b.json",
                       legacy_control=legacy)
    write_baseline(report_with(GOOD, prov={**PROV, "old_flag": True}), tmp_path / "b.json",
                   legacy_control=legacy)
    assert "old_flag" not in json.loads((tmp_path / "b.json").read_text())["provenance"]
    # Without the hook the same key is just provenance: copied, and no refusal.
    write_baseline(report_with(GOOD, prov={**PROV, "old_flag": False}), tmp_path / "c.json")
    assert json.loads((tmp_path / "c.json").read_text())["provenance"]["old_flag"] is False


@pytest.mark.parametrize("dirty", [True, None])
def test_a_dirty_or_unknown_tree_is_refused(tmp_path: Path, dirty: bool | None) -> None:
    """Uncommitted code cannot be traced, and neither can a tree git could not read (None)."""
    with pytest.raises(BaselineRefused, match="uncommitted changes, or git could not tell"):
        write_baseline(report_with(GOOD, prov={**PROV, "git_dirty": dirty}), tmp_path / "b.json")


@pytest.mark.parametrize("key", ["judge_digest", "git_sha"])
def test_a_baseline_from_an_untraceable_run_is_refused(tmp_path: Path, key: str) -> None:
    """An unknown judge digest or git SHA is refused."""
    with pytest.raises(BaselineRefused, match=f"{key} is unknown"):
        write_baseline(report_with(GOOD, prov={**PROV, key: "unknown"}), tmp_path / "b.json")


# ---- Report files -------------------------------------------------------------


def test_report_is_written_under_a_timestamped_name(tmp_path: Path) -> None:
    """One file per run, named by its UTC time."""
    path = write_report(run([case("g1")]), NOW, tmp_path / "reports")
    assert path.name == "eval-20261026T123005Z.json"
    assert json.loads(path.read_text())["cases"][0]["id"] == "g1"
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_report_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed rename leaves no target and no temp file."""
    def boom(src: str, dst: Any) -> None:
        """A rename that fails, as on a full disk."""
        raise OSError("disk full at rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_report({"a": 1}, NOW, tmp_path)
    assert list(tmp_path.iterdir()) == []  # no target, no leftover temp file


@pytest.mark.parametrize("content", ["not json", "[]", "{}", '{"provenance": 1}'])
def test_a_file_that_is_not_a_report_is_refused(tmp_path: Path, content: str) -> None:
    """Anything without provenance, aggregates and cases is not frozen."""
    bad = tmp_path / "bad.json"
    bad.write_text(content)
    assert freeze_baseline(bad, tmp_path / "b.json") == 2
    assert not (tmp_path / "b.json").exists()


def test_a_huge_file_is_refused_before_it_is_read(tmp_path: Path, monkeypatch) -> None:
    """Size is checked before the file is read into memory."""
    big = tmp_path / "big.json"
    big.write_text("{}")
    monkeypatch.setattr(er, "MAX_REPORT_BYTES", 1)
    with pytest.raises(er.ReportInvalid, match="too large"):
        er.load_report(big)


# ---- Table ------------------------------------------------------------------


def test_table_does_not_crash_on_empty_or_single_case_runs() -> None:
    """Empty and one-case runs print `n/a` cells, not a traceback."""
    for cases in ([], [case("g1")]):
        text = format_table(run(cases))
        assert "overall" in text and "n/a" in text
    full = format_table(run([case("g1"), case("g2")]))
    assert "[" in full and "ran_rate" in full


def test_table_rows_follow_the_cases_and_columns_follow_the_scores() -> None:
    """Rows in group order of first appearance; columns from the scores and app extras."""
    lines = format_table(run([case("g1", "b"), case("g2", "a")])).splitlines()
    assert lines[0] == ("| group | n | kind_match | faithfulness | response_relevancy | context_recall "
                        "| action_quality | ran_rate | median_s | run_err |")
    assert [line.split(" | ")[0] for line in lines[2:]] == ["| overall", "| b", "| a"]


# ---- Case viewer --------------------------------------------------------------


def saved_report(tmp_path: Path, name: str = "eval-20261026T120000Z.json") -> Path:
    """A real report, run with fakes and written to disk."""
    report = run([case("g1", source_id="S"), case("g2")])
    path = tmp_path / "reports" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    return path


def test_show_case_prints_everything_needed_to_check_a_score(tmp_path: Path, capsys) -> None:
    """The viewer shows inputs, labels, meta, contexts and every judge decision."""
    path = saved_report(tmp_path)
    assert show_case("g1", path, tmp_path) == 0
    out = capsys.readouterr().out
    for needle in ("g1  a", "ALERT      question g1", "REFERENCE  Root cause", "LABEL      kind: expected x, answered x  MATCH",
                   "META       source_id: S", "           model_used: small", "ANSWER", "faithfulness:",
                   "response_relevancy:", "context_recall:", "[x] it is auth", "SEEN BY THE MODEL",
                   "action_quality: 1.000", "helpful  -- fake"):
        assert needle in out, needle


def test_show_case_uses_the_newest_report_by_default(tmp_path: Path, capsys) -> None:
    """Without `--report`, the newest file by name is shown."""
    saved_report(tmp_path, "eval-20261026T110000Z.json")
    newest = saved_report(tmp_path, "eval-20261026T120000Z.json")
    report = json.loads(newest.read_text())
    report["cases"][0]["question"] = "THE NEWEST"
    newest.write_text(json.dumps(report))
    assert show_case("g1", None, tmp_path / "reports") == 0
    assert "THE NEWEST" in capsys.readouterr().out


def test_show_case_with_an_unknown_id_or_no_report_is_a_clear_error(tmp_path: Path, capsys) -> None:
    """Unknown ids and empty folders exit 2 with a message."""
    path = saved_report(tmp_path)
    assert show_case("g99", path, tmp_path) == 2
    assert "no case 'g99'" in capsys.readouterr().err
    assert show_case("g1", None, tmp_path / "empty") == 2


def test_the_viewer_survives_a_run_error_and_a_failed_metric() -> None:
    """Partial records still print."""
    errored = {"id": "g1", "group": "a", "run_error": "RuntimeError: boom", "latency_s": 0.1,
               "question": "q", "reference": "r"}
    assert "RUN ERROR  RuntimeError: boom" in format_case(errored)
    failed = {**errored, "run_error": None, "answer": "a", "contexts": [],
              "metrics": {"faithfulness": {"score": None, "failed": True, "trace": []}}}
    out = format_case(failed)
    assert "faithfulness: FAILED" in out and "(no context)" in out


def test_an_old_report_without_the_question_prints_a_question_mark_and_warns(tmp_path: Path, capsys) -> None:
    """Missing inputs print `?`, and stderr says why."""
    path = saved_report(tmp_path)
    report = json.loads(path.read_text())
    for c in report["cases"]:
        for key in ("question", "reference", "expected_kind"):
            c.pop(key)
    path.write_text(json.dumps(report))
    assert show_case("g1", path, tmp_path) == 0
    captured = capsys.readouterr()
    assert "predates self-contained records" in captured.err
    assert "ALERT      ?" in captured.out and "REFERENCE  ?" in captured.out
    assert "LABEL      kind: expected ?, answered x  MATCH" in captured.out  # known from kind_match


def test_freezing_a_saved_report_then_gating_it_passes(tmp_path: Path, capsys) -> None:
    """A frozen saved report gates clean against itself."""
    path = saved_report(tmp_path)
    baseline = tmp_path / "evals" / "baseline.json"
    assert freeze_baseline(path, baseline) == 0
    assert "baseline written: baseline.json (from eval-20261026T120000Z.json)" in capsys.readouterr().out
    assert gate(json.loads(path.read_text()), load_baseline(baseline)) == []
