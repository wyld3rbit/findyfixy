"""Fast tests for the answer eval. Fakes only: no model, no database, no network.

The fakes stand in for the three slow things: the graph (a model), the
retriever (a database) and the judge (a model). Everything else is real code.
"""

import copy
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from app import eval_report as er
from app.eval_answers import (
    holdout,
    main,
    make_holdout_recall,
    render_answer,
    render_contexts,
    run_eval,
)
from app.eval_provenance import build_provenance
from app.eval_report import (
    BaselineMissing,
    BaselineRefused,
    format_table,
    gate,
    load_baseline,
    write_baseline,
    write_report,
)
from app.eval_stats import aggregate, bootstrap_ci
from app.golden import GoldenCase
from app.settings import MissingEnvironment, require_env
from app.judge import JudgeError
from app.metrics import PROMPT_VERSIONS, ActionRating, Claims, Questions, Verdict
from app.triage import MAX_PRIOR_CHARS, Prior, Triage, format_prior

NOW = datetime(2026, 10, 26, 12, 30, 5, tzinfo=timezone.utc)


def prior(incident_id: str, summary: str = "s", cause: str = "c") -> Prior:
    return Prior(incident_id, "auth", "sev2", summary, cause, "r")


def case(
    cid: str, source: str, twin: str | None = None, severity: str = "sev2"
) -> GoldenCase:
    return GoldenCase(
        id=cid,
        source_id=source,
        group="twin" if twin else "no_twin",
        twin_id=twin,
        question=f"question {cid}",
        reference="Root cause: a. Remediation: b.",
        expected_severity=severity,  # type: ignore[arg-type]
    )


class FakeCandidates:
    """A retriever that always ranks the source first: the worst case for holdout."""

    def __init__(self, ranked: list[Prior]) -> None:
        self.ranked = ranked
        self.asked: list[int] = []

    def __call__(self, question: str, n: int) -> list[Prior]:
        self.asked.append(n)
        return self.ranked[:n]


class FakeGraph:
    """Calls recall like the real recall node, then returns a canned triage."""

    def __init__(self, recall: Any, severity: str, fail: bool) -> None:
        self.recall, self.severity, self.fail = recall, severity, fail

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.recall(state["question"])
        if self.fail:
            raise RuntimeError("model never produced a valid Triage")
        triage = Triage(severity=self.severity, component="auth", next_action="roll back", confidence=0.9)  # type: ignore[arg-type]
        return {"triage": triage, "model_used": "small"}


def graph_factory(severity: str = "sev2", fail_on: set[str] | None = None):
    """make_graph for tests. A graph fails when its question's id is in `fail_on`."""
    questions: list[str] = []

    def make_graph(recall: Any) -> FakeGraph:
        def spy(question: str):
            questions.append(question)
            return recall(question)

        graph = FakeGraph(spy, severity, False)
        original = graph.invoke

        def invoke(state: dict[str, Any]) -> dict[str, Any]:
            graph.fail = any(state["question"].endswith(f" {i}") for i in (fail_on or set()))
            return original(state)

        graph.invoke = invoke  # type: ignore[method-assign]
        return graph

    return make_graph


class FakeJudge:
    """Answers every question. Records prompts. Fails when told to."""

    def __init__(self, supported: bool = True, fail: bool = False) -> None:
        self.supported, self.fail, self.prompts = supported, fail, []

    def ask(self, prompt: str, schema: type[BaseModel]) -> Any:
        self.prompts.append(prompt)
        if self.fail:
            raise JudgeError("fake failure")
        if schema is Claims:
            return Claims(claims=["it is auth"])
        if schema is Questions:
            return Questions(questions=["q1", "q2", "q3"])
        if schema is ActionRating:
            return ActionRating(rating="helpful" if self.supported else "unhelpful", reason="fake")
        return Verdict(supported=self.supported)


def embed(texts: Any) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]


PROV = {"judge_digest": "d1", "prompt_versions": {"m": "h"}, "golden_sha256": "g1", "git_sha": "abc1234"}


def run(cases, candidates=None, judge=None, make_graph=None, **kw):
    return run_eval(
        cases,
        make_graph=make_graph or graph_factory(),
        candidates=candidates or FakeCandidates([prior("S"), prior("T"), prior("X"), prior("Y")]),
        judge=judge or FakeJudge(),
        embed=embed,
        provenance=kw.pop("provenance", PROV),
        **kw,
    )


# ---- Holdout ----------------------------------------------------------------


def test_holdout_drops_the_source_keeps_the_twin_and_trims_to_k() -> None:
    cands = [prior("S"), prior("T"), prior("X"), prior("Y")]
    kept = holdout(cands, "S", 3)
    assert [p.incident_id for p in kept] == ["T", "X", "Y"]


def test_holdout_with_source_absent_still_trims_to_k() -> None:
    assert [p.incident_id for p in holdout([prior("A"), prior("B"), prior("C")], "S", 2)] == ["A", "B"]


def test_recall_asks_for_k_plus_one_and_fills_the_sink() -> None:
    cands = FakeCandidates([prior("S"), prior("T"), prior("X"), prior("Y")])
    sink: list[Prior] = []
    recall = make_holdout_recall(cands, "S", 3, sink)
    out = recall("q")
    assert cands.asked == [4]
    assert [p.incident_id for p in out] == ["T", "X", "Y"]
    assert sink == out


def test_the_source_never_reaches_the_graph_even_when_ranked_first() -> None:
    report = run([case("g1", "S", "T")])
    rec = report["cases"][0]
    assert "S" not in rec["recalled_ids"]
    assert rec["recalled_ids"] == ["T", "X", "Y"]
    assert rec["twin_recalled"] is True


# ---- What the judge sees ----------------------------------------------------


def test_contexts_equal_what_the_prompt_used_and_are_capped() -> None:
    long = prior("L", summary="x" * 1000)
    ctx = render_contexts([prior("A"), long])
    assert ctx[0] == format_prior([prior("A")])
    assert ctx[1] == format_prior([long])
    assert len(ctx[1]) == MAX_PRIOR_CHARS


def test_judge_prompts_contain_the_recalled_context_and_never_the_source() -> None:
    judge = FakeJudge()
    run([case("g1", "S", "T")], judge=judge,
        candidates=FakeCandidates([prior("S", summary="SOURCE-TEXT"), prior("T", summary="TWIN-TEXT"),
                                   prior("X"), prior("Y")]))
    joined = "\n".join(judge.prompts)
    assert "TWIN-TEXT" in joined
    assert "SOURCE-TEXT" not in joined


def test_render_answer_is_plain_text_without_confidence() -> None:
    t = Triage(severity="sev1", component="db", next_action="failover", confidence=0.5)
    text = render_answer(t)
    assert text == "Severity sev1. Component db. Next action: failover"


# ---- Run errors -------------------------------------------------------------


def test_a_graph_error_is_recorded_and_the_run_continues() -> None:
    cases = [case("g1", "S1"), case("g2", "S2"), case("g3", "S3")]
    report = run(cases, make_graph=graph_factory(fail_on={"g2"}))
    errors = [c["run_error"] for c in report["cases"]]
    assert errors[0] is None and errors[2] is None
    assert "RuntimeError" in errors[1] and "never produced" in errors[1]
    assert report["aggregates"]["overall"]["n_run_errors"] == 1
    assert report["aggregates"]["overall"]["scores"]["severity_match"]["n_scored"] == 2


def test_a_metric_that_raises_is_a_failed_metric_not_an_abort() -> None:
    def bad_embed(texts: Any) -> list[list[float]]:
        raise OSError("embedder down")

    report = run_eval([case("g1", "S")], make_graph=graph_factory(),
                      candidates=FakeCandidates([prior("A")] * 4), judge=FakeJudge(),
                      embed=bad_embed, provenance=PROV)
    assert report["cases"][0]["metrics"]["response_relevancy"] == {
        "score": None, "failed": True, "error": "OSError: embedder down", "trace": []}
    assert report["cases"][0]["run_error"] is None


# ---- Aggregation ------------------------------------------------------------


def rec(group: str, sev_ok: bool, faith: float | None, twin: str | None = None, hit: bool = False,
        latency: float = 1.0, error: str | None = None) -> dict[str, Any]:
    def m(score: float | None) -> dict[str, Any]:
        return {"score": score, "failed": score is None}

    return {"id": "x", "group": group, "source_id": "s", "twin_id": twin, "recalled_ids": [],
            "twin_recalled": hit, "model_used": "small", "latency_s": latency, "severity": "sev2",
            "severity_match": sev_ok, "run_error": error,
            "metrics": {} if error else {"faithfulness": m(faith), "response_relevancy": m(0.8),
                                         "context_recall": m(0.4), "action_quality": m(0.5)}}


def test_aggregate_by_hand_excludes_failures_and_never_scores_them_zero() -> None:
    records = [rec("twin", True, 1.0, "t", True), rec("twin", False, 0.5, "t", False),
               rec("no_twin", True, None), rec("no_twin", True, 0.0)]
    agg = aggregate(records)
    faith = agg["scores"]["faithfulness"]
    assert faith["mean"] == pytest.approx(0.5)  # (1.0 + 0.5 + 0.0) / 3, the failure is out
    assert faith["n_scored"] == 3 and faith["n_failed"] == 1
    assert faith["failure_rate"] == pytest.approx(0.25)
    assert agg["scores"]["severity_match"]["mean"] == pytest.approx(0.75)
    assert agg["twin_recall_rate"] == pytest.approx(0.5)
    assert agg["scores"]["context_recall"]["mean"] == pytest.approx(0.4)


def test_run_eval_splits_by_group() -> None:
    cases = [case("g1", "S1", "T1", "sev2"), case("g2", "S2", None, "sev1")]
    report = run(cases)  # the fake graph always says sev2
    agg = report["aggregates"]
    assert agg["twin"]["scores"]["severity_match"]["mean"] == 1.0
    assert agg["no_twin"]["scores"]["severity_match"]["mean"] == 0.0
    assert agg["overall"]["scores"]["severity_match"]["mean"] == 0.5
    assert agg["no_twin"]["twin_recall_rate"] is None


def test_twin_recall_rate_counts_only_twin_cases_where_the_twin_came_back() -> None:
    cands = FakeCandidates([prior("S"), prior("T1"), prior("X"), prior("Y")])
    cases = [case("g1", "S", "T1"), case("g2", "S", "MISSING")]
    report = run(cases, candidates=cands)
    assert report["aggregates"]["twin"]["twin_recall_rate"] == 0.5


def test_failed_judge_metrics_are_counted_in_the_report() -> None:
    report = run([case("g1", "S")], judge=FakeJudge(fail=True))
    s = report["aggregates"]["overall"]["scores"]["faithfulness"]
    assert s["mean"] is None and s["n_failed"] == 1 and s["failure_rate"] == 1.0


# ---- Bootstrap --------------------------------------------------------------


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first, second = bootstrap_ci(values), bootstrap_ci(values)
    assert first == second
    low, high = first  # type: ignore[misc]
    assert low <= sum(values) / len(values) <= high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_depends_on_the_seed() -> None:
    values = [0.1, 0.9, 0.4, 0.7, 0.3, 0.8]
    assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)


def test_n0_and_n1_have_no_interval_and_do_not_crash() -> None:
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([0.5]) is None
    empty = run([])
    assert empty["aggregates"]["overall"]["scores"]["faithfulness"]["mean"] is None
    one = run([case("g1", "S")])
    stat = one["aggregates"]["overall"]["scores"]["faithfulness"]
    assert stat["mean"] == 1.0 and stat["ci"] is None


# ---- Provenance and report file --------------------------------------------


def provenance(**over: Any) -> dict[str, Any]:
    args: dict[str, Any] = dict(
        judge_model="qwen2.5:7b", digest=lambda m: "dig-" + m, golden_sha256="gold",
        small_model="s", large_model="l", git=lambda: ("abc1234", False), now=lambda: NOW)
    args.update(over)
    return build_provenance(**args)


def test_provenance_has_every_key() -> None:
    p = provenance()
    assert set(p) >= {"judge_model", "judge_digest", "prompt_versions", "golden_sha256",
                      "small_model", "large_model", "threshold", "recall_k", "git_sha",
                      "git_dirty", "timestamp_utc", "bootstrap_seed"}
    assert p["judge_digest"] == "dig-qwen2.5:7b"
    assert p["prompt_versions"] == PROMPT_VERSIONS
    assert p["timestamp_utc"] == "2026-10-26T12:30:05+00:00"
    assert p["git_sha"] == "abc1234" and p["git_dirty"] is False


def test_report_is_written_under_a_timestamped_name(tmp_path: Path) -> None:
    report = run([case("g1", "S")], provenance=provenance())
    path = write_report(report, NOW, tmp_path / "reports")
    assert path.name == "eval-20261026T123005Z.json"
    assert json.loads(path.read_text())["provenance"]["small_model"] == "s"
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_report_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(src: str, dst: Any) -> None:
        raise OSError("disk full at rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_report({"a": 1}, NOW, tmp_path)
    assert list(tmp_path.iterdir()) == []  # no target, no leftover temp file


# ---- Gate -------------------------------------------------------------------


def stat(mean: float | None, ci: list[float] | None = None, rate: float = 0.0) -> dict[str, Any]:
    return {"mean": mean, "ci": ci, "n_scored": 30, "n_failed": 0, "failure_rate": rate}


def report_with(means: dict[str, tuple[float, list[float]]], rate: float = 0.0, errors: int = 0,
                prov: dict[str, Any] | None = None) -> dict[str, Any]:
    scores = {n: stat(m, ci, rate) for n, (m, ci) in means.items()}
    cases = [{"id": f"g{i}", "run_error": "boom"} for i in range(errors)]
    overall = {"n": 30, "n_run_errors": errors, "scores": scores,
               "twin_recall_rate": 0.9, "median_latency_s": 12.0}
    return {"provenance": prov or dict(PROV),
            "aggregates": {"overall": overall, "twin": overall, "no_twin": overall},
            "cases": cases}


GOOD = {"severity_match": (0.8, [0.7, 0.9]), "faithfulness": (0.8, [0.7, 0.9]),
        "response_relevancy": (0.8, [0.7, 0.9]), "context_recall": (0.8, [0.7, 0.9]),
        "action_quality": (0.8, [0.7, 0.9])}


def baseline_for(report: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "evals" / "baseline.json"
    write_baseline(report, path)
    return load_baseline(path)


def test_a_clean_report_passes(tmp_path: Path) -> None:
    report = report_with(GOOD)
    assert gate(report, baseline_for(report, tmp_path)) == []


def test_floor_breach_fails(tmp_path: Path) -> None:
    base = baseline_for(report_with(GOOD), tmp_path)
    low = dict(GOOD, faithfulness=(0.6, [0.5, 0.95]))  # baseline 0.8 is inside the CI: not a regression
    problems = gate(report_with(low), base)
    assert len(problems) == 1 and "faithfulness" in problems[0] and "floor" in problems[0]


def test_regression_needs_the_baseline_to_sit_above_the_interval(tmp_path: Path) -> None:
    base = baseline_for(report_with(GOOD), tmp_path)
    base["floors"] = {}  # isolate rule (b)
    noisy = dict(GOOD, context_recall=(0.75, [0.6, 0.9]))  # dip inside the CI
    assert gate(report_with(noisy), base) == []
    clear = dict(GOOD, context_recall=(0.75, [0.65, 0.78]))  # baseline 0.8 above 0.78
    problems = gate(report_with(clear), base)
    assert len(problems) == 1 and "regression" in problems[0] and "context_recall" in problems[0]


def test_an_improvement_is_not_a_regression(tmp_path: Path) -> None:
    base = baseline_for(report_with(GOOD), tmp_path)
    better = dict(GOOD, faithfulness=(0.95, [0.9, 0.99]))
    assert gate(report_with(better), base) == []


def test_failure_rate_above_the_cap_fails(tmp_path: Path) -> None:
    base = baseline_for(report_with(GOOD), tmp_path)
    problems = gate(report_with(GOOD, rate=0.2), base)
    assert problems and all("failure rate" in p for p in problems)


def test_any_run_error_fails_and_names_the_case(tmp_path: Path) -> None:
    base = baseline_for(report_with(GOOD), tmp_path)
    problems = gate(report_with(GOOD, errors=1), base)
    assert len(problems) == 1 and "run error" in problems[0] and "g0: boom" in problems[0]


@pytest.mark.parametrize("key,value", [("judge_digest", "d2"), ("prompt_versions", {"m": "h2"}),
                                       ("golden_sha256", "g2")])
def test_incomparable_baseline_says_rebaseline_and_skips_the_comparison(
    tmp_path: Path, key: str, value: Any
) -> None:
    base = baseline_for(report_with(GOOD), tmp_path)
    terrible = {n: (0.1, [0.0, 0.2]) for n in GOOD}  # would breach floors and regress
    prov = dict(PROV, **{key: value})
    problems = gate(report_with(terrible, prov=prov), base)
    assert any(key in p for p in problems)
    assert any("re-baseline deliberately" in p for p in problems)
    assert not any("floor" in p or "regression" in p for p in problems)


def test_incomparable_baseline_still_runs_the_run_error_and_failure_checks(tmp_path: Path) -> None:
    base = baseline_for(report_with(GOOD), tmp_path)
    prov = dict(PROV, golden_sha256="other")
    problems = gate(report_with(GOOD, rate=0.5, errors=2, prov=prov), base)
    assert any("run error" in p for p in problems)
    assert any("failure rate" in p for p in problems)
    assert any("re-baseline" in p for p in problems)


def test_missing_baseline_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(BaselineMissing, match="--write-baseline"):
        load_baseline(tmp_path / "nope.json")


# ---- Baseline writer --------------------------------------------------------


def test_write_baseline_sets_floors_from_the_overall_means(tmp_path: Path) -> None:
    means = dict(GOOD, context_recall=(0.03, [0.0, 0.1]))
    path = tmp_path / "b.json"
    write_baseline(report_with(means), path, floor_margin=0.05, max_failure_rate=0.1)
    data = json.loads(path.read_text())
    assert data["floors"]["faithfulness"] == pytest.approx(0.75)
    assert data["floors"]["context_recall"] == 0.0  # clamped, not negative
    assert data["max_failure_rate"] == 0.1
    assert data["scores"]["faithfulness"] == {"mean": 0.8, "ci": [0.7, 0.9]}
    assert set(data["provenance"]) >= {"judge_digest", "prompt_versions", "golden_sha256"}


def test_write_baseline_refuses_run_errors_and_high_failure_rates(tmp_path: Path) -> None:
    with pytest.raises(BaselineRefused, match="run error"):
        write_baseline(report_with(GOOD, errors=1), tmp_path / "b.json")
    with pytest.raises(BaselineRefused, match="failure rate"):
        write_baseline(report_with(GOOD, rate=0.5), tmp_path / "b.json")
    assert not (tmp_path / "b.json").exists()


# ---- CLI --------------------------------------------------------------------


def runner_for(report: dict[str, Any]):
    return lambda progress: copy.deepcopy(report)


def test_main_write_baseline_refuses_when_the_run_had_errors(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "evals" / "baseline.json"
    code = main(["--write-baseline"], runner=runner_for(report_with(GOOD, errors=1)),
                now=lambda: NOW, reports_dir=tmp_path / "r", baseline_path=baseline)
    assert code == 1 and not baseline.exists()
    assert "baseline refused" in capsys.readouterr().err


def test_main_write_baseline_then_gate_passes(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "evals" / "baseline.json"
    report = run([case("g1", "S", "T")], provenance=provenance())
    kw = dict(runner=runner_for(report), now=lambda: NOW, reports_dir=tmp_path / "r",
              baseline_path=baseline)
    assert main(["--write-baseline"], **kw) == 0 and baseline.exists()
    assert main(["--gate"], **kw) == 0
    assert "gate: pass" in capsys.readouterr().out


def test_main_gate_exits_1_on_problems_and_2_without_a_baseline(tmp_path: Path, capsys) -> None:
    kw = dict(runner=runner_for(report_with(GOOD, errors=1)), now=lambda: NOW,
              reports_dir=tmp_path / "r", baseline_path=tmp_path / "none.json")
    assert main(["--gate"], **kw) == 2
    assert "no baseline" in capsys.readouterr().err
    base = baseline_for(report_with(GOOD), tmp_path)
    (tmp_path / "none.json").write_text(json.dumps(base))
    assert main(["--gate"], **kw) == 1
    assert "GATE FAIL" in capsys.readouterr().out


def test_table_does_not_crash_on_empty_or_single_case_runs() -> None:
    for cases in ([], [case("g1", "S", "T")]):
        text = format_table(run(cases))
        assert "overall" in text and "n/a" in text
    full = format_table(run([case("g1", "S", "T"), case("g2", "S2", "T")]))
    assert "[" in full and "twin_recall" in full


def test_no_recall_flag_reaches_the_runner_and_normal_runs_do_not_pass_it(tmp_path: Path) -> None:
    seen: list[dict] = []

    def runner(progress, **kwargs):
        seen.append(kwargs)
        return report_with(GOOD)

    kw = dict(now=lambda: NOW, reports_dir=tmp_path / "r", baseline_path=tmp_path / "b.json")
    main([], runner=runner, **kw)
    main(["--no-recall"], runner=runner, **kw)
    assert seen == [{}, {"recall_enabled": False}]


def test_a_no_recall_run_can_never_become_a_baseline(tmp_path: Path, capsys) -> None:
    report = report_with(GOOD, prov={**PROV, "recall_enabled": False})
    baseline = tmp_path / "evals" / "baseline.json"
    code = main(["--write-baseline"], runner=runner_for(report), now=lambda: NOW,
                reports_dir=tmp_path / "r", baseline_path=baseline)
    assert code == 1 and not baseline.exists()
    assert "negative control" in capsys.readouterr().err


def test_provenance_records_whether_recall_was_enabled() -> None:
    prov = build_provenance(
        judge_model="j", digest=lambda m: "d", golden_sha256="g", small_model="s",
        large_model="l", git=lambda: ("abc", False), now=lambda: NOW, recall_enabled=False,
    )
    assert prov["recall_enabled"] is False


# ---- Parallel workers -------------------------------------------------------


def many_cases(n: int) -> list[GoldenCase]:
    return [case(f"g{i}", f"S{i}", twin=f"T{i}" if i % 2 else None) for i in range(n)]


def test_parallel_and_serial_runs_give_the_same_records_in_the_same_order() -> None:
    cases = many_cases(8)
    serial = run(cases, workers=1)
    parallel = run(cases, workers=4)
    strip = lambda r: [{k: v for k, v in c.items() if k != "latency_s"} for c in r["cases"]]  # noqa: E731
    assert strip(parallel) == strip(serial)
    assert [c["id"] for c in parallel["cases"]] == [c.id for c in cases]
    assert parallel["aggregates"]["overall"]["scores"] == serial["aggregates"]["overall"]["scores"]


def test_workers_really_overlap_cases() -> None:
    """Two cases wait for each other at a barrier. Only concurrent execution passes it."""
    barrier = threading.Barrier(2, timeout=5)
    plain = graph_factory()

    def make_graph(recall: Any):
        graph = plain(recall)
        original = graph.invoke

        def invoke(state: dict[str, Any]) -> dict[str, Any]:
            barrier.wait()
            return original(state)

        graph.invoke = invoke  # type: ignore[method-assign]
        return graph

    report = run(many_cases(2), make_graph=make_graph, workers=2)
    assert report["aggregates"]["overall"]["n_run_errors"] == 0


def test_one_worker_does_not_overlap_cases() -> None:
    """The same barrier with one worker is never released: both cases record a run error."""
    barrier = threading.Barrier(2, timeout=0.2)
    plain = graph_factory()

    def make_graph(recall: Any):
        graph = plain(recall)
        original = graph.invoke

        def invoke(state: dict[str, Any]) -> dict[str, Any]:
            barrier.wait()
            return original(state)

        graph.invoke = invoke  # type: ignore[method-assign]
        return graph

    report = run(many_cases(2), make_graph=make_graph, workers=1)
    assert report["aggregates"]["overall"]["n_run_errors"] == 2


def test_each_case_keeps_its_own_holdout_and_context_under_parallel_load() -> None:
    """Shared state would show up as one case seeing another's recalled incidents."""
    cases = many_cases(12)
    ranked = [prior(f"S{i}") for i in range(12)] + [prior("T1"), prior("X")]
    judge = FakeJudge()
    report = run(cases, candidates=FakeCandidates(ranked), judge=judge, workers=4)
    for record in report["cases"]:
        assert record["source_id"] not in record["recalled_ids"]
        assert len(record["recalled_ids"]) == 3


def test_progress_counts_each_case_once_even_in_parallel() -> None:
    lines: list[str] = []
    run(many_cases(10), workers=4, progress=lines.append)
    assert sorted(int(line.split("/")[0][1:]) for line in lines) == list(range(1, 11))


@pytest.mark.parametrize("bad", [0, -1, 5, 100])
def test_workers_outside_the_cap_are_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="workers must be between"):
        run(many_cases(1), workers=bad)


def test_the_report_records_workers_and_throughput() -> None:
    ticks = iter([0.0, 12.0])
    report = run(many_cases(6), workers=2, timer=lambda: next(ticks, 12.0))
    assert report["timing"] == {"workers": 2, "wall_s": 12.0, "cases_per_min": 30.0}


def test_main_passes_workers_only_when_above_one_and_rejects_values_over_the_cap(tmp_path: Path) -> None:
    seen: list[dict] = []

    def runner(progress, **kwargs):
        seen.append(kwargs)
        return report_with(GOOD)

    kw = dict(now=lambda: NOW, reports_dir=tmp_path / "r", baseline_path=tmp_path / "b.json")
    main([], runner=runner, **kw)
    main(["--workers", "3"], runner=runner, **kw)
    assert seen == [{}, {"workers": 3}]
    with pytest.raises(SystemExit):
        main(["--workers", "9"], runner=runner, **kw)


# ---- Audit trail, case viewer, freezing from a saved report -------------------


def saved_report(tmp_path: Path, **prov) -> Path:
    """A real report, run with fakes and written to disk."""
    report = run([case("g1", "S", "T"), case("g2", "S2")], provenance=provenance(**prov))
    path = tmp_path / "reports" / "eval-20261026T120000Z.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    return path


def test_the_case_record_keeps_the_answer_the_contexts_and_the_judge_decisions() -> None:
    report = run([case("g1", "S", "T")])
    record = report["cases"][0]
    assert record["answer"].startswith("Severity sev2")
    assert record["contexts"] and all(isinstance(c, str) for c in record["contexts"])
    assert record["metrics"]["faithfulness"]["trace"][0]["claim"] == "it is auth"
    assert len(record["metrics"]["response_relevancy"]["trace"]) == 3


def test_show_case_prints_everything_needed_to_check_a_score(tmp_path: Path, capsys, monkeypatch) -> None:
    path = saved_report(tmp_path)
    monkeypatch.setattr(er, "load_golden", lambda: [case("g1", "S", "T")])
    assert main(["--show-case", "g1", "--report", str(path)]) == 0
    out = capsys.readouterr().out
    for needle in ("g1", "ALERT", "REFERENCE", "RECALLED", "ANSWER", "faithfulness:",
                   "response_relevancy:", "context_recall:", "[x] it is auth", "SEEN BY THE MODEL"):
        assert needle in out, needle


def test_show_case_uses_the_newest_report_by_default(tmp_path: Path, capsys, monkeypatch) -> None:
    saved_report(tmp_path)
    monkeypatch.setattr(er, "load_golden", lambda: [case("g1", "S", "T")])
    assert main(["--show-case", "g1"], reports_dir=tmp_path / "reports") == 0
    assert "g1" in capsys.readouterr().out


def test_show_case_with_an_unknown_id_or_no_report_is_a_clear_error(tmp_path: Path, capsys) -> None:
    path = saved_report(tmp_path)
    assert main(["--show-case", "g99", "--report", str(path)]) == 2
    assert "no case 'g99'" in capsys.readouterr().err
    assert main(["--show-case", "g1"], reports_dir=tmp_path / "empty") == 2


def test_show_case_survives_a_run_error_and_a_failed_metric() -> None:
    errored = {"id": "g1", "group": "twin", "source_id": "S", "twin_id": "T", "recalled_ids": [],
               "run_error": "RuntimeError: boom", "latency_s": 0.1}
    assert "RUN ERROR  RuntimeError: boom" in er.format_case(errored, case("g1", "S", "T"))
    failed = {**errored, "run_error": None, "severity": "sev2", "severity_match": True, "answer": "a",
              "contexts": [], "metrics": {"faithfulness": {"score": None, "failed": True, "trace": []}}}
    out = er.format_case(failed, None)
    assert "faithfulness: FAILED" in out and "nothing was recalled" in out


@pytest.mark.parametrize("content", ["not json", "[]", "{}", '{"provenance": 1}'])
def test_a_file_that_is_not_a_report_is_refused(tmp_path: Path, content: str, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(content)
    assert main(["--baseline-from", str(bad)], baseline_path=tmp_path / "b.json") == 2
    assert not (tmp_path / "b.json").exists()


def test_a_huge_file_is_refused_before_it_is_read(tmp_path: Path, monkeypatch) -> None:
    big = tmp_path / "big.json"
    big.write_text("{}")
    monkeypatch.setattr(er, "MAX_REPORT_BYTES", 1)
    with pytest.raises(er.ReportInvalid, match="too large"):
        er.load_report(big)


def test_baseline_from_a_saved_report_runs_nothing_and_then_the_gate_passes(tmp_path: Path) -> None:
    path = saved_report(tmp_path)
    baseline = tmp_path / "evals" / "baseline.json"

    def runner(progress, **kwargs):
        raise AssertionError("freezing a saved report must not run the evaluation")

    assert main(["--baseline-from", str(path)], runner=runner, baseline_path=baseline) == 0
    assert baseline.exists()
    report = json.loads(path.read_text())
    assert gate(report, load_baseline(baseline)) == []


def test_baseline_from_refuses_a_dirty_tree_and_a_no_recall_run(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "evals" / "baseline.json"
    dirty = saved_report(tmp_path, git=lambda: ("abc1234", True))
    assert main(["--baseline-from", str(dirty)], baseline_path=baseline) == 1
    assert "uncommitted changes" in capsys.readouterr().err and not baseline.exists()
    other = tmp_path / "other"
    other.mkdir()
    no_recall = saved_report(other, recall_enabled=False)
    assert main(["--baseline-from", str(no_recall)], baseline_path=baseline) == 1
    assert "negative control" in capsys.readouterr().err


# ---- Fixes from the independent review ---------------------------------------


def test_a_bug_in_a_metric_is_not_hidden_as_a_judge_failure() -> None:
    """Expected faults become failed results; a KeyError is our bug and must stop the run."""
    def buggy_embed(texts: Any) -> list[list[float]]:
        raise KeyError("oops")

    with pytest.raises(KeyError):
        run_eval([case("g1", "S")], make_graph=graph_factory(),
                 candidates=FakeCandidates([prior("A")] * 4), judge=FakeJudge(),
                 embed=buggy_embed, provenance=PROV)


def test_a_failed_gate_never_rewrites_the_baseline(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "evals" / "baseline.json"
    good = run([case("g1", "S", "T")], provenance=provenance())
    write_baseline(good, baseline)
    before = baseline.read_bytes()
    worse = run([case("g1", "S", "T", severity="sev1")], provenance=provenance())  # answer sev2 vs expected sev1
    code = main(["--gate", "--write-baseline"], runner=runner_for(worse), now=lambda: NOW,
                reports_dir=tmp_path / "r", baseline_path=baseline)
    assert code == 1
    assert baseline.read_bytes() == before
    assert "gate failed in this same run" in capsys.readouterr().err


def test_a_saved_report_shows_what_was_scored_even_if_the_golden_set_changed(tmp_path: Path, capsys, monkeypatch) -> None:
    path = saved_report(tmp_path)
    changed = case("g1", "S", "T")
    changed = changed.model_copy(update={"question": "A DIFFERENT QUESTION", "reference": "A DIFFERENT REFERENCE"})
    monkeypatch.setattr(er, "load_golden", lambda: [changed])
    assert main(["--show-case", "g1", "--report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "question g1" in out and "DIFFERENT" not in out


def test_an_old_report_without_the_question_warns_when_the_golden_set_changed(tmp_path: Path, capsys, monkeypatch) -> None:
    path = saved_report(tmp_path)
    report = json.loads(path.read_text())
    for c in report["cases"]:
        for key in ("question", "reference", "expected_severity"):
            c.pop(key)
    report["provenance"]["golden_sha256"] = "an-old-hash"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(er, "load_golden", lambda: [case("g1", "S", "T")])
    assert main(["--show-case", "g1", "--report", str(path)]) == 0
    assert "golden set changed" in capsys.readouterr().err


def test_a_missing_golden_file_does_not_crash_the_viewer_for_an_old_report(tmp_path: Path, capsys, monkeypatch) -> None:
    path = saved_report(tmp_path)
    report = json.loads(path.read_text())
    for c in report["cases"]:
        c.pop("question")
    path.write_text(json.dumps(report))

    def gone():
        raise FileNotFoundError("golden.jsonl")

    monkeypatch.setattr(er, "load_golden", gone)
    assert main(["--show-case", "g1", "--report", str(path)]) == 0
    assert "golden file is missing" in capsys.readouterr().err


@pytest.mark.parametrize("key", ["judge_digest", "git_sha"])
def test_a_baseline_from_an_untraceable_run_is_refused(tmp_path: Path, key: str) -> None:
    report = report_with(GOOD, prov={**PROV, key: "unknown"})
    with pytest.raises(er.BaselineRefused, match=f"{key} is unknown"):
        write_baseline(report, tmp_path / "b.json")


def test_two_unknown_judge_digests_are_not_the_same_judge(tmp_path: Path) -> None:
    prov = {**PROV, "judge_digest": "unknown"}
    report = report_with(GOOD, prov=prov)
    baseline = baseline_for(report_with(GOOD), tmp_path)
    baseline["provenance"]["judge_digest"] = "unknown"
    problems = gate(report, baseline)
    assert any("judge_digest is unknown" in p for p in problems)


def test_only_a_missing_environment_variable_says_set_the_variable(tmp_path: Path, capsys) -> None:
    kw = dict(now=lambda: NOW, reports_dir=tmp_path / "r", baseline_path=tmp_path / "b.json")

    def missing(progress, **kwargs):
        raise MissingEnvironment("SMALL_MODEL")

    assert main([], runner=missing, **kw) == 2
    assert "set the environment variable SMALL_MODEL" in capsys.readouterr().err

    def bug(progress, **kwargs):
        raise KeyError("some_record_key")

    with pytest.raises(KeyError):
        main([], runner=bug, **kw)


def test_require_env_names_the_missing_variable(monkeypatch) -> None:
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(MissingEnvironment, match="NOT_SET_ANYWHERE"):
        require_env("NOT_SET_ANYWHERE")
    monkeypatch.setenv("SET_HERE", "x")
    assert require_env("SET_HERE") == "x"


def test_the_case_record_is_self_contained() -> None:
    record = run([case("g1", "S", "T", severity="sev1")])["cases"][0]
    assert record["question"] == "question g1"
    assert record["reference"].startswith("Root cause")
    assert record["expected_severity"] == "sev1"


# ---- action quality in the runner and the viewer -------------------------------


def test_the_runner_scores_action_quality_and_keeps_the_judges_reasoning() -> None:
    record = run([case("g1", "S", "T")])["cases"][0]
    m = record["metrics"]["action_quality"]
    assert m["score"] == 1.0 and m["trace"][0]["rating"] == "helpful"
    assert m["trace"][0]["action"] == "roll back"  # the action, not the whole answer


def test_the_viewer_prints_the_action_rating() -> None:
    record = run([case("g1", "S", "T")])["cases"][0]
    text = er.format_case(record, None)
    assert "action_quality: 1.000" in text and "helpful  -- fake" in text


def test_the_gate_fails_when_action_quality_drops_below_its_floor(tmp_path: Path) -> None:
    good = report_with(GOOD)
    baseline = baseline_for(good, tmp_path)
    worse = report_with({**GOOD, "action_quality": (0.3, [0.2, 0.4])})
    assert any("action_quality" in p for p in gate(worse, baseline))

