"""Fast tests for FindyFixy as a system under test. Fakes only: no model, no database.

The fakes stand in for the three slow things: the graph (a model), the
retriever (a database) and the judge (a model). The adapter and the harness are
real code, so these tests also prove that a FindyFixy run still writes the
record and report schema every saved report and the baseline use.

The harness's own rules (gate, baseline, table, workers) are tested in
tests/test_harness_eval_*.py with a fake system that is not FindyFixy.
"""

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from app import eval_answers
from app.eval_adapter import (
    FindyFixy,
    holdout,
    make_holdout_recall,
    render_answer,
    render_contexts,
    to_case,
    twin_recall_rate,
)
from app.golden import GoldenCase
from app.triage import MAX_PRIOR_CHARS, Prior, Triage, format_prior
from harness.eval import cli
from harness.eval.cli import Session
from harness.eval.runner import run_eval
from tests.harness_fakes import NOW, PROV, FakeJudge, embed

# The record keys every FindyFixy report has carried since Day 5, plus `labels`
# (the label names, stored explicitly since 2026-11-09).
RECORD_KEYS = {
    "id", "group", "source_id", "twin_id", "recalled_ids", "twin_recalled", "model_used", "latency_s",
    "severity", "severity_match", "question", "reference", "expected_severity", "answer", "contexts",
    "metrics", "run_error", "labels",
}
AGGREGATE_KEYS = {"n", "n_run_errors", "scores", "twin_recall_rate", "median_latency_s"}


def prior(incident_id: str, summary: str = "s", cause: str = "c") -> Prior:
    """A prior incident with this id."""
    return Prior(incident_id, "auth", "sev2", summary, cause, "r")


def golden(cid: str, source: str, twin: str | None = None, severity: str = "sev2") -> GoldenCase:
    """A golden row; it has a twin when `twin` is given."""
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
        """Hold the ranking and record each request size."""
        self.ranked = ranked
        self.asked: list[int] = []

    def __call__(self, question: str, n: int) -> list[Prior]:
        """The first `n` of the ranking."""
        self.asked.append(n)
        return self.ranked[:n]


class FakeGraph:
    """Calls recall like the real recall node, then returns a canned triage."""

    def __init__(self, recall: Any, severity: str, fail: bool) -> None:
        """Hold the recall step, the severity to answer and whether to fail."""
        self.recall, self.severity, self.fail = recall, severity, fail

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Call recall like the real node, then return a canned triage."""
        if self.recall is not None:
            self.recall(state["question"])
        if self.fail:
            raise RuntimeError("model never produced a valid Triage")
        triage = Triage(severity=self.severity, component="auth", next_action="roll back", confidence=0.9)  # type: ignore[arg-type]
        return {"triage": triage, "model_used": "small"}


def graph_factory(severity: str = "sev2", fail_on: set[str] | None = None, use_recall: bool = True):
    """make_graph for tests. A graph fails when its question names an id in `fail_on`."""

    def make_graph(recall: Any) -> FakeGraph:
        """Compile a fake graph; it fails for questions that name an id in `fail_on`."""
        graph = FakeGraph(recall if use_recall else None, severity, False)
        original = graph.invoke

        def invoke(state: dict[str, Any]) -> dict[str, Any]:
            """Decide whether this case fails, then run the graph."""
            graph.fail = any(state["question"].endswith(f" {i}") for i in (fail_on or set()))
            return original(state)

        graph.invoke = invoke  # type: ignore[method-assign]
        return graph

    return make_graph


def ranked() -> FakeCandidates:
    """A retriever that ranks S, T, X, Y."""
    return FakeCandidates([prior("S"), prior("T"), prior("X"), prior("Y")])


def adapter(candidates: FakeCandidates | None = None, **graph: Any) -> FindyFixy:
    """The FindyFixy adapter over the fake graph and retriever."""
    return FindyFixy(make_graph=graph_factory(**graph), candidates=candidates or ranked(),
                     small_model="s", large_model="l")


def run(goldens: list[GoldenCase], sut: FindyFixy | None = None, judge: FakeJudge | None = None,
        **kw: Any) -> dict[str, Any]:
    """`run_eval` over the adapter, with the fake judge and embedder."""
    return run_eval([to_case(g) for g in goldens], sut=sut or adapter(), judge=judge or FakeJudge(),
                    embed=embed, provenance=PROV, **kw)


# ---- Holdout ----------------------------------------------------------------


def test_holdout_drops_the_source_keeps_the_twin_and_trims_to_k() -> None:
    """The source goes, the twin stays, k remain."""
    kept = holdout([prior("S"), prior("T"), prior("X"), prior("Y")], "S", 3)
    assert [p.incident_id for p in kept] == ["T", "X", "Y"]


def test_holdout_with_source_absent_still_trims_to_k() -> None:
    """Without the source in the list, the first k are kept."""
    assert [p.incident_id for p in holdout([prior("A"), prior("B"), prior("C")], "S", 2)] == ["A", "B"]


def test_recall_asks_for_k_plus_one_and_fills_the_sink() -> None:
    """k + 1 are fetched, so k remain after the drop; the sink sees what the model saw."""
    cands = ranked()
    sink: list[Prior] = []
    out = make_holdout_recall(cands, "S", 3, sink)("q")
    assert cands.asked == [4]
    assert [p.incident_id for p in out] == ["T", "X", "Y"]
    assert sink == out


def test_the_source_never_reaches_the_graph_even_when_ranked_first() -> None:
    """The worst-case ranking still keeps the source out."""
    rec = run([golden("g1", "S", "T")])["cases"][0]
    assert "S" not in rec["recalled_ids"]
    assert rec["recalled_ids"] == ["T", "X", "Y"]
    assert rec["twin_recalled"] is True


def test_each_case_keeps_its_own_holdout_under_parallel_load() -> None:
    """Shared state would show up as one case seeing another's recalled incidents."""
    goldens = [golden(f"g{i}", f"S{i}", twin=f"T{i}" if i % 2 else None) for i in range(12)]
    cands = FakeCandidates([prior(f"S{i}") for i in range(12)] + [prior("T1"), prior("X")])
    report = run(goldens, sut=adapter(cands), workers=4)
    for record in report["cases"]:
        assert record["source_id"] not in record["recalled_ids"]
        assert len(record["recalled_ids"]) == 3


# ---- What the judge sees ----------------------------------------------------


def test_contexts_equal_what_the_prompt_used_and_are_capped() -> None:
    """The judge sees the prompt's own rendering, capped the same way."""
    long = prior("L", summary="x" * 1000)
    ctx = render_contexts([prior("A"), long])
    assert ctx[0] == format_prior([prior("A")])
    assert ctx[1] == format_prior([long])
    assert len(ctx[1]) == MAX_PRIOR_CHARS


def test_judge_prompts_contain_the_recalled_context_and_never_the_source() -> None:
    """The twin's text reaches the judge; the source's never does."""
    judge = FakeJudge()
    cands = FakeCandidates([prior("S", summary="SOURCE-TEXT"), prior("T", summary="TWIN-TEXT"),
                            prior("X"), prior("Y")])
    run([golden("g1", "S", "T")], sut=adapter(cands), judge=judge)
    joined = "\n".join(judge.prompts)
    assert "TWIN-TEXT" in joined
    assert "SOURCE-TEXT" not in joined


def test_render_answer_is_plain_text_without_confidence() -> None:
    """Confidence is not a claim, so the judge does not see it."""
    t = Triage(severity="sev1", component="db", next_action="failover", confidence=0.5)
    assert render_answer(t) == "Severity sev1. Component db. Next action: failover"


def test_the_action_scored_is_the_next_action_not_the_whole_answer() -> None:
    """Action quality judges `next_action` only."""
    m = run([golden("g1", "S", "T")])["cases"][0]["metrics"]["action_quality"]
    assert m["score"] == 1.0 and m["trace"][0]["action"] == "roll back"


# ---- The record and report schema -------------------------------------------


def test_a_findyfixy_record_has_exactly_the_keys_of_every_saved_report() -> None:
    """The record and aggregate schema saved reports and the baseline read."""
    report = run([golden("g1", "S", "T", severity="sev1"), golden("g2", "S2")], sut=adapter(fail_on={"g2"}))
    ok, failed = report["cases"]
    assert set(ok) == RECORD_KEYS and set(failed) == RECORD_KEYS
    assert ok["severity"] == "sev2" and ok["severity_match"] is False and ok["expected_severity"] == "sev1"
    assert ok["model_used"] == "small" and ok["twin_id"] == "T" and ok["labels"] == ["severity"]
    assert list(report["aggregates"]) == ["overall", "twin", "no_twin"]
    assert all(set(a) == AGGREGATE_KEYS for a in report["aggregates"].values())
    assert list(report["aggregates"]["overall"]["scores"])[0] == "severity_match"


def test_a_graph_error_is_recorded_with_what_was_recalled() -> None:
    """A failed case still shows the context the model had."""
    report = run([golden("g1", "S1"), golden("g2", "S2"), golden("g3", "S3")], sut=adapter(fail_on={"g2"}))
    rec = report["cases"][1]
    assert rec["run_error"] == "RuntimeError: model never produced a valid Triage"
    assert rec["recalled_ids"] == ["S", "T", "X"]  # the holdout drops S2, not S
    assert rec["twin_recalled"] is False and rec["model_used"] is None
    assert rec["severity"] is None and rec["severity_match"] is None
    assert report["aggregates"]["overall"]["n_run_errors"] == 1


def test_a_graph_that_returns_no_triage_is_a_run_error() -> None:
    """No triage in the final state is a run error, not a crash."""
    class Empty:
        """A graph that never produces a triage."""
        def invoke(self, state):
            """A final state with no triage."""
            return {}

    sut = FindyFixy(make_graph=lambda recall: Empty(), candidates=ranked(), small_model="s", large_model="l")
    assert run([golden("g1", "S")], sut=sut)["cases"][0]["run_error"] == "ValueError: graph returned no triage"


def test_severity_splits_by_group() -> None:
    """Severity match is computed per group."""
    report = run([golden("g1", "S1", "T1", "sev2"), golden("g2", "S2", None, "sev1")])  # the fake says sev2
    agg = report["aggregates"]
    assert agg["twin"]["scores"]["severity_match"]["mean"] == 1.0
    assert agg["no_twin"]["scores"]["severity_match"]["mean"] == 0.0
    assert agg["overall"]["scores"]["severity_match"]["mean"] == 0.5
    assert agg["no_twin"]["twin_recall_rate"] is None


def test_twin_recall_rate_counts_only_twin_cases_where_the_twin_came_back() -> None:
    """The rate counts twin cases whose twin was recalled."""
    cands = FakeCandidates([prior("S"), prior("T1"), prior("X"), prior("Y")])
    report = run([golden("g1", "S", "T1"), golden("g2", "S", "MISSING")], sut=adapter(cands))
    assert report["aggregates"]["twin"]["twin_recall_rate"] == 0.5


def test_twin_recall_rate_leaves_out_run_errors() -> None:
    """Run errors are not in the rate; no twin cases means None."""
    records = [{"run_error": None, "twin_id": "T", "twin_recalled": True},
               {"run_error": "boom", "twin_id": "T"},
               {"run_error": None, "twin_id": None, "twin_recalled": False}]
    assert twin_recall_rate(records) == 1.0
    assert twin_recall_rate([]) is None


def test_describe_names_the_settings_that_shape_an_answer() -> None:
    """Models, threshold and recall depth go into provenance."""
    assert adapter().describe() == {"small_model": "s", "large_model": "l", "threshold": 0.7, "recall_k": 3}


def test_to_case_carries_the_ids_in_meta_and_the_severity_as_a_label() -> None:
    """The golden row becomes a harness case with no FindyFixy type in it."""
    c = to_case(golden("g1", "S", "T", "sev1"))
    assert c.expected == {"severity": "sev1"} and dict(c.meta) == {"source_id": "S", "twin_id": "T"}
    assert c.group == "twin" and c.question == "question g1"


# ---- The entry point --------------------------------------------------------


def fake_session(graph: dict[str, Any] | None = None, log: list[bool] | None = None):
    """A session factory over the fakes. It honours the control flag like the live one."""

    @contextmanager
    def open_session(negative_control: bool):
        """Yield one fake session, honouring the control flag like the live one."""
        if log is not None:
            log.append(negative_control)
        sut = FindyFixy(make_graph=graph_factory(use_recall=not negative_control, **(graph or {})),
                        candidates=ranked(), small_model="s", large_model="l")
        yield Session(sut=sut, cases=[to_case(golden("g1", "S", "T")), to_case(golden("g2", "S2"))],
                      judge_model="j", embed=embed)

    return open_session


def offline(monkeypatch) -> None:
    """Keep the real live_runner, but with a fake judge, digest, git and clock."""
    real = cli.live_runner
    monkeypatch.setattr(cli, "live_runner", lambda open_session, **kw: real(
        open_session, **kw, make_judge=lambda m: FakeJudge(), digest=lambda m: "dig",
        git=lambda repo: ("abc1234", False), now=lambda: NOW))


def entry(argv: list[str], tmp_path: Path, **kw: Any) -> int:
    """The app's real `main`, with test paths for reports and baseline."""
    return eval_answers.main(argv, reports_dir=tmp_path / "r", baseline_path=tmp_path / "b.json", **kw)


def test_no_recall_runs_the_negative_control_and_it_can_never_become_a_baseline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """`--no-recall` reaches the session, the report says control, and no baseline is written."""
    offline(monkeypatch)
    log: list[bool] = []
    assert entry(["--no-recall", "--write-baseline"], tmp_path, open_session=fake_session(log=log)) == 1
    assert log == [True]
    assert "negative control" in capsys.readouterr().err and not (tmp_path / "b.json").exists()
    report = json.loads(next((tmp_path / "r").glob("eval-*.json")).read_text())
    assert report["provenance"]["negative_control"] is True
    assert all(c["recalled_ids"] == [] for c in report["cases"])  # recall was never called


def test_a_normal_run_writes_findyfixy_provenance_and_the_gate_passes_against_itself(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A normal run records FindyFixy's settings and gates clean against itself."""
    offline(monkeypatch)
    log: list[bool] = []
    assert entry(["--write-baseline"], tmp_path, open_session=fake_session(log=log)) == 0
    assert entry(["--gate"], tmp_path, open_session=fake_session(log=log)) == 0
    assert log == [False, False]
    out = capsys.readouterr().out
    assert "gate: pass" in out and "| twin |" in out and "twin_recall_rate" in out
    baseline = json.loads((tmp_path / "b.json").read_text())
    assert {"small_model", "large_model", "threshold", "recall_k"} <= set(baseline["provenance"])
    assert "negative_control" not in baseline["provenance"]


def test_the_case_viewer_shows_the_findyfixy_fields(tmp_path: Path, monkeypatch, capsys) -> None:
    """`--show-case` prints the severity label and the twin data."""
    offline(monkeypatch)
    entry([], tmp_path, open_session=fake_session())
    capsys.readouterr()
    assert entry(["--show-case", "g1"], tmp_path) == 0
    out = capsys.readouterr().out
    for needle in ("LABEL      severity: expected sev2, answered sev2  MATCH", "recalled_ids: T, X, Y",
                   "source_id: S", "twin_id: T", "twin_recalled: True", "model_used: small"):
        assert needle in out, needle


def test_an_unset_model_name_exits_2_through_the_real_live_session(tmp_path: Path, monkeypatch, capsys) -> None:
    """The real `open_live_session` path: no model, no database is touched before the error."""
    monkeypatch.setattr(eval_answers, "load_env", lambda: None)  # a local .env must not fill the gap
    monkeypatch.delenv("SMALL_MODEL", raising=False)
    monkeypatch.setenv("LARGE_MODEL", "l")

    def no_database():
        """Fail the test if the session got as far as the database."""
        raise AssertionError("the database was opened before the missing setting was reported")

    monkeypatch.setattr(eval_answers, "connect", no_database)
    assert entry([], tmp_path) == 2
    assert "error: set the environment variable SMALL_MODEL" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()  # no report for a run that never started
