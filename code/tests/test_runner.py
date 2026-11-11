"""Tests for the runner (wyld3rness/eval/runner.py). Harness types and fakes only.

What these tests cover
    The record schema, label scoring, how an expected failure and a raised
    exception become a run error, the up-front validation, how key
    collisions are refused or contained, the app's extra aggregates, groups in
    order of first appearance, what the judge is shown, and parallel workers.

What a staff+ reader should notice
    A mistake visible before the run must stop it before the first judge call,
    and one found after must never throw away the cases already scored. The
    tests that count `answered` prove the first half; the collision-as-run-error
    tests prove the second.
"""

import threading
from typing import Any

import pytest

from wyld3rness.eval.case import Case
from wyld3rness.eval.runner import new_record
from wyld3rness.eval.stats import CORE_RECORD_KEYS
from wyld3rness.eval.sut import Answer
from wyld3rness.testing import FakeJudge, FakeSUT, case, run

CORE = set(CORE_RECORD_KEYS)


class CountingSUT(FakeSUT):
    """A fake that counts its answers, to prove validation ran before any case."""

    def __init__(self, **kw: Any) -> None:
        """Start at zero answers."""
        super().__init__(**kw)
        self.answered = 0

    def answer(self, c: Case) -> Answer:
        """Count, then answer like the plain fake."""
        self.answered += 1
        return super().answer(c)


# ---- The record -------------------------------------------------------------


def test_the_record_has_the_core_keys_the_label_keys_and_the_meta_flat() -> None:
    """Core keys from the shared schema, three keys per label, and both metas, all flat."""
    record = run([case("g1", source_id="S")])["cases"][0]
    assert set(record) == CORE | {"kind", "kind_match", "expected_kind", "source_id", "model_used"}
    assert record["labels"] == ["kind"]
    assert record["kind"] == "x" and record["kind_match"] is True and record["expected_kind"] == "x"
    assert record["source_id"] == "S" and record["model_used"] == "small"


def test_the_empty_record_matches_the_shared_schema() -> None:
    """The writer and the schema constant agree; `require_schema` would raise otherwise."""
    assert set(new_record(case("g1", kind=None))) == CORE


def test_a_wrong_or_missing_label_is_a_miss() -> None:
    """A different label is a miss, and so is an answer that carries no label at all."""
    assert run([case("g1", kind="y")])["cases"][0]["kind_match"] is False

    class NoLabel(FakeSUT):
        """Answers without the label."""

        def answer(self, c: Case) -> Answer:
            """The plain answer with its labels removed."""
            a = super().answer(c)
            return Answer(text=a.text, contexts=a.contexts, action=a.action, labels={}, meta=a.meta)

    record = run([case("g1")], sut=NoLabel())["cases"][0]
    assert record["kind"] is None and record["kind_match"] is False


def test_the_record_is_self_contained() -> None:
    """Question, reference and expected label are copied in, so the golden file can change later."""
    record = run([case("g1", kind="sev1")])["cases"][0]
    assert record["question"] == "question g1"
    assert record["reference"].startswith("Root cause")
    assert record["expected_kind"] == "sev1"


def test_the_record_keeps_the_answer_the_contexts_and_the_judge_decisions() -> None:
    """Everything needed to hand-check a score is in the record."""
    record = run([case("g1")])["cases"][0]
    assert record["answer"].startswith("Kind x")
    assert record["contexts"] == ["context for g1"]
    assert record["metrics"]["faithfulness"]["trace"][0]["claim"] == "it is auth"
    assert len(record["metrics"]["response_relevancy"]["trace"]) == 3


def test_a_case_without_labels_scores_only_the_metrics() -> None:
    """No label means no `_match` score, and the four metrics in their fixed order."""
    report = run([case("g1", kind=None)])
    assert "kind_match" not in report["cases"][0] and report["cases"][0]["labels"] == []
    assert list(report["aggregates"]["overall"]["scores"]) == [
        "faithfulness", "response_relevancy", "context_recall", "action_quality"]


# ---- Run errors -------------------------------------------------------------


def test_an_expected_failure_keeps_its_meta_and_scores_nothing() -> None:
    """`Answer.failed` becomes the run error, keeps the partial meta, and is left out of the means."""
    report = run([case("g1"), case("g2")], sut=FakeSUT(fail_on=["g2"]))
    failed = report["cases"][1]
    assert failed["run_error"] == "TriageError: no valid record"
    assert failed["model_used"] is None  # partial meta from the system
    assert failed["kind"] is None and failed["kind_match"] is None
    assert failed["metrics"] == {} and failed["answer"] is None
    assert report["aggregates"]["overall"]["n_run_errors"] == 1
    assert report["aggregates"]["overall"]["scores"]["kind_match"]["n_scored"] == 1


def test_an_exception_from_the_system_is_a_run_error_and_the_run_continues() -> None:
    """Whatever `answer` raises is recorded as "TypeName: message"; the other cases still run."""
    report = run([case("g1"), case("g2"), case("g3")], sut=FakeSUT(raise_on=["g2"]))
    errors = [c["run_error"] for c in report["cases"]]
    assert errors[0] is None and errors[2] is None
    assert errors[1] == "RuntimeError: model never produced a valid record"
    assert "model_used" not in report["cases"][1]  # no answer, so no answer-side meta


def test_a_metric_that_raises_is_a_failed_metric_not_an_abort() -> None:
    """An embedder fault fails one metric with its reason; the case is not a run error."""
    def bad_embed(texts: Any) -> list[list[float]]:
        """An embedder that is down."""
        raise OSError("embedder down")

    report = run([case("g1")], embed=bad_embed)
    assert report["cases"][0]["metrics"]["response_relevancy"] == {
        "score": None, "failed": True, "error": "OSError: embedder down", "trace": []}
    assert report["cases"][0]["run_error"] is None


def test_a_bug_in_a_metric_is_not_hidden_as_a_judge_failure() -> None:
    """Expected faults become failed results; a KeyError is our bug and must stop the run."""
    def buggy_embed(texts: Any) -> list[list[float]]:
        """An embedder with a bug."""
        raise KeyError("oops")

    with pytest.raises(KeyError):
        run([case("g1")], embed=buggy_embed)


def test_failed_judge_metrics_are_counted_in_the_report() -> None:
    """A judge that always fails gives no mean, one failure, rate 1.0. Never a zero score."""
    report = run([case("g1")], judge=FakeJudge(fail=True))
    s = report["aggregates"]["overall"]["scores"]["faithfulness"]
    assert s["mean"] is None and s["n_failed"] == 1 and s["failure_rate"] == 1.0


def test_no_action_fails_action_quality_and_never_asks_the_judge() -> None:
    """`action=None` is a failed metric, and no judge call is spent on it."""
    judge = FakeJudge()
    record = run([case("g1")], sut=FakeSUT(action=None), judge=judge)["cases"][0]
    m = record["metrics"]["action_quality"]
    assert m["failed"] is True and "no next action" in m["error"]
    assert not any("first action" in p for p in judge.prompts)


# ---- Collisions and up-front validation ---------------------------------------


@pytest.mark.parametrize("key", ["run_error", "answer", "latency_s", "labels", "kind_match", "source_id"])
def test_answer_meta_that_collides_is_that_cases_run_error_and_the_run_continues(key: str) -> None:
    """A collision known only after answering costs one case, never the run."""
    sut = FakeSUT(meta={key: 1})
    report = run([case("g1", source_id="S"), case("g2", source_id="S")], sut=sut)
    for record in report["cases"]:
        assert record["run_error"].startswith("ValueError: answer meta for case")
        assert key in record["run_error"]
        assert record.get(key) != 1  # nothing was overwritten
    assert report["aggregates"]["overall"]["n_run_errors"] == 2


def test_an_answer_meta_collision_under_workers_is_also_a_run_error() -> None:
    """The thread pool must not turn one bad answer into a lost run."""
    report = run([case(f"g{i}") for i in range(4)], sut=FakeSUT(meta={"answer": 1}), workers=4)
    assert report["aggregates"]["overall"]["n_run_errors"] == 4


@pytest.mark.parametrize("key", ["latency_s", "labels", "kind_match", "expected_kind", "id"])
def test_case_meta_that_collides_stops_the_run_before_any_case_runs(key: str) -> None:
    """Visible before the run, so refused before the first judge call."""
    sut = CountingSUT()
    with pytest.raises(ValueError, match="reuses record keys"):
        run([case("g1"), case("g2", **{key: 1})], sut=sut)
    assert sut.answered == 0


def test_a_group_named_overall_stops_the_run_before_any_case_runs() -> None:
    """It would replace the overall aggregate, so it is refused up front."""
    sut = CountingSUT()
    with pytest.raises(ValueError, match="overall"):
        run([case("g1"), case("g2", "overall")], sut=sut)
    assert sut.answered == 0


def test_summarise_keys_that_clash_are_refused_before_any_case_runs() -> None:
    """`summarise([])` is probed up front, so a clash costs no judge call."""
    class Bad(CountingSUT):
        """Claims a harness aggregate key."""

        def summarise(self, records):
            """Always returns `n`."""
            return {"n": 99}

    sut = Bad()
    with pytest.raises(ValueError, match="summarise"):
        run([case("g1")], sut=sut)
    assert sut.answered == 0


def test_a_summarise_clash_seen_only_on_real_records_is_dropped_with_a_warning() -> None:
    """Raising after scoring would lose the run, so the harness count wins and a warning is printed."""
    class Late(FakeSUT):
        """Clashes only when given records."""

        def summarise(self, records):
            """Returns `n` for a non-empty list only, so the probe misses it."""
            return {"n": 99} if records else {}

    lines: list[str] = []
    report = run([case("g1")], sut=Late(), progress=lines.append)
    assert report["aggregates"]["overall"]["n"] == 1
    assert any("summarise() key 'n'" in line and "dropped" in line for line in lines)


# ---- Aggregates and groups --------------------------------------------------


def test_groups_come_from_the_cases_in_order_of_first_appearance() -> None:
    """No hard-coded group list: the cases define the groups and their order."""
    report = run([case("g1", "b"), case("g2", "a"), case("g3", "b")])
    assert list(report["aggregates"]) == ["overall", "b", "a"]
    assert report["aggregates"]["b"]["n"] == 2 and report["aggregates"]["a"]["n"] == 1


def test_the_apps_summary_is_merged_into_every_aggregate() -> None:
    """`summarise` runs on overall and on each group separately."""
    report = run([case("g1", "a"), case("g2", "b")], sut=FakeSUT(fail_on=["g2"]))
    assert report["aggregates"]["overall"]["ran_rate"] == 0.5
    assert report["aggregates"]["a"]["ran_rate"] == 1.0
    assert report["aggregates"]["b"]["ran_rate"] == 0.0


def test_labels_split_by_group() -> None:
    """Each group's label rate counts only its own cases."""
    report = run([case("g1", "a", kind="x"), case("g2", "b", kind="y")])  # the fake always says x
    agg = report["aggregates"]
    assert agg["a"]["scores"]["kind_match"]["mean"] == 1.0
    assert agg["b"]["scores"]["kind_match"]["mean"] == 0.0
    assert agg["overall"]["scores"]["kind_match"]["mean"] == 0.5


# ---- What the judge sees ----------------------------------------------------


def test_the_judge_sees_exactly_the_answers_contexts() -> None:
    """The contexts the system reported are the ones the judge is asked about."""
    judge = FakeJudge()
    run([case("g1")], judge=judge)
    assert "context for g1" in "\n".join(judge.prompts)


def test_the_action_metric_judges_the_action_not_the_whole_answer() -> None:
    """`action_quality` gets `Answer.action`, not the rendered text."""
    m = run([case("g1")])["cases"][0]["metrics"]["action_quality"]
    assert m["score"] == 1.0 and m["trace"][0]["rating"] == "helpful"
    assert m["trace"][0]["action"] == "roll back"


# ---- Parallel workers -------------------------------------------------------


def many_cases(n: int) -> list[Case]:
    """`n` cases split over two groups."""
    return [case(f"g{i}", "a" if i % 2 else "b") for i in range(n)]


def test_parallel_and_serial_runs_give_the_same_records_in_the_same_order() -> None:
    """Workers change the timing, never the records or their order."""
    cases = many_cases(8)
    serial = run(cases, workers=1)
    parallel = run(cases, workers=4)
    strip = lambda r: [{k: v for k, v in c.items() if k != "latency_s"} for c in r["cases"]]  # noqa: E731
    assert strip(parallel) == strip(serial)
    assert [c["id"] for c in parallel["cases"]] == [c.id for c in cases]
    assert parallel["aggregates"]["overall"]["scores"] == serial["aggregates"]["overall"]["scores"]


def test_workers_really_overlap_cases() -> None:
    """Two cases wait for each other at a barrier. Only concurrent execution passes it."""
    report = run(many_cases(2), sut=FakeSUT(barrier=threading.Barrier(2, timeout=5)), workers=2)
    assert report["aggregates"]["overall"]["n_run_errors"] == 0


def test_one_worker_does_not_overlap_cases() -> None:
    """The same barrier with one worker is never released: both cases record a run error."""
    report = run(many_cases(2), sut=FakeSUT(barrier=threading.Barrier(2, timeout=0.2)), workers=1)
    assert report["aggregates"]["overall"]["n_run_errors"] == 2


def test_each_case_keeps_its_own_context_under_parallel_load() -> None:
    """Shared state would show up as one case carrying another's context."""
    report = run(many_cases(12), workers=4)
    for record in report["cases"]:
        assert record["contexts"] == [f"context for {record['id']}"]


def test_progress_counts_each_case_once_per_pass_even_in_parallel() -> None:
    """The shared counters are locked: every number from 1 to n appears once in each pass."""
    lines: list[str] = []
    run(many_cases(10), workers=4, progress=lines.append)
    answered = [line for line in lines if not line.startswith("[judged")]
    judged = [line for line in lines if line.startswith("[judged")]
    assert sorted(int(line.split("/")[0][1:]) for line in answered) == list(range(1, 11))
    assert sorted(int(line.split("/")[0].split()[1]) for line in judged) == list(range(1, 11))


def test_progress_names_the_label_match_or_the_run_error() -> None:
    """One line per answered case, in the shape the FindyFixy runs printed, then one per judged case."""
    lines: list[str] = []
    run([case("g1"), case("g2")], sut=FakeSUT(fail_on=["g2"]), progress=lines.append, timer=lambda: 0.0)
    assert lines == [
        "[1/2] g1 kind_match=True (0.0s)",
        "[2/2] g2 TriageError: no valid record (0.0s)",
        "[judged 1/1] g1",  # g2 has a run error: nothing to judge
    ]


@pytest.mark.parametrize("bad", [0, -1, 5, 100])
def test_workers_outside_the_cap_are_refused(bad: int) -> None:
    """The cap protects the model server's memory; it is not a suggestion."""
    with pytest.raises(ValueError, match="workers must be between"):
        run(many_cases(1), workers=bad)


class SteppingClock:
    """A clock that moves forward by `answer_step` per answer and `judge_step` per judge call.

    Lets a test set how long each pass takes without sleeping. The runner reads
    the clock; the fakes move it.
    """

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        """The current time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward. Locked, because workers advance it from threads."""
        with self.lock:
            self.now += seconds


class TimedSUT(FakeSUT):
    """A fake whose answer moves the clock, so the answer pass has a known length."""

    def __init__(self, clock: SteppingClock, step: float) -> None:
        """Keep the clock and the step."""
        super().__init__()
        self.clock, self.step = clock, step

    def answer(self, c: Case) -> Answer:
        """Advance the clock by one step, then answer like the plain fake."""
        self.clock.advance(self.step)
        return super().answer(c)


class TimedJudge(FakeJudge):
    """A fake judge whose every call moves the clock."""

    def __init__(self, clock: SteppingClock, step: float) -> None:
        """Keep the clock and the step."""
        super().__init__()
        self.clock, self.step = clock, step

    def ask(self, prompt: str, schema: Any) -> Any:
        """Advance the clock by one step, then answer like the plain fake."""
        self.clock.advance(self.step)
        return super().ask(prompt, schema)


def test_the_report_records_each_pass_workers_and_throughput() -> None:
    """Wall time, each pass, workers and cases per minute go into `timing`."""
    clock = SteppingClock()
    # 6 answers at 1 s each, and 1 s per judge call. The number of judge calls
    # per case is the metrics' business, so it is counted, not assumed.
    judge = TimedJudge(clock, 1.0)
    report = run(many_cases(6), sut=TimedSUT(clock, 1.0), judge=judge, timer=clock)
    calls = len(judge.prompts)
    assert report["timing"] == {
        "workers": 1,
        "wall_s": round(6.0 + calls, 1),
        "answer_s": 6.0,
        "judge_s": float(calls),
        "cases_per_min": round(60 * 6 / (6.0 + calls), 2),
    }


def test_every_answer_comes_before_the_first_judge_call() -> None:
    """The point of the two passes: the answering model is never interrupted by the judge."""
    calls: list[str] = []

    class LoggingSUT(FakeSUT):
        """Logs each answer."""

        def answer(self, c: Case) -> Answer:
            """Log, then answer."""
            calls.append(f"answer {c.id}")
            return super().answer(c)

    class LoggingJudge(FakeJudge):
        """Logs each judge call."""

        def ask(self, prompt: str, schema: Any) -> Any:
            """Log, then judge."""
            calls.append("judge")
            return super().ask(prompt, schema)

    run(many_cases(4), sut=LoggingSUT(), judge=LoggingJudge())
    first_judge = calls.index("judge")
    assert calls[:first_judge] == [f"answer g{i}" for i in range(4)]
    assert all(c == "judge" for c in calls[first_judge:])


def test_the_judge_sees_the_same_prompts_in_the_same_order_as_case_by_case() -> None:
    """Moving the judge to its own pass changes when it is asked, never what or in which order.

    The reference order is built by running each case on its own, which is
    exactly the old case-by-case sequence of judge calls.
    """
    cases = many_cases(5)
    together = FakeJudge()
    run(cases, judge=together)
    one_by_one: list[str] = []
    for c in cases:
        alone = FakeJudge()
        run([c], judge=alone)
        one_by_one += alone.prompts
    assert together.prompts == one_by_one


def test_parallel_judging_keeps_the_answer_pass_first() -> None:
    """With workers the passes still do not overlap: no judge call before the last answer."""
    clock = SteppingClock()
    order: list[str] = []
    lock = threading.Lock()

    class LoggingSUT(FakeSUT):
        """Logs each answer from any thread."""

        def answer(self, c: Case) -> Answer:
            """Log, then answer."""
            with lock:
                order.append("answer")
            return super().answer(c)

    class LoggingJudge(FakeJudge):
        """Logs each judge call from any thread."""

        def ask(self, prompt: str, schema: Any) -> Any:
            """Log, then judge."""
            with lock:
                order.append("judge")
            return super().ask(prompt, schema)

    run(many_cases(8), sut=LoggingSUT(), judge=LoggingJudge(), workers=4, timer=clock)
    assert order.index("judge") == order.count("answer") == 8


def test_an_empty_run_does_not_crash() -> None:
    """No cases: only the overall aggregate, with no means."""
    report = run([])
    assert list(report["aggregates"]) == ["overall"]
    assert report["aggregates"]["overall"]["scores"]["faithfulness"]["mean"] is None


# ---- Live events --------------------------------------------------------------


def recorder() -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    """A list and an `on_event` that appends (kind, data) to it."""
    seen: list[tuple[str, dict[str, Any]]] = []
    return seen, lambda kind, **data: seen.append((kind, data))


def test_events_answer_every_case_then_score_every_case_in_order() -> None:
    """Started; per case started and answered; then per case one event per metric in order, scored."""
    seen, on_event = recorder()
    run([case("g1"), case("g2", group="b", kind="y")], on_event=on_event)
    kinds = [k for k, _ in seen]
    answer_pass = ["case_started", "case_answered"]
    judge_pass = [*["metric_scored"] * 4, "case_scored"]
    assert kinds == ["eval_started", *answer_pass, *answer_pass, *judge_pass, *judge_pass]
    assert seen[0][1] == {"n": 2, "cases": ["g1", "g2"], "system": {"fake_model": "m1", "fake_threshold": 0.5}}
    assert seen[1][1] == {"id": "g1", "group": "a", "index": 1, "n": 2}
    assert seen[4][1] == {"id": "g2", "labels": {"kind": "x"}, "expected": {"kind": "y"}, "run_error": None}
    assert [d["id"] for k, d in seen if k == "case_scored"] == ["g1", "g2"]
    metrics = [d for k, d in seen[5:10] if k == "metric_scored"]
    assert {d["id"] for d in metrics} == {"g1"}
    assert [d["name"] for d in metrics] == ["faithfulness", "response_relevancy", "context_recall", "action_quality"]
    assert all(d["score"] == 1.0 and d["failed"] is False for d in metrics)


def test_a_run_error_is_answered_and_scored_with_no_metric_events() -> None:
    """A failed case still closes on the stream, and no metric is reported for it."""
    seen, on_event = recorder()
    run([case("g1")], sut=FakeSUT(raise_on=["g1"]), on_event=on_event)
    assert [k for k, _ in seen] == ["eval_started", "case_started", "case_answered", "case_scored"]
    assert seen[2][1]["run_error"] == "RuntimeError: model never produced a valid record"
    assert seen[2][1]["labels"] == {"kind": None}


def test_a_failed_metric_is_reported_failed_with_no_score() -> None:
    """A judge failure is an event with score None, never a zero."""
    seen, on_event = recorder()
    run([case("g1")], judge=FakeJudge(fail=True), on_event=on_event)
    metrics = [d for k, d in seen if k == "metric_scored"]
    assert metrics and all(d["score"] is None and d["failed"] for d in metrics)


def test_listening_changes_nothing_in_the_report() -> None:
    """The callback only observes: the report is the same with and without it."""
    cases = [case("g1"), case("g2", kind="y")]
    _, on_event = recorder()
    quiet = run(cases, timer=lambda: 0.0)
    loud = run(cases, timer=lambda: 0.0, on_event=on_event)
    assert quiet == loud


def test_a_run_refused_up_front_reports_nothing() -> None:
    """Validation fails before `eval_started`, so a viewer never sees a run that did not start."""
    seen, on_event = recorder()
    with pytest.raises(ValueError):
        run([case("g1", group="overall")], on_event=on_event)
    assert seen == []


def test_events_keep_each_cases_order_under_parallel_workers() -> None:
    """Cases interleave across threads, but each case's own events stay in order."""
    seen, on_event = recorder()
    lock = threading.Lock()

    def safe(kind: str, **data: Any) -> None:
        """A thread-safe sink, as the contract asks for under workers > 1."""
        with lock:
            on_event(kind, **data)

    run([case(f"g{i}") for i in range(4)], workers=4, on_event=safe)
    for i in range(4):
        mine = [k for k, d in seen if d.get("id") == f"g{i}"]
        assert mine == ["case_started", "case_answered", *["metric_scored"] * 4, "case_scored"]


def test_a_stop_ends_the_run_before_the_next_case_with_no_report() -> None:
    """`should_stop` is asked before each case; yes raises after the case in flight."""
    from wyld3rness.eval.runner import EvalStopped

    seen, on_event = recorder()
    sut = CountingSUT()
    with pytest.raises(EvalStopped, match="stopped after 1 of 3 cases"):
        run([case("g1"), case("g2"), case("g3")], sut=sut, on_event=on_event,
            should_stop=lambda: sut.answered >= 1)
    assert sut.answered == 1
    assert [d["id"] for k, d in seen if k == "case_started"] == ["g1"]


def test_a_stop_in_the_judge_pass_ends_the_run_with_no_report() -> None:
    """`should_stop` is asked before each judged case too: every case answered, one judged, then stop."""
    from wyld3rness.eval.runner import EvalStopped

    seen, on_event = recorder()
    sut = CountingSUT()
    with pytest.raises(EvalStopped, match="stopped after judging 1 of 3 answered cases"):
        run([case("g1"), case("g2"), case("g3")], sut=sut, on_event=on_event,
            should_stop=lambda: sum(k == "case_scored" for k, _ in seen) >= 1)
    assert sut.answered == 3
    assert [d["id"] for k, d in seen if k == "case_scored"] == ["g1"]


def test_a_run_error_case_is_scored_in_the_answer_pass_and_never_judged() -> None:
    """A failed case closes at once, before the judge pass, and costs no judge call."""
    seen, on_event = recorder()
    judge = FakeJudge()
    run([case("g1"), case("g2")], sut=FakeSUT(raise_on=["g1"]), judge=judge, on_event=on_event)
    kinds = [(k, d.get("id")) for k, d in seen]
    assert kinds[:5] == [("eval_started", None), ("case_started", "g1"), ("case_answered", "g1"),
                         ("case_scored", "g1"), ("case_started", "g2")]
    alone = FakeJudge()
    run([case("g2")], judge=alone)
    assert judge.prompts == alone.prompts
