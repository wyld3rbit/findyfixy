"""Tests for eval cases as traces with scores (wyld3rness/eval/runner.py, wyld3rness/trace/scores.py).

What these tests cover
    Each case is one trace: an "eval case" span current while the system
    answers (so the system's own spans nest under it), then a "judge" span
    with one child per metric in the same trace. Label matches and metric
    scores go to the score sink on that trace; a failed metric is an error on
    its span and never a score. Tracing changes no number in the report.
    The Langfuse sender posts the right body to a real local HTTP server,
    and drops (never raises) when the backend is down.

What a staff+ reader should notice
    "Tracing changes nothing" is asserted on the whole report, minus the one
    key tracing adds. That is the property that lets tracing ride along on
    the attested gate run.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, ClassVar

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
import pytest

from wyld3rness.eval.case import Case
from wyld3rness.eval.sut import Answer
from wyld3rness.testing import FakeJudge, FakeSUT, case, run
from wyld3rness.trace.scores import LangfuseScores
from wyld3rness.trace.tracing import TraceConfig, Tracing

CONFIG = TraceConfig(host="http://127.0.0.1:1", public_key="pk-lf-t", secret_key="sk-lf-t", service_name="t")


class FakeScores:
    """A score sink that keeps every score."""

    def __init__(self) -> None:
        """Start empty."""
        self.got: list[tuple[str, str, float, str]] = []
        self.lock = threading.Lock()

    def score(self, trace_id: str, name: str, value: float, kind: str, comment: str | None = None) -> None:
        """Keep the score."""
        with self.lock:
            self.got.append((trace_id, name, value, kind))


class SpanMakingSUT(FakeSUT):
    """A system under test that makes its own span, as a traced app does, with no span handed to it."""

    def __init__(self, tracing: Tracing, **kw: Any) -> None:
        """Keep the provider the app would have."""
        super().__init__(**kw)
        self.tracer = tracing.tracer()

    def answer(self, c: Case) -> Answer:
        """Answer inside a span of its own."""
        with self.tracer.start_as_current_span("app run"):
            return super().answer(c)


@pytest.fixture
def traced():
    """A Tracing to memory and its exporter."""
    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    yield tracing, exporter
    tracing.close()


def spans_of(tracing: Tracing, exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    """Flush, then every finished span."""
    assert tracing.flush()
    return list(exporter.get_finished_spans())


def parent(spans: list[ReadableSpan], span: ReadableSpan) -> str | None:
    """The parent's name, or None for a root."""
    if span.parent is None:
        return None
    return next(s.name for s in spans if s.context.span_id == span.parent.span_id)


def test_each_case_is_one_trace_with_the_apps_spans_and_the_judge_inside(traced) -> None:
    """eval case > app run, and eval case > judge > one span per metric, all in one trace per case."""
    tracing, exporter = traced
    report = run([case("g1"), case("g2")], sut=SpanMakingSUT(tracing), tracing=tracing, scores=FakeScores())
    spans = spans_of(tracing, exporter)
    for record in report["cases"]:
        mine = [s for s in spans if format(s.context.trace_id, "032x") == record["trace_id"]]
        names = sorted(s.name for s in mine)
        assert names == sorted(
            [
                f"eval case {record['id']}",
                "app run",
                "judge",
                "faithfulness",
                "response_relevancy",
                "context_recall",
                "action_quality",
            ]
        )
        by = {s.name: s for s in mine}
        assert parent(mine, by["app run"]) == f"eval case {record['id']}"
        assert parent(mine, by["judge"]) == f"eval case {record['id']}"
        assert parent(mine, by["faithfulness"]) == "judge"
        assert by[f"eval case {record['id']}"].attributes["eval.kind_match"] is True
    assert len({r["trace_id"] for r in report["cases"]}) == 2


def test_label_matches_and_metric_scores_go_to_their_trace(traced) -> None:
    """A boolean per label match, a number per metric, all keyed by the case's trace id."""
    tracing, _ = traced
    scores = FakeScores()
    report = run([case("g1"), case("g2", kind="y")], tracing=tracing, scores=scores)
    ids = {r["id"]: r["trace_id"] for r in report["cases"]}
    by_trace = {tid: {name: (value, kind) for t, name, value, kind in scores.got if t == tid} for tid in ids.values()}
    assert by_trace[ids["g1"]]["kind_match"] == (1.0, "BOOLEAN")
    assert by_trace[ids["g2"]]["kind_match"] == (0.0, "BOOLEAN")
    assert by_trace[ids["g1"]]["faithfulness"] == (1.0, "NUMERIC")
    assert set(by_trace[ids["g1"]]) == {
        "kind_match",
        "faithfulness",
        "response_relevancy",
        "context_recall",
        "action_quality",
    }


def test_a_failed_metric_is_an_error_span_and_never_a_score(traced) -> None:
    """The judge never answered: no score (0 would be a lie), and the span says why."""
    tracing, exporter = traced
    scores = FakeScores()
    run([case("g1")], judge=FakeJudge(fail=True), tracing=tracing, scores=scores)
    assert [name for _, name, _, _ in scores.got] == ["kind_match"]
    metric = next(s for s in spans_of(tracing, exporter) if s.name == "faithfulness")
    assert metric.status.status_code is StatusCode.ERROR


def test_a_run_error_case_is_an_error_trace_with_no_judge_and_no_scores(traced) -> None:
    """Nothing to judge: the case span carries the run error, and nothing is scored."""
    tracing, exporter = traced
    scores = FakeScores()
    report = run([case("g1")], sut=FakeSUT(raise_on=["g1"]), tracing=tracing, scores=scores)
    spans = spans_of(tracing, exporter)
    assert [s.name for s in spans] == ["eval case g1"]
    assert spans[0].status.status_code is StatusCode.ERROR
    assert scores.got == [] and report["cases"][0]["trace_id"]


def test_tracing_changes_no_number_in_the_report(traced) -> None:
    """The same run with and without tracing: identical but for the trace ids."""
    tracing, _ = traced
    cases = [case("g1"), case("g2", kind="y"), case("g3")]
    plain = run(cases, timer=lambda: 0.0)
    traced_report = run(cases, timer=lambda: 0.0, tracing=tracing, scores=FakeScores())
    for record in traced_report["cases"]:
        assert len(record.pop("trace_id")) == 32
    assert traced_report == plain


def test_parallel_workers_keep_each_cases_spans_in_its_own_trace(traced) -> None:
    """Four threads at once: every app span sits in the trace of the case it answered."""
    tracing, exporter = traced
    report = run(
        [case(f"g{i}") for i in range(8)], sut=SpanMakingSUT(tracing), tracing=tracing, scores=FakeScores(), workers=4
    )
    spans = spans_of(tracing, exporter)
    for record in report["cases"]:
        mine = [s for s in spans if format(s.context.trace_id, "032x") == record["trace_id"]]
        assert [s.name for s in mine].count("app run") == 1


# ---- the Langfuse sender --------------------------------------------------------


class Capture(BaseHTTPRequestHandler):
    """A stand-in for Langfuse's score endpoint: keeps each request."""

    got: ClassVar[list[tuple[str, dict[str, Any], str]]] = []

    def do_POST(self) -> None:
        """Keep the path, the JSON body and the auth header; answer like Langfuse."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Capture.got.append((self.path, body, self.headers["Authorization"]))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"id": "x"}')

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet."""


def test_scores_are_posted_with_a_deterministic_id_and_the_key_pair() -> None:
    """The body Langfuse expects; a resend of the same score has the same id, so it replaces."""
    Capture.got = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Capture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = TraceConfig(
        host=f"http://127.0.0.1:{server.server_address[1]}",
        public_key="pk-lf-a",
        secret_key="sk-lf-b",
        service_name="t",
    )
    sink = LangfuseScores(config)
    sink.score("t1", "faithfulness", 0.75, "NUMERIC")
    sink.score("t1", "faithfulness", 0.75, "NUMERIC")
    sink.close()
    server.shutdown()
    assert sink.sent == 2 and sink.dropped == 0
    path, body, auth = Capture.got[0]
    assert path == "/api/public/scores" and auth == config.headers["Authorization"]
    assert body == {
        "id": "t1-faithfulness",
        "traceId": "t1",
        "name": "faithfulness",
        "value": 0.75,
        "dataType": "NUMERIC",
    }
    assert Capture.got[1][1]["id"] == body["id"]


def test_a_backend_that_is_down_drops_scores_and_never_raises() -> None:
    """Connection refused: every score is counted as dropped; the caller is never interrupted."""
    sink = LangfuseScores(CONFIG, timeout_s=1.0)  # port 1 accepts no connection
    for i in range(3):
        sink.score(f"t{i}", "faithfulness", 1.0, "NUMERIC")
    sink.close()
    assert (sink.sent, sink.dropped) == (0, 3)


def test_case_traces_carry_the_runs_session_and_the_control_flag(traced) -> None:
    """Two runs, or a control run and a real one, never mix: each case span names its run."""
    tracing, exporter = traced
    run([case("g1")], tracing=tracing, provenance={"timestamp_utc": "T1", "negative_control": True})
    root = next(s for s in spans_of(tracing, exporter) if s.name == "eval case g1")
    assert root.attributes["session.id"] == "eval T1"
    assert root.attributes["eval.negative_control"] is True


def test_a_case_span_ends_even_when_a_viewer_raises(traced) -> None:
    """A broken on_event stops the run, but the case span is still ended and exported, not orphaned."""
    tracing, exporter = traced

    def broken(kind: str, **data: Any) -> None:
        """A viewer with a bug in it."""
        if kind == "case_answered":
            raise RuntimeError("viewer bug")

    with pytest.raises(RuntimeError):
        run([case("g1")], tracing=tracing, on_event=broken)
    root = next(s for s in spans_of(tracing, exporter) if s.name == "eval case g1")
    assert root.status.status_code is StatusCode.ERROR


def test_an_app_meta_key_named_trace_id_is_a_run_error_not_overwritten(traced) -> None:
    """The record's trace_id goes through the same clash check as every other key."""
    tracing, _ = traced
    events: list[tuple[str, dict]] = []
    report = run(
        [case("g1")],
        sut=FakeSUT(meta={"model_used": "small", "trace_id": "the apps own"}),
        tracing=tracing,
        on_event=lambda kind, **data: events.append((kind, data)),
    )
    record = report["cases"][0]
    assert "trace_id" in record["run_error"]
    # Closed like any run error: unlabelled, answered with the error, then scored.
    assert all(record[label] is None for label in record["labels"])
    kinds = [(k, d.get("run_error")) for k, d in events if k in ("case_answered", "case_scored")]
    assert kinds[0][0] == "case_answered" and kinds[0][1] and kinds[1][0] == "case_scored"


def test_an_app_meta_key_named_trace_id_is_a_run_error_without_tracing_too() -> None:
    """Turning tracing on must not change a score: the name is refused either way."""
    report = run([case("g1")], sut=FakeSUT(meta={"model_used": "small", "trace_id": "the apps own"}))
    assert "trace_id" in report["cases"][0]["run_error"]


def test_a_case_meta_key_named_trace_id_stops_the_run_like_any_case_meta_clash() -> None:
    """A golden set that uses a runner key is a bug in the set: it fails loudly, not as a quiet run error."""
    with pytest.raises(ValueError, match="trace_id"):
        run([case("g1", trace_id="x")])


class BadStatus(BaseHTTPRequestHandler):
    """A backend that answers with a broken status line: http.client raises a non-OSError."""

    def do_POST(self) -> None:
        """Write garbage instead of an HTTP response."""
        self.rfile.read(int(self.headers["Content-Length"]))
        self.wfile.write(b"garbage\r\n\r\n")

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet."""


def test_the_sender_survives_a_malformed_response_and_counts_it() -> None:
    """Any failure drops that score; the thread lives on and sends the next one."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), BadStatus)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = TraceConfig(
        host=f"http://127.0.0.1:{server.server_address[1]}",
        public_key="pk-lf-a",
        secret_key="sk-lf-b",
        service_name="t",
    )
    sink = LangfuseScores(config, timeout_s=2.0)
    sink.score("t1", "a", 1.0, "NUMERIC")
    sink.score("t1", "b", 1.0, "NUMERIC")
    sink.close()
    server.shutdown()
    assert sink.dropped == 2 and not sink._thread.is_alive()


def test_scores_left_when_close_times_out_are_counted_as_dropped() -> None:
    """A backend that hangs: whatever close could not send shows up in `dropped`, so the warning is true."""
    gate = threading.Event()

    class Hang(BaseHTTPRequestHandler):
        """Holds every request until the test ends."""

        def do_POST(self) -> None:
            """Wait, then answer."""
            gate.wait(10)
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            """Quiet."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Hang)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = TraceConfig(
        host=f"http://127.0.0.1:{server.server_address[1]}",
        public_key="pk-lf-a",
        secret_key="sk-lf-b",
        service_name="t",
    )
    sink = LangfuseScores(config, timeout_s=5.0)
    for i in range(5):
        sink.score("t1", f"s{i}", 1.0, "NUMERIC")
    sink.close(timeout_s=0.5)
    assert sink.sent + sink.dropped >= 4  # the one in flight may still finish later
    gate.set()
    server.shutdown()
