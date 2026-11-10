"""Tests for harness/trace: the span tree, the provider, and that tracing never slows a run.

What these tests cover
    Parents come from the innermost open span; closing a span ends the spans
    opened inside it; a failed run marks its open spans and the root; None
    attributes are dropped, long values cut; the config never prints its
    secret; and a backend that is down or hangs does not slow the code that
    makes spans.

What a staff+ reader should notice
    The slow-backend test is the contract that matters most. Tracing that
    blocks on its backend turns an observability outage into a product
    outage. The test makes the exporter hang and times the run, not the export.
"""

from __future__ import annotations

import threading
import time

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from harness.trace.spans import (
    GEN_AI_INPUT_TOKENS,
    GEN_AI_REQUEST_MODEL,
    LF_INPUT,
    LF_OUTPUT,
    LF_TRACE_NAME,
    LF_TYPE,
    MAX_VALUE_CHARS,
    SpanTree,
    as_text,
)
from harness.trace.tracing import TraceConfig, Tracing

CONFIG = TraceConfig(host="http://127.0.0.1:1", public_key="pk-lf-test", secret_key="sk-lf-SECRET", service_name="t")


@pytest.fixture
def traced():
    """A Tracing that exports to memory, and the exporter to read the spans from."""
    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    yield tracing, exporter
    tracing.close()


def finished(tracing: Tracing, exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    """Flush, then every exported span by name."""
    assert tracing.flush()
    return {s.name: s for s in exporter.get_finished_spans()}


def parent_name(spans: dict[str, ReadableSpan], span: ReadableSpan) -> str | None:
    """The name of a span's parent among `spans`, or None for a root."""
    if span.parent is None:
        return None
    return next(s.name for s in spans.values() if s.context.span_id == span.parent.span_id)


def test_parents_come_from_the_innermost_open_span(traced) -> None:
    """node > model, then node > tool; all in one trace under the root."""
    tracing, exporter = traced
    tree = SpanTree(tracing.tracer(), "run", input="the alert")
    tree.open("node", "node investigate")
    tree.open("model", "model m", kind="generation", attributes={GEN_AI_REQUEST_MODEL: "m"})
    tree.close("model", attributes={GEN_AI_INPUT_TOKENS: 12}, output="reply")
    tree.open("tool", "tool deploys", kind="tool")
    tree.close("tool", output="3 deploys")
    tree.close("node")
    tree.finish(output={"severity": "sev2"})
    spans = finished(tracing, exporter)
    assert parent_name(spans, spans["run"]) is None
    assert parent_name(spans, spans["node investigate"]) == "run"
    assert parent_name(spans, spans["model m"]) == "node investigate"
    assert parent_name(spans, spans["tool deploys"]) == "node investigate"
    assert len({s.context.trace_id for s in spans.values()}) == 1
    assert format(spans["run"].context.trace_id, "032x") == tree.trace_id
    model = spans["model m"].attributes
    assert model[LF_TYPE] == "generation" and model[GEN_AI_REQUEST_MODEL] == "m"
    assert model[GEN_AI_INPUT_TOKENS] == 12 and model[LF_OUTPUT] == "reply"
    assert spans["run"].attributes[LF_TRACE_NAME] == "run"
    assert spans["run"].attributes[LF_INPUT] == "the alert"
    assert spans["run"].attributes[LF_OUTPUT] == '{"severity": "sev2"}'


def test_closing_a_span_ends_what_was_opened_inside_it(traced) -> None:
    """A node that ends while its model call is still open ends the model call first."""
    tracing, exporter = traced
    tree = SpanTree(tracing.tracer(), "run")
    tree.open("node", "node small")
    tree.open("model", "model m", kind="generation")
    tree.close("node")
    tree.open("node", "node large")  # a sibling, not a child of the closed model span
    tree.finish()
    spans = finished(tracing, exporter)
    assert spans["model m"].end_time <= spans["node small"].end_time
    assert parent_name(spans, spans["node large"]) == "run"


def test_closing_a_key_that_is_not_open_is_ignored(traced) -> None:
    """An event without its opener must not break the run or end the wrong span."""
    tracing, exporter = traced
    tree = SpanTree(tracing.tracer(), "run")
    tree.open("node", "node small")
    tree.close("tool")
    tree.close("node")
    tree.finish()
    assert set(finished(tracing, exporter)) == {"run", "node small"}


def test_a_failed_run_marks_the_open_spans_and_the_root(traced) -> None:
    """Where the run stopped stays visible: the span in progress and the root are errors."""
    tracing, exporter = traced
    tree = SpanTree(tracing.tracer(), "run")
    tree.open("node", "node large")
    tree.open("model", "model m", kind="generation")
    tree.finish(error="ConnectError: Ollama is down")
    spans = finished(tracing, exporter)
    for name in ("run", "node large", "model m"):
        assert spans[name].status.status_code is StatusCode.ERROR, name
        assert spans[name].attributes["langfuse.observation.level"] == "ERROR"
    assert "Ollama is down" in spans["run"].status.description


def test_unknown_values_are_left_out_never_sent_as_zero(traced) -> None:
    """A model that reports no tokens has no token attribute: unknown is not 0."""
    tracing, exporter = traced
    tree = SpanTree(tracing.tracer(), "run")
    tree.open("model", "model m", kind="generation")
    tree.close("model", attributes={GEN_AI_INPUT_TOKENS: None})
    tree.finish()
    assert GEN_AI_INPUT_TOKENS not in finished(tracing, exporter)["model m"].attributes


def test_long_values_are_cut_and_marked() -> None:
    """A huge prompt cannot make a span the backend refuses."""
    text = as_text("x" * (MAX_VALUE_CHARS + 10))
    assert len(text) < MAX_VALUE_CHARS + 40 and text.endswith(f"[cut, {MAX_VALUE_CHARS + 10} chars]")
    assert as_text({"a": 1}) == '{"a": 1}'


def test_the_config_never_shows_its_secret() -> None:
    """A config in a traceback or log line leaks nothing; the header carries the pair."""
    assert "SECRET" not in repr(CONFIG) and "SECRET" not in str(CONFIG)
    assert CONFIG.endpoint == "http://127.0.0.1:1/api/public/otel/v1/traces"
    assert CONFIG.headers["Authorization"].startswith("Basic ")


class HangingExporter(SpanExporter):
    """A backend that hangs: every export blocks until released."""

    def __init__(self) -> None:
        """Start blocked."""
        self.release = threading.Event()

    def export(self, spans) -> SpanExportResult:
        """Block, as a hung network call would."""
        self.release.wait(10)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Let a blocked export finish."""
        self.release.set()


def test_a_hung_backend_does_not_slow_the_code_that_makes_spans() -> None:
    """2,000 spans while the exporter hangs: making them stays fast; only the export waits."""
    exporter = HangingExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    start = time.perf_counter()
    for i in range(100):
        tree = SpanTree(tracing.tracer(), f"run {i}")
        for j in range(19):
            tree.open("node", f"node {j}")
            tree.close("node")
        tree.finish()
    elapsed = time.perf_counter() - start
    exporter.release.set()
    tracing.close()
    # Measured about 0.05 s on the laptop. The bound is loose on purpose: a
    # blocking exporter would take at least one 10 s wait.
    assert elapsed < 2.0


def test_a_backend_that_refuses_connections_does_not_slow_a_run() -> None:
    """The real OTLP exporter against a closed port: the run is as fast as without tracing."""
    tracing = Tracing.start(CONFIG)  # http://127.0.0.1:1 accepts no connection
    start = time.perf_counter()
    for i in range(50):
        tree = SpanTree(tracing.tracer(), f"run {i}")
        tree.open("node", "node small")
        tree.close("node")
        tree.finish()
    elapsed = time.perf_counter() - start
    tracing.close()
    assert elapsed < 1.0


def test_siblings_opened_under_a_named_parent_close_independently(traced) -> None:
    """Two spans open at once under one node: closing the first leaves the second open."""
    tracing, exporter = traced
    tree = SpanTree(tracing.tracer(), "run")
    tree.open("node", "node investigate")
    tree.open("a", "tool a", kind="tool", under="node")
    tree.open("b", "tool b", kind="tool", under="node")
    tree.close("a", output="result a")
    assert tracing.flush()
    assert [s.name for s in exporter.get_finished_spans()] == ["tool a"]  # b and the node still open
    tree.close("b", output="result b")
    tree.close("node")
    tree.finish()
    spans = finished(tracing, exporter)
    assert parent_name(spans, spans["tool a"]) == parent_name(spans, spans["tool b"]) == "node investigate"
    assert spans["tool a"].attributes[LF_OUTPUT] == "result a"
    assert spans["tool b"].attributes[LF_OUTPUT] == "result b"


def test_a_current_span_parents_spans_made_inside_it_and_restores_the_context(traced) -> None:
    """Code that makes its own spans while a node is current nests them under the node."""
    from opentelemetry import trace

    tracing, exporter = traced
    tree = SpanTree(tracing.tracer(), "run")
    before = trace.get_current_span()
    tree.open("node", "node recall", current=True)
    with tracing.tracer().start_as_current_span("recall rerank"):
        pass
    tree.close("node")
    assert trace.get_current_span() is before
    tree.finish()
    spans = finished(tracing, exporter)
    assert parent_name(spans, spans["recall rerank"]) == "node recall"


def test_a_current_span_closed_from_another_context_ends_quietly(traced, caplog) -> None:
    """LangGraph runs a node in a context copy; a node that raises is closed outside it. No traceback."""
    import contextvars

    tracing, exporter = traced
    tree = SpanTree(tracing.tracer(), "run")
    contextvars.copy_context().run(tree.open, "node", "node large", current=True)
    tree.finish(error="TriageError")
    assert "Failed to detach" not in caplog.text
    assert finished(tracing, exporter)["node large"].status.status_code is StatusCode.ERROR


def test_a_generation_records_the_call_and_marks_a_failure(traced) -> None:
    """The helper for model calls outside an event stream: model, prompt, reply, tokens; errors re-raised."""
    from harness.trace.spans import generation

    tracing, exporter = traced
    with generation(tracing.tracer(), "judge-m", "is it supported?") as gen:
        gen.done('{"supported": true}', input_tokens=40, output_tokens=6, response_model="judge-m")
    with pytest.raises(TimeoutError), generation(tracing.tracer(), "judge-m", "again?", name="judge retry"):
        raise TimeoutError("stalled")
    spans = finished(tracing, exporter)
    ok = spans["model judge-m"].attributes
    assert ok[LF_TYPE] == "generation" and ok[GEN_AI_REQUEST_MODEL] == "judge-m"
    assert ok[GEN_AI_INPUT_TOKENS] == 40 and ok[LF_OUTPUT] == '{"supported": true}'
    assert spans["judge retry"].status.status_code is StatusCode.ERROR


def test_each_judge_call_is_a_generation_under_the_current_metric(traced) -> None:
    """A retried judge question is two generations inside its metric span, with the reported tokens."""
    from pydantic import BaseModel

    from harness.eval.judge import ChatText, make_ollama_judge

    class Verdict(BaseModel):
        """A one-field answer."""

        supported: bool

    replies = iter([ChatText("not json", 30, 2, "judge-m"), ChatText('{"supported": true}', 50, 6, "judge-m")])
    tracing, exporter = traced
    judge = make_ollama_judge("judge-m", chat=lambda m, p, s: next(replies), tracer=tracing.tracer())
    tree = SpanTree(tracing.tracer(), "judge")
    tree.open("metric", "faithfulness", current=True)
    assert judge.ask("is it supported?", Verdict).supported is True
    tree.close("metric")
    tree.finish()
    spans = list(exporter.get_finished_spans()) if tracing.flush() else []
    calls = [s for s in spans if s.name == "judge judge-m"]
    by = {s.context.span_id: s.name for s in spans}
    assert len(calls) == 2 and all(by[c.parent.span_id] == "faithfulness" for c in calls)
    assert sorted(c.attributes[GEN_AI_INPUT_TOKENS] for c in calls) == [30, 50]
