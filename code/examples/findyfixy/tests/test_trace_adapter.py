"""Tests for FindyFixy's tracing: findyfixy/observability/adapter.py, read_tracing, and traced console runs.

What these tests cover
    Every fake console scenario runs through a traced `RunHub`, so every event
    kind the graph really emits passes through the mapping: the tree has the
    right parents, model calls are generations with their model and tokens,
    the route lands on the root, a failed run is an error, and the mapping
    never swallowed an exception (`errors` stays 0). Settings: tracing is off
    without keys, and the local stack's keys are found without copying them.

What a staff+ reader should notice
    The scenarios are the console's own scripts, not events written for the
    test. When the graph starts emitting a new event kind, these tests feed it
    through the adapter without anyone remembering to add a case.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
import pytest

from findyfixy.console.fakes import FAKE_RECALL
import findyfixy.console.server as console
from findyfixy.console.server import RunHub, RunRequest
from findyfixy.events import Event
from findyfixy.observability.adapter import EventSpans, traced_sink
from findyfixy.settings import read_tracing
from wyld3rness.trace.spans import GEN_AI_INPUT_TOKENS, GEN_AI_OUTPUT_TOKENS, GEN_AI_REQUEST_MODEL, LF_TYPE
from wyld3rness.trace.tracing import TraceConfig, Tracing

CONFIG = TraceConfig(host="http://127.0.0.1:1", public_key="pk-lf-t", secret_key="sk-lf-t", service_name="t")


@pytest.fixture
def traced(monkeypatch):
    """A Tracing to memory, its exporter, and every EventSpans the console builds."""
    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    made: list[EventSpans] = []

    def keep(tracer: Any) -> EventSpans:
        """Build the real adapter and remember it, to read its error count."""
        made.append(EventSpans(tracer))
        return made[-1]

    monkeypatch.setattr(console, "EventSpans", keep)
    yield tracing, exporter, made
    tracing.close()


def run_fake(tracing: Tracing, scenario: str) -> None:
    """One fake console run with tracing and a recall step, waited to the end."""
    hub = RunHub(live_models=None, sleep=lambda s: None, fake_delay=0, tracing=tracing, recall=lambda q: FAKE_RECALL)
    hub.start(RunRequest(mode="fake", scenario=scenario))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        _, done = hub.wait_for(0, 0.5)
        if done:
            return
    raise AssertionError("the fake run did not finish")


def spans_of(tracing: Tracing, exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    """Flush, then every finished span."""
    assert tracing.flush()
    return list(exporter.get_finished_spans())


def parent(spans: list[ReadableSpan], span: ReadableSpan) -> str | None:
    """The parent's name, or None for the root."""
    if span.parent is None:
        return None
    return next(s.name for s in spans if s.context.span_id == span.parent.span_id)


@pytest.mark.parametrize(
    "scenario", ["stay_small", "retry_then_small", "escalate", "large_fails", "tool_lookup", "tool_wrong_window"]
)
def test_every_scenario_maps_without_a_swallowed_error_and_ends_its_trace(traced, scenario: str) -> None:
    """One trace per run, one root, every span ended, no mapping error."""
    tracing, exporter, made = traced
    run_fake(tracing, scenario)
    spans = spans_of(tracing, exporter)
    assert [a.errors for a in made] == [0]
    assert len({s.context.trace_id for s in spans}) == 1
    roots = [s for s in spans if s.parent is None]
    assert [r.name for r in roots] == ["findyfixy run"]
    assert all(s.end_time is not None for s in spans)


def test_an_escalated_run_is_a_tree_of_nodes_and_model_calls(traced) -> None:
    """recall is a retriever; each model call is a generation inside its node; the route is on the root."""
    tracing, exporter, _ = traced
    run_fake(tracing, "escalate")
    spans = spans_of(tracing, exporter)
    by = {s.name: s for s in spans}
    assert parent(spans, by["node recall"]) == "findyfixy run"
    assert by["node recall"].attributes[LF_TYPE] == "retriever"
    assert parent(spans, by["model small"]) == "node small"
    assert parent(spans, by["model large"]) == "node large"
    for name, model in (("model small", "small"), ("model large", "large")):
        assert by[name].attributes[LF_TYPE] == "generation"
        assert by[name].attributes[GEN_AI_REQUEST_MODEL] == model
        assert GEN_AI_INPUT_TOKENS not in by[name].attributes  # fakes report none: unknown, not 0
    root = by["findyfixy run"]
    assert root.attributes["findyfixy.route"] == "large"
    assert root.attributes["findyfixy.model_used"] == "large"
    assert "route" in [e.name for e in root.events]


def test_tool_calls_are_tool_spans_inside_investigate(traced) -> None:
    """The deploy lookup sits under the investigate node, next to the model calls that asked for it."""
    tracing, exporter, _ = traced
    run_fake(tracing, "tool_lookup")
    spans = spans_of(tracing, exporter)
    tools = [s for s in spans if s.attributes.get(LF_TYPE) == "tool"]
    assert tools and all(parent(spans, t) == "node investigate" for t in tools)
    assert all(t.attributes["gen_ai.tool.name"] for t in tools)


def test_a_failed_run_is_an_error_trace(traced) -> None:
    """The large model never validates: the run fails and the root says so."""
    tracing, exporter, _ = traced
    run_fake(tracing, "large_fails")
    root = next(s for s in spans_of(tracing, exporter) if s.parent is None)
    assert root.status.status_code is StatusCode.ERROR
    assert "TriageError" in root.status.description


def test_reported_tokens_reach_the_generation_span() -> None:
    """A live model reports tokens on model_reply; they become gen_ai.usage.*."""
    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    spans = EventSpans(tracing.tracer())
    for kind, data in [
        ("run_started", {"question": "q"}),
        ("node_entered", {"node": "small", "model": "m"}),
        ("model_request", {"model": "m", "attempt": 1, "prompt": "p"}),
        ("model_reply", {"model": "m", "reply": "r", "served_by": "m", "input_tokens": 71, "output_tokens": 62}),
        ("node_exited", {"node": "small", "model": "m", "triage": None}),
        ("run_finished", {"model_used": "m", "triage": {}}),
    ]:
        spans(Event(kind, 0.0, data))
    tracing.flush()
    gen = next(s for s in exporter.get_finished_spans() if s.name == "model m")
    assert gen.attributes[GEN_AI_INPUT_TOKENS] == 71 and gen.attributes[GEN_AI_OUTPUT_TOKENS] == 62
    assert spans.errors == 0
    tracing.close()


def test_a_mapping_bug_never_fails_the_run(monkeypatch) -> None:
    """An exception inside the mapping is counted and swallowed; the next sink still gets the event."""
    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    spans = EventSpans(tracing.tracer())
    seen: list[Event] = []
    sink = traced_sink(spans, seen.append, lambda trace_id: None)

    def broken(event: Event) -> None:
        """A mapping with a bug in it."""
        raise KeyError("node")

    monkeypatch.setattr(spans, "_handle", broken)
    sink(Event("node_entered", 0.0, {}))
    assert spans.errors == 1 and len(seen) == 1
    tracing.close()


# ---- read_tracing -------------------------------------------------------------


def test_tracing_is_off_without_keys(tmp_path: Path, monkeypatch) -> None:
    """No keys in the environment and no stack file: off, and nothing raises."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TRACING"):
        monkeypatch.delenv(name, raising=False)
    assert read_tracing(stack_env=tmp_path / "missing.env") is None


def test_the_local_stacks_keys_are_used_without_copying_them(tmp_path: Path, monkeypatch) -> None:
    """The project keys written by `make obs-secrets` turn tracing on for the local host."""
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "TRACING"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text("LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-a\nLANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-b\n")
    config = read_tracing(stack_env=env)
    assert config is not None and config.host == "http://127.0.0.1:3000"
    assert (config.public_key, config.secret_key) == ("pk-lf-a", "sk-lf-b")


def test_the_environment_wins_and_tracing_off_wins_over_everything(tmp_path: Path, monkeypatch) -> None:
    """Keys set in the environment (CI, another host) beat the stack file; TRACING=off beats both."""
    env = tmp_path / ".env"
    env.write_text("LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-a\nLANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-b\n")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-env")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-env")
    monkeypatch.setenv("LANGFUSE_HOST", "http://tracing.example.com")
    monkeypatch.delenv("TRACING", raising=False)
    config = read_tracing(stack_env=env)
    assert config is not None and config.public_key == "pk-lf-env" and config.host == "http://tracing.example.com"
    monkeypatch.setenv("TRACING", "off")
    assert read_tracing(stack_env=env) is None


def feed(events: list[tuple[str, dict]]) -> tuple[EventSpans, list[ReadableSpan], Tracing]:
    """Feed hand-written events through a fresh adapter; return it and the finished spans."""
    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    spans = EventSpans(tracing.tracer())
    for kind, data in events:
        spans(Event(kind, 0.0, data))
    tracing.flush()
    return spans, list(exporter.get_finished_spans()), tracing


def test_two_tool_calls_from_one_reply_are_siblings_with_their_own_results() -> None:
    """The graph emits both calls, then both results, in call order (graph.py, ToolNode)."""
    adapter, spans, tracing = feed(
        [
            ("run_started", {"question": "q"}),
            ("node_entered", {"node": "investigate", "model": "m"}),
            ("model_request", {"model": "m", "attempt": 1}),
            ("model_reply", {"model": "m", "reply": ""}),
            ("tool_call", {"name": "deploys", "args": {"hours": 1}}),
            ("tool_call", {"name": "deploys", "args": {"hours": 24}}),
            ("tool_result", {"name": "deploys", "result": "one hour", "error": False}),
            ("tool_result", {"name": "deploys", "result": "one day", "error": False}),
            ("node_exited", {"node": "investigate", "model": "m"}),
            ("run_finished", {"model_used": "m", "triage": {}}),
        ]
    )
    tools = sorted((s for s in spans if s.attributes.get(LF_TYPE) == "tool"), key=lambda s: s.start_time)
    assert [parent(spans, t) for t in tools] == ["node investigate", "node investigate"]
    assert [t.attributes["langfuse.observation.output"] for t in tools] == ["one hour", "one day"]
    assert adapter.errors == 0
    tracing.close()


def test_a_tool_call_the_loop_never_ran_ends_as_an_error() -> None:
    """At the turn cap a reply can ask for a tool that never runs: the span must not look like a lookup."""
    adapter, spans, tracing = feed(
        [
            ("run_started", {"question": "q"}),
            ("node_entered", {"node": "investigate", "model": "m"}),
            ("tool_call", {"name": "deploys", "args": {}}),
            ("node_exited", {"node": "investigate", "model": "m", "hit_turn_cap": True}),
            ("run_finished", {"model_used": "m", "triage": {}}),
        ]
    )
    tool = next(s for s in spans if s.attributes.get(LF_TYPE) == "tool")
    assert tool.status.status_code is StatusCode.ERROR and "not run" in tool.status.description
    assert adapter.errors == 0
    tracing.close()


def test_a_mapping_bug_ends_the_trace_as_an_error_not_an_orphan(monkeypatch) -> None:
    """Spans already ended are exported; the root must be too, or they have no parent in the backend."""
    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    adapter = EventSpans(tracing.tracer())
    adapter(Event("run_started", 0.0, {"question": "q"}))
    adapter(Event("node_entered", 0.0, {"node": "small", "model": "m"}))
    monkeypatch.setattr(adapter, "_handle", lambda event: (_ for _ in ()).throw(KeyError("x")))
    adapter(Event("route", 0.0, {}))
    tracing.flush()
    root = next(s for s in exporter.get_finished_spans() if s.parent is None)
    assert adapter.errors == 1 and root.status.status_code is StatusCode.ERROR
    tracing.close()


def test_a_scripted_model_call_has_a_readable_name(traced) -> None:
    """Fake investigate calls carry no model name: the span says scripted, not "model None"."""
    tracing, exporter, _ = traced
    run_fake(tracing, "tool_lookup")
    names = {s.name for s in spans_of(tracing, exporter)}
    assert "model (scripted)" in names and "model None" not in names


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "no"])
def test_every_off_spelling_turns_tracing_off(tmp_path: Path, monkeypatch, value: str) -> None:
    """A user who writes TRACING=false means off; prompts must not flow anyway."""
    env = tmp_path / ".env"
    env.write_text("LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-a\nLANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-b\n")
    monkeypatch.setenv("TRACING", value)
    assert read_tracing(stack_env=env) is None


def test_an_unknown_tracing_value_or_half_a_key_pair_is_refused(monkeypatch) -> None:
    """A typo or a missing secret fails loudly instead of silently changing where traces go."""
    from findyfixy.settings import MissingEnvironment

    monkeypatch.setenv("TRACING", "of")
    with pytest.raises(ValueError, match="TRACING"):
        read_tracing()
    monkeypatch.delenv("TRACING")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-env")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(MissingEnvironment, match="LANGFUSE_SECRET_KEY"):
        read_tracing()


def test_the_local_keys_are_never_sent_to_another_host(tmp_path: Path, monkeypatch) -> None:
    """LANGFUSE_HOST set to a remote backend, no keys in the environment: off, not local keys sent there."""
    env = tmp_path / ".env"
    env.write_text("LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-a\nLANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-b\n")
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TRACING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "https://tracing.example.com")
    assert read_tracing(stack_env=env) is None


# ---- RunTracer: eval cases, one shared emitter ------------------------------------


def test_events_reach_only_the_run_open_on_the_emitting_thread() -> None:
    """Two threads share one wrapped emitter; each run's trace holds only its own thread's events."""
    from findyfixy.observability.adapter import RunTracer

    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    runs = RunTracer(tracing.tracer())
    seen: list[str] = []
    emit = runs.wrap(lambda kind, **data: seen.append(kind))
    barrier = threading.Barrier(2, timeout=5)

    def case(node: str) -> None:
        """One run on this thread: enter and leave one node, interleaved with the other thread."""
        with runs.run(f"question {node}") as spans:
            barrier.wait()
            emit("node_entered", node=node, model=None)
            barrier.wait()
            emit("node_exited", node=node, model=None)
            spans.finish_run(output={"node": node})

    threads = [threading.Thread(target=case, args=(n,)) for n in ("small", "large")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    emit("route", target="x")  # no run open on this thread: emitted, not traced
    tracing.flush()
    spans = list(exporter.get_finished_spans())
    for node in ("small", "large"):
        node_span = next(s for s in spans if s.name == f"node {node}")
        root = next(s for s in spans if s.context.span_id == node_span.parent.span_id)
        assert root.attributes["langfuse.observation.input"] == f"question {node}"
    assert seen.count("node_entered") == 2 and "route" in seen
    tracing.close()


def test_a_case_run_nests_under_the_span_current_on_its_thread() -> None:
    """Inside the harness's case span, the app's run is a child in the same trace and does not rename it."""
    from findyfixy.observability.adapter import RunTracer
    from wyld3rness.trace.spans import LF_TRACE_NAME, SpanTree

    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    case_tree = SpanTree(tracing.tracer(), "eval case g1")
    with case_tree.activate(), RunTracer(tracing.tracer()).run("q") as spans:
        spans.finish_run(output={})
    case_tree.finish()
    tracing.flush()
    by = {s.name: s for s in exporter.get_finished_spans()}
    assert by["findyfixy run"].parent.span_id == by["eval case g1"].context.span_id
    assert by["findyfixy run"].context.trace_id == by["eval case g1"].context.trace_id
    assert LF_TRACE_NAME not in by["findyfixy run"].attributes
    tracing.close()


def test_the_open_run_follows_a_copied_context_to_another_thread() -> None:
    """A node run on an executor with the context copied (as LangGraph does) still lands in the run."""
    import contextvars

    from findyfixy.observability.adapter import RunTracer

    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    runs = RunTracer(tracing.tracer())
    emit = runs.wrap(lambda kind, **data: None)
    with runs.run("q") as spans:
        ctx = contextvars.copy_context()
        worker = threading.Thread(target=ctx.run, args=(emit, "node_entered"), kwargs={"node": "small", "model": None})
        worker.start()
        worker.join()
        emit("node_exited", node="small", model=None)
        spans.finish_run(output={})
    tracing.flush()
    assert "node small" in {s.name for s in exporter.get_finished_spans()}
    tracing.close()
