"""Tests for "everything is observable" (Day 6, step 5): command traces, user, sessions, requests.

What these tests cover
    `CommandTrace` is a no-op without tracing and one trace with steps inside
    with it; a failed command or step is an error trace, re-raised. Every
    trace's top span carries `user.id` (and only the top span). A hunt's runs
    share one session. The console's HTTP requests are traces with method,
    path and status, except the dev poll. A tool path run and an MCP tool
    call record what they cost and did.

What a staff+ reader should notice
    The user lands on top spans through a span processor in the provider,
    not at call sites; the test makes a raw OpenTelemetry span to prove a
    call site that knows nothing about users still gets one.
"""

from __future__ import annotations

import asyncio
import http.client
import threading

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app.events import Event
from app.trace_adapter import EventSpans
from harness.trace.command import CommandTrace
from harness.trace.tracing import TraceConfig, Tracing

CONFIG = TraceConfig(host="http://127.0.0.1:1", public_key="pk-lf-t", secret_key="sk-lf-t",
                     service_name="t", user_id="oncall-7")


@pytest.fixture
def traced():
    """A Tracing to memory and its exporter."""
    exporter = InMemorySpanExporter()
    tracing = Tracing.start(CONFIG, exporter=exporter)
    yield tracing, exporter
    tracing.close()


def spans(tracing: Tracing, exporter: InMemorySpanExporter) -> dict:
    """Flush, then the finished spans by name."""
    assert tracing.flush()
    return {s.name: s for s in exporter.get_finished_spans()}


def test_a_command_without_tracing_runs_the_same_and_records_nothing() -> None:
    """Off is a no-op: steps run, output is accepted, nothing is imported or sent."""
    with CommandTrace(None, "ingest") as command:
        with command.step("embed", {"texts": 3}) as step:
            step.set(done=True)
        command.output("written 3")
    assert command.tracer is None


def test_a_command_is_one_trace_with_its_steps_inside(traced) -> None:
    """The command's top span is current, so steps (and anything else made in the block) nest under it."""
    tracing, exporter = traced
    with CommandTrace(tracing, "ingest", input="200 incidents") as command:
        with command.step("embed", {"texts": 64}) as step:
            step.set(seconds=1.5, model=None)
        with tracing.tracer().start_as_current_span("made by other code"):
            pass
        command.output("written 64")
    by = spans(tracing, exporter)
    root = by["ingest"]
    assert by["embed"].parent.span_id == root.context.span_id
    assert by["made by other code"].parent.span_id == root.context.span_id
    assert by["embed"].attributes["texts"] == 64 and "model" not in by["embed"].attributes
    assert root.attributes["langfuse.observation.output"] == "written 64"


def test_a_failed_command_and_a_failed_step_are_error_traces_and_re_raise(traced) -> None:
    """A failed ingest is a red trace, not a missing one."""
    tracing, exporter = traced
    with pytest.raises(OSError), CommandTrace(tracing, "ingest") as command, command.step("write"):
        raise OSError("database is down")
    by = spans(tracing, exporter)
    assert by["write"].status.status_code is StatusCode.ERROR
    assert by["ingest"].status.status_code is StatusCode.ERROR
    assert "database is down" in by["ingest"].status.description


def test_every_top_span_carries_the_user_and_children_do_not(traced) -> None:
    """The provider adds user.id to any span that starts a trace, even one made with raw OpenTelemetry."""
    tracing, exporter = traced
    tracer = tracing.tracer()
    with tracer.start_as_current_span("raw top"), tracer.start_as_current_span("raw child"):
        pass
    by = spans(tracing, exporter)
    assert by["raw top"].attributes["user.id"] == "oncall-7"
    assert "user.id" not in by["raw child"].attributes


def test_trace_user_comes_from_the_setting_with_local_as_the_default(tmp_path, monkeypatch) -> None:
    """TRACE_USER names who ran it; unset means "local" until Day 7's login."""
    from app.settings import read_tracing

    env = tmp_path / ".env"
    env.write_text("LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-a\nLANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-b\n")
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "TRACING", "TRACE_USER"):
        monkeypatch.delenv(name, raising=False)
    assert read_tracing(stack_env=env).user_id == "local"
    monkeypatch.setenv("TRACE_USER", "oncall-7")
    assert read_tracing(stack_env=env).user_id == "oncall-7"


def test_a_hunts_runs_share_one_session_and_a_later_run_has_none(traced) -> None:
    """hunt_started to hunt_finished: every run in between carries the same session.id."""
    tracing, exporter = traced
    adapter = EventSpans(tracing.tracer())
    events = [("hunt_started", {"max": 2}), ("run_started", {"question": "a"}), ("run_finished", {"triage": {}}),
              ("run_started", {"question": "b"}), ("run_finished", {"triage": {}}), ("hunt_finished", {}),
              ("run_started", {"question": "c"}), ("run_finished", {"triage": {}})]
    for kind, data in events:
        adapter(Event(kind, 0.0, data))
    tracing.flush()
    roots = [s for s in exporter.get_finished_spans() if s.parent is None]
    by_input = {s.attributes["langfuse.observation.input"]: s.attributes.get("session.id") for s in roots}
    assert by_input["a"] == by_input["b"] and by_input["a"].startswith("hunt ")
    assert by_input["c"] is None
    assert adapter.errors == 0


def test_console_requests_are_traces_except_the_dev_poll(traced) -> None:
    """GET /config is a trace with method, path and status; /favicon.ico is not."""
    from app.console import RunHub, serve

    tracing, exporter = traced
    hub = RunHub(live_models=None, sleep=lambda s: None, fake_delay=0, tracing=tracing)
    server = serve(hub, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    for path in ("/config", "/favicon.ico"):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
        conn.getresponse().read()
        conn.close()
    server.shutdown()
    server.server_close()
    by = spans(tracing, exporter)
    assert "GET /favicon.ico" not in by
    request = by["GET /config"]
    assert request.attributes["http.request.method"] == "GET" and request.attributes["url.path"] == "/config"
    assert request.attributes["http.response.status_code"] == 200


def test_a_tool_path_run_is_a_step_with_what_it_cost(traced) -> None:
    """direct_run's runs land in the command trace with tokens, turns and correctness."""
    from app.direct_run import RunStats, traced_run

    tracing, exporter = traced
    stats = RunStats(input_tokens=900, output_tokens=40, seconds=4.2, turns=2, tool_calls=1, tool_errors=0,
                     hit_turn_cap=False, correct=True, answer="2.14.1 at 3h", args="service=auth | hours=24")
    with CommandTrace(tracing, "tool path direct") as command:
        assert traced_run(command, "run 1", lambda: stats) is stats
    step = spans(tracing, exporter)["run 1"].attributes
    assert (step["input_tokens"], step["turns"], step["correct"]) == (900, 2, True)


def test_an_mcp_tool_call_is_a_trace_of_its_own_with_its_arguments_and_result(traced) -> None:
    """The server process cannot join the client's trace over stdio; each call is its own trace."""
    from app.mcp_server import TOOL_NAME, build_server

    tracing, exporter = traced
    server = build_server(tracing=tracing)
    asyncio.run(server.call_tool(TOOL_NAME, {"service": "auth", "hours": 24}))
    call = spans(tracing, exporter)[f"mcp tool {TOOL_NAME}"]
    assert call.parent is None
    assert '"service": "auth"' in call.attributes["langfuse.observation.input"]
    assert "2.14.1" in call.attributes["langfuse.observation.output"]


def test_the_test_session_never_traces_to_a_live_backend() -> None:
    """conftest turns tracing off for every test and every subprocess it starts."""
    import os

    from app.settings import read_tracing

    assert os.environ["TRACING"] == "off"
    assert read_tracing() is None


def test_the_mcp_server_inherits_the_clients_tracing_choice(monkeypatch) -> None:
    """The server gets a minimal environment from the MCP library; the tracing settings must reach it."""
    from app.mcp_client import server_connection

    monkeypatch.setenv("TRACING", "off")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    env = server_connection()["env"]
    assert env == {"TRACING": "off", "LANGFUSE_HOST": "http://127.0.0.1:3000"}


def test_command_trace_reads_dot_env_before_deciding(monkeypatch) -> None:
    """TRACING=off in code/.env must turn a command's tracing off: the file is loaded first."""
    import app.settings as settings
    from app.tracing import command_trace

    calls: list[str] = []
    monkeypatch.setattr(settings, "load_env", lambda: calls.append("load_env"))
    with command_trace("ingest") as command:
        assert command.tracer is None  # the test session has TRACING=off
    assert calls == ["load_env"]


def test_the_event_stream_is_not_traced_as_a_request() -> None:
    """A viewer's stream would hold a request span open for minutes; the runs it streams are traced."""
    from app.console import QUIET_ROUTES, UNTRACED_ROUTES

    assert "/events" in UNTRACED_ROUTES and set(QUIET_ROUTES) <= set(UNTRACED_ROUTES)
