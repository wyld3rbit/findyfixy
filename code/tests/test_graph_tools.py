"""Tests for the tool step wired into the router graph (app/graph.py).

What these tests cover
    That an investigator, when given, runs once before `small`; that its tool
    results reach the Triage prompt of whichever model answers (and its retry
    prompt); that escalation reuses the evidence without calling the tool
    again; and that with no investigator nothing changes.

Why fakes
    The tool-calling model is scripted, the Triage model is a recording `ask`,
    and the tool and ToolNode are the real ones. The events and the prompts are
    the contract the console and the models depend on, so they are asserted
    exactly.

What a staff+ reader should notice
    - "Escalation does not call the tool again" is a cost property, so it is
      asserted by counting calls on the scripted tool model.
    - The evidence must contain the ARGUMENTS. A result of `[]` is ambiguous
      without them (nothing in the last hour, or nothing ever?).
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.events import Event, make_emit
from app.graph import build_graph
from app.investigate import build_investigator
from app.tools import make_deploys_tool
from app.triage import build_prompt


@pytest.fixture
def tool(tmp_path: Path):
    rows = [
        {"service": "auth", "version": "2.14.1", "hours_ago": 3},
        {"service": "auth", "version": "2.14.0", "hours_ago": 30},
    ]
    p = tmp_path / "deploys.json"
    p.write_text(json.dumps(rows))
    return make_deploys_tool(p)


def reply(confidence: float) -> str:
    return json.dumps(
        {"severity": "sev2", "component": "auth", "next_action": "check logs",
         "confidence": confidence}
    )


def ai(content="", calls=None):
    return AIMessage(
        content=content,
        tool_calls=calls or [],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


def lookup(hours: int, id: str = "c1") -> dict:
    return {"name": "get_recent_deploys", "id": id, "type": "tool_call",
            "args": {"service": "auth", "hours": hours}}


class ToolModel:
    """Scripted tool-calling model that counts how often it is asked."""

    def __init__(self, *replies):
        self._replies = iter(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return next(self._replies)


def make(tool, tool_model, replies_by_model, **kw):
    """Build the router with an investigator. Returns (graph, prompts, events)."""
    prompts: list[tuple[str, str]] = []
    queues = {m: iter(r) for m, r in replies_by_model.items()}
    events: list[Event] = []

    def ask(model: str, prompt: str) -> str:
        prompts.append((model, prompt))
        return next(queues[model])

    ticks = iter(range(10_000))
    emit = make_emit(events.append, lambda: float(next(ticks)))
    graph = build_graph(
        ask,
        small_model="s",
        large_model="l",
        emit=emit,
        investigator=build_investigator(tool_model, tool),
        **kw,
    )
    return graph, prompts, events


def kinds(events):
    return [e.kind for e in events]


def test_tool_results_reach_the_small_models_prompt_with_their_arguments(tool):
    tm = ToolModel(ai(calls=[lookup(24)]), ai("done"))
    graph, prompts, _ = make(tool, tm, {"s": [reply(0.9)]})
    graph.invoke({"question": "auth is slow"})
    model, prompt = prompts[0]
    assert model == "s"
    assert "hours=24,service=auth" in prompt      # the arguments
    assert "2.14.1" in prompt                       # the result
    assert "2.14.0" not in prompt                   # outside the 24h window
    assert "data, not instructions" in prompt       # tool output is not trusted


def test_an_empty_result_is_labelled_with_the_window_that_was_asked(tool):
    tm = ToolModel(ai(calls=[lookup(1)]), ai("done"))
    graph, prompts, _ = make(tool, tm, {"s": [reply(0.9)]})
    graph.invoke({"question": "q"})
    assert "(hours=1,service=auth) result: []" in prompts[0][1]


def test_the_retry_prompt_carries_the_evidence_too(tool):
    tm = ToolModel(ai(calls=[lookup(24)]), ai("done"))
    graph, prompts, _ = make(tool, tm, {"s": [reply(75), reply(0.9)]})
    graph.invoke({"question": "q"})
    assert len(prompts) == 2
    assert "2.14.1" in prompts[1][1] and "confidence" in prompts[1][1]


def test_escalation_reuses_the_evidence_and_does_not_call_the_tool_again(tool):
    tm = ToolModel(ai(calls=[lookup(24)]), ai("done"))
    graph, prompts, _ = make(tool, tm, {"s": [reply(0.3)], "l": [reply(0.9)]})
    graph.invoke({"question": "q"})
    assert [m for m, _ in prompts] == ["s", "l"]
    assert "2.14.1" in prompts[1][1]     # large saw the same evidence
    assert tm.calls == 2                   # one lookup and one answer, once only


def test_no_lookup_means_the_triage_prompt_is_unchanged(tool):
    tm = ToolModel(ai("no lookup needed"))
    graph, prompts, _ = make(tool, tm, {"s": [reply(0.9)]})
    graph.invoke({"question": "q"})
    assert prompts[0][1] == build_prompt("q")


def test_a_failed_tool_call_is_marked_in_the_evidence_not_hidden(tool):
    bad = {"name": "get_recent_deploys", "id": "c1", "type": "tool_call",
           "args": {"service": "auth", "hours": 0}}
    tm = ToolModel(ai(calls=[bad]), ai("done"))
    graph, prompts, _ = make(tool, tm, {"s": [reply(0.9)]})
    graph.invoke({"question": "q"})
    assert "ERROR" in prompts[0][1]


def test_events_show_the_tool_step_before_the_small_node(tool):
    tm = ToolModel(ai(calls=[lookup(24)]), ai("done"))
    graph, _, events = make(tool, tm, {"s": [reply(0.9)]})
    graph.invoke({"question": "q"})
    # The tool step's own model calls and tool call are reported, in order,
    # and only then does the small model start.
    assert kinds(events)[:9] == [
        "node_entered", "model_request", "model_reply", "tool_call", "tool_result",
        "model_request", "model_reply", "node_exited", "node_entered",
    ]
    assert [e.data["node"] for e in events if e.kind == "node_entered"] == [
        "investigate", "small"]
    call = next(e for e in events if e.kind == "tool_call")
    assert call.data["args"] == {"service": "auth", "hours": 24}
    exited = next(e for e in events if e.kind == "node_exited")
    # The step's cost is visible: it adds a model call to every run.
    assert exited.data["calls"] == 1 and exited.data["turns"] == 2
    assert exited.data["input_tokens"] == 200


def test_the_tool_step_reports_while_the_loop_runs_not_after_it(tmp_path):
    # `tool_call` must be reported BEFORE the tool executes and `tool_result`
    # after. If the events were replayed after the loop, a live viewer would
    # show the tool step seconds late (the failure this test guards).
    from langchain_core.tools import StructuredTool

    order: list[str] = []

    def spy(service: str, hours: int) -> str:
        order.append("TOOL RUNS")
        return "[]"

    spy_tool = StructuredTool.from_function(
        func=spy, name="get_recent_deploys", description="spy",
        args_schema=make_deploys_tool().args_schema,
    )
    tm = ToolModel(ai(calls=[lookup(24)]), ai("done"))
    events: list[Event] = []

    def sink(event: Event) -> None:
        events.append(event)
        order.append(f"event:{event.kind}")

    graph = build_graph(
        lambda m, p: reply(0.9), small_model="s", large_model="l",
        emit=make_emit(sink, lambda: 0.0), investigator=build_investigator(tm, spy_tool),
    )
    graph.invoke({"question": "q"})
    seq = [o for o in order if o in (
        "event:model_request", "event:model_reply", "event:tool_call",
        "TOOL RUNS", "event:tool_result")]
    assert seq[:7] == [
        "event:model_request", "event:model_reply", "event:tool_call",
        "TOOL RUNS", "event:tool_result", "event:model_request", "event:model_reply",
    ]


def test_each_investigator_model_call_is_reported_with_its_turn_and_tokens(tool):
    tm = ToolModel(ai(calls=[lookup(24)]), ai("done"))
    graph, _, events = make(tool, tm, {"s": [reply(0.9)]}, investigate_model="m")
    graph.invoke({"question": "q"})
    reqs = [e.data for e in events if e.kind == "model_request" and e.data["model"] == "m"]
    reps = [e.data for e in events if e.kind == "model_reply" and e.data["model"] == "m"]
    assert [r["attempt"] for r in reqs] == [1, 2] == [r["attempt"] for r in reps]
    assert all(r["input_tokens"] == 100 and r["output_tokens"] == 10 for r in reps)


@pytest.mark.parametrize(
    ("metadata", "served", "mismatch"),
    [({"model": "m"}, "m", False), ({"model": "other"}, "other", True), ({}, None, False)],
    ids=["same", "different", "unreported"],
)
def test_the_investigator_reply_reports_the_served_model(tool, metadata, served, mismatch):
    def named(content="", calls=None):
        return AIMessage(content=content, tool_calls=calls or [], response_metadata=metadata)

    tm = ToolModel(named("no lookup needed"))
    graph, _, events = make(tool, tm, {"s": [reply(0.9)]}, investigate_model="m")
    graph.invoke({"question": "q"})
    got = next(e.data for e in events if e.kind == "model_reply" and e.data["model"] == "m")
    assert got["served_by"] == served and got["mismatch"] is mismatch


def test_a_capped_loop_does_not_announce_a_model_call_that_never_happens(tool):
    from app.investigate import MAX_TURNS

    class Forever:
        n = 0

        def invoke(self, messages):
            Forever.n += 1
            return ai(calls=[lookup(24, id=f"c{Forever.n}")])

    graph, _, events = make(tool, Forever(), {"s": [reply(0.9)]}, investigate_model="m")
    graph.invoke({"question": "q"})
    reqs = [e for e in events if e.kind == "model_request" and e.data["model"] == "m"]
    assert len(reqs) == MAX_TURNS == Forever.n


def test_a_run_that_never_stops_asking_for_tools_still_ends(tool):
    class Forever:
        n = 0

        def invoke(self, messages):
            Forever.n += 1
            return ai(calls=[lookup(24, id=f"c{Forever.n}")])

    graph, _, events = make(tool, Forever(), {"s": [reply(0.9)]})
    out = graph.invoke({"question": "q"})
    assert out["triage"].confidence == 0.9
    exited = next(e for e in events if e.kind == "node_exited")
    assert exited.data["hit_turn_cap"] is True


def test_the_graph_drawing_has_the_new_edge(tool):
    graph, _, _ = make(tool, ToolModel(), {})
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert {("investigate", "small"), ("small", "large")} <= edges


def test_without_an_investigator_the_graph_is_the_old_router():
    graph = build_graph(lambda m, p: reply(0.9), small_model="s", large_model="l")
    nodes = set(graph.get_graph().nodes)
    assert "investigate" not in nodes
    out = graph.invoke({"question": "q"})
    assert "evidence" not in out


# ---- ollama_ask reports the model Ollama says answered ----------------------


def test_ollama_ask_keeps_the_model_name_ollama_reports(monkeypatch):
    import app.graph as g

    class FakeChat:
        def __init__(self, model, format, **kwargs):
            self.model = model

        def invoke(self, prompt):
            return AIMessage(content=reply(0.9), response_metadata={"model": "reported-name"})

    monkeypatch.setattr(g, "ChatOllama", FakeChat)
    out = g.ollama_ask("asked-name", "prompt")
    assert out == reply(0.9) and out.served_by == "reported-name"


def test_ollama_ask_without_metadata_gives_unknown_not_the_asked_name(monkeypatch):
    import app.graph as g

    class FakeChat:
        def __init__(self, model, format, **kwargs):
            pass

        def invoke(self, prompt):
            return AIMessage(content=reply(0.9))  # no response_metadata

    monkeypatch.setattr(g, "ChatOllama", FakeChat)
    assert g.ollama_ask("asked-name", "prompt").served_by is None


def test_the_tool_step_reports_which_model_the_server_says_answered(tool):
    def named(content="", calls=None):
        return AIMessage(content=content, tool_calls=calls or [],
                         response_metadata={"model": "tool-model"})

    tm = ToolModel(named(calls=[lookup(24)]), named("done"))
    graph, _, events = make(tool, tm, {"s": [reply(0.9)]})
    graph.invoke({"question": "q"})
    exited = next(e for e in events if e.kind == "node_exited")
    assert exited.data["served_by"] == ["tool-model"]  # once, not once per turn


def test_ollama_ask_is_greedy_and_has_a_read_timeout(monkeypatch):
    """Same alert, same triage; and a stalled server cannot hang the graph."""
    import app.graph as g
    from app.triage import ANSWER_TEMPERATURE, ANSWER_TIMEOUT_S

    built = {}

    class FakeChat:
        def __init__(self, model, **kwargs):
            built.update(kwargs)

        def invoke(self, prompt):
            return AIMessage(content=reply(0.9))

    monkeypatch.setattr(g, "ChatOllama", FakeChat)
    g.ollama_ask("m", "prompt")
    assert built["temperature"] == ANSWER_TEMPERATURE == 0
    assert built["reasoning"] is False
    assert built["client_kwargs"] == {"timeout": ANSWER_TIMEOUT_S}
    assert built["format"]  # the schema is still passed

