"""Tests for the tool-calling graph (app/investigate.py).

What these tests cover
    The loop's logic with the REAL `ToolNode` and the REAL deploy tool: a
    requested tool is executed and its result is shown to the model, bad
    calls become error messages and not crashes, the turn cap holds, and the
    token, call and evidence accounting is right.

Why a scripted fake model
    `build_investigator` takes a model as an argument. The fake replays
    prepared `AIMessage`s and records what it was shown. No Ollama, no
    randomness. Whether a real 1.5B model calls the tool correctly is a live
    measurement (app/direct_run.py), not a unit test.

What a staff+ reader should notice
    - The cap test uses a model that never stops asking for tools. Without
      the cap it would not finish. The assertion is on the bound.
    - The error tests assert the model is SHOWN the error. A swallowed
      exception would pass a "did not crash" test and still be useless.
    - The ToolNode here is the shipped one, not a stand-in, so a change in
      LangGraph's error format shows up as a failure here.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.investigate import (
    MAX_TURNS,
    ToolCallRecord,
    analyse,
    build_investigator,
    format_evidence,
)
from app.tools import make_deploys_tool


@pytest.fixture
def tool(tmp_path: Path):
    rows = [
        {"service": "auth", "version": "2.14.1", "hours_ago": 3},
        {"service": "auth", "version": "2.14.0", "hours_ago": 30},
    ]
    p = tmp_path / "deploys.json"
    p.write_text(json.dumps(rows))
    return make_deploys_tool(p)


def call(args: dict, name: str = "get_recent_deploys", id: str = "c1") -> dict:
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def ai(content: str = "", calls=None, tokens=(10, 5)) -> AIMessage:
    usage = None if tokens is None else {
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        "total_tokens": sum(tokens),
    }
    return AIMessage(content=content, tool_calls=calls or [], usage_metadata=usage)


class ScriptedModel:
    """Plays prepared replies in order and records what it was shown."""

    def __init__(self, *replies: AIMessage):
        self._replies = iter(replies)
        self.seen: list[list] = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return next(self._replies)


def run(model, tool, **kw):
    graph = build_investigator(model, tool, **kw)
    final = graph.invoke({"messages": [HumanMessage("question")]})
    return analyse(final["messages"])


def tool_messages(seen_call: list) -> list[ToolMessage]:
    return [m for m in seen_call if isinstance(m, ToolMessage)]


def test_tool_result_is_sent_back_to_the_model(tool):
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 24})]),
        ai("Only auth 2.14.1 was deployed."),
    )
    inv = run(model, tool)
    assert inv.turns == 2 and len(inv.calls) == 1 and not inv.calls[0].error
    shown = tool_messages(model.seen[1])
    assert len(shown) == 1
    assert "2.14.1" in shown[0].content
    assert "2.14.0" not in shown[0].content  # 30h is outside the 24h window
    assert inv.answer == "Only auth 2.14.1 was deployed."


def test_tokens_are_summed_across_turns(tool):
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 24})], tokens=(100, 20)),
        ai("done", tokens=(150, 8)),
    )
    inv = run(model, tool)
    assert (inv.input_tokens, inv.output_tokens) == (250, 28)


def test_missing_usage_makes_tokens_unknown_not_a_partial_sum(tool):
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 24})], tokens=(100, 20)),
        ai("done", tokens=None),
    )
    inv = run(model, tool)
    assert inv.input_tokens is None and inv.output_tokens is None


def test_answering_without_the_tool_is_zero_calls(tool):
    inv = run(ScriptedModel(ai("no tool needed")), tool)
    assert inv.calls == () and inv.turns == 1 and not inv.hit_turn_cap


@pytest.mark.parametrize(
    "bad_args",
    [
        {"service": "auth", "hours": 0},
        {"service": "auth", "hours": 9999},
        {"service": "../../etc/passwd", "hours": 24},
        {"service": "auth"},
    ],
)
def test_bad_arguments_are_shown_to_the_model_not_raised(tool, bad_args):
    model = ScriptedModel(ai(calls=[call(bad_args)]), ai("could not look it up"))
    inv = run(model, tool)
    assert inv.calls[0].error
    shown = tool_messages(model.seen[1])
    assert len(shown) == 1 and shown[0].status == "error"
    assert "get_recent_deploys" not in shown[0].content or "error" in shown[0].content.lower()


def test_unknown_tool_name_is_an_error_message_not_a_crash(tool):
    model = ScriptedModel(ai(calls=[call({}, name="rm_rf")]), ai("ok"))
    inv = run(model, tool)
    assert inv.calls[0].error
    assert tool_messages(model.seen[1])[0].status == "error"


def test_a_model_that_never_stops_is_cut_off_at_the_turn_cap(tool):
    class Forever:
        count = 0

        def invoke(self, messages):
            Forever.count += 1
            return ai(calls=[call({"service": "auth", "hours": 24}, id=f"c{Forever.count}")])

    inv = run(Forever(), tool)
    assert Forever.count == MAX_TURNS
    assert inv.turns == MAX_TURNS and inv.hit_turn_cap
    # The final request is NOT executed: it has no result, and is flagged.
    assert inv.calls[-1].error and inv.calls[-1].result == ""
    assert all(not c.error for c in inv.calls[:-1])
    assert inv.answer == ""


def test_the_cap_is_configurable(tool):
    class Forever:
        count = 0

        def invoke(self, messages):
            Forever.count += 1
            return ai(calls=[call({"service": "auth", "hours": 24}, id=f"c{Forever.count}")])

    run(Forever(), tool, max_turns=2)
    assert Forever.count == 2


def test_parallel_tool_calls_in_one_message_are_each_recorded(tool):
    model = ScriptedModel(
        ai(calls=[
            call({"service": "auth", "hours": 24}, id="a"),
            call({"service": "auth", "hours": 100}, id="b"),
        ]),
        ai("done"),
    )
    inv = run(model, tool)
    assert [c.args["hours"] for c in inv.calls] == [24, 100]
    assert "2.14.0" in inv.calls[1].result and "2.14.0" not in inv.calls[0].result


def test_graph_drawing_shows_the_loop():
    # LangGraph reads `after_agent`'s Literal return type to draw its edges.
    graph = build_investigator(ScriptedModel(), make_deploys_tool()).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("agent", "tools"), ("tools", "agent"), ("agent", "__end__")} <= edges


# ---- evidence text ----------------------------------------------------------


def test_evidence_names_the_arguments_so_an_empty_result_is_readable():
    calls = (ToolCallRecord("get_recent_deploys", {"service": "auth", "hours": 1}, "[]", False),)
    text = format_evidence(calls)
    assert "hours=1,service=auth" in text  # keys sorted, so output is stable and text.endswith("result: []")


def test_evidence_marks_errors_and_clips_long_results():
    calls = (
        ToolCallRecord("t", {"x": 1}, "boom", True),
        ToolCallRecord("t", {"x": 2}, "y" * 5000, False),
    )
    lines = format_evidence(calls).splitlines()
    assert "ERROR: boom" in lines[0]
    assert len(lines[1]) < 700


def test_no_calls_means_no_evidence():
    assert format_evidence(()) == ""
