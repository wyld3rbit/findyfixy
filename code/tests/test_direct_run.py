"""Tests for the direct tool-calling loop (app/direct_run.py).

What these tests cover
    The loop's logic: tool calls are executed and their results are fed back,
    bad calls become error messages and not crashes, the turn cap holds, and
    token, time and correctness accounting is right.

Why a scripted fake model
    `run_once` takes a model as an argument. The fake replays prepared
    `AIMessage`s (with and without `tool_calls`) and records the messages it
    was shown. No Ollama, no randomness. Whether a real 1.5B model calls the
    tool correctly is a live measurement, not a unit test.

What a staff+ reader should notice
    - The cap test uses a model that never stops asking for tools. Without the
      cap it would not finish. The assertion is on the bound, not the speed.
    - The error path asserts the model is SHOWN the error. A swallowed
      exception would pass a "did not crash" test and still be useless.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.direct_run import (
    MAX_TURNS,
    RunStats,
    execute_tool_call,
    is_correct,
    run_once,
    summarise,
)
from app.tools import make_deploys_tool


@pytest.fixture
def tools(tmp_path: Path):
    rows = [
        {"service": "auth", "version": "2.14.1", "hours_ago": 3},
        {"service": "auth", "version": "2.14.0", "hours_ago": 30},
    ]
    p = tmp_path / "deploys.json"
    p.write_text(json.dumps(rows))
    tool = make_deploys_tool(p)
    return {tool.name: tool}


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


def ticking_clock(step: float = 0.5):
    t = {"now": 0.0}

    def clock() -> float:
        t["now"] += step
        return t["now"]

    return clock


def test_tool_result_is_sent_back_and_the_answer_is_scored(tools):
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 24})]),
        ai("Only auth 2.14.1 was deployed."),
    )
    stats = run_once(model, tools, clock=ticking_clock())
    assert stats.tool_calls == 1 and stats.tool_errors == 0
    assert stats.turns == 2 and stats.correct
    # The second model call was shown the real tool result.
    tool_msgs = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "2.14.1" in tool_msgs[0].content
    assert "2.14.0" not in tool_msgs[0].content  # 30h is outside the 24h window


def test_tokens_are_summed_across_turns_and_time_comes_from_the_clock(tools):
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 24})], tokens=(100, 20)),
        ai("2.14.1", tokens=(150, 8)),
    )
    stats = run_once(model, tools, clock=ticking_clock(0.5))
    assert (stats.input_tokens, stats.output_tokens) == (250, 28)
    assert stats.seconds == 0.5  # clock read twice: start and end


def test_missing_usage_makes_tokens_unknown_not_a_partial_sum(tools):
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 24})], tokens=(100, 20)),
        ai("2.14.1", tokens=None),
    )
    stats = run_once(model, tools, clock=ticking_clock())
    assert stats.input_tokens is None and stats.output_tokens is None


def test_answering_without_the_tool_is_recorded_as_zero_calls(tools):
    model = ScriptedModel(ai("I think it was 2.14.1."))
    stats = run_once(model, tools, clock=ticking_clock())
    assert stats.tool_calls == 0 and stats.turns == 1


@pytest.mark.parametrize(
    "bad_args",
    [
        {"service": "auth", "hours": 0},
        {"service": "auth", "hours": 9999},
        {"service": "../../etc/passwd", "hours": 24},
        {"service": "auth"},
    ],
)
def test_bad_arguments_are_shown_to_the_model_not_raised(tools, bad_args):
    model = ScriptedModel(ai(calls=[call(bad_args)]), ai("could not look it up"))
    stats = run_once(model, tools, clock=ticking_clock())
    assert stats.tool_errors == 1
    shown = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert shown[0].status == "error" and shown[0].content.startswith("error:")


def test_unknown_tool_name_is_an_error_message(tools):
    msg = execute_tool_call(tools, call({}, name="rm_rf"))
    assert msg.status == "error" and "unknown tool" in msg.content


def test_a_model_that_never_stops_is_cut_off_at_the_turn_cap(tools):
    class Forever:
        count = 0

        def invoke(self, messages):
            Forever.count += 1
            return ai(calls=[call({"service": "auth", "hours": 24}, id=f"c{Forever.count}")])

    stats = run_once(Forever(), tools, clock=ticking_clock())
    assert Forever.count == MAX_TURNS
    assert stats.turns == MAX_TURNS and stats.hit_turn_cap
    assert not stats.correct


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("auth 2.14.1 only", True),
        ("auth 2.14.1 and 2.14.0", False),  # includes the stale deploy
        ("no deploys found", False),
    ],
)
def test_correctness_check(answer, expected):
    assert is_correct(answer) is expected


def test_summary_uses_median_time_and_skips_unknown_tokens():
    def r(secs, tin, ok=True, calls=1):
        return RunStats(tin, 10, secs, 2, calls, 0, False, ok, "")

    s = summarise([r(1.0, 100), r(1.2, None), r(30.0, 200, ok=False, calls=0)])
    assert s["runs"] == 3 and s["called_tool"] == 2 and s["correct"] == 2
    assert s["median_seconds"] == 1.2  # the 30s outlier does not move it
    assert s["mean_input_tokens"] == 150  # None skipped
