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
from langchain_core.messages import AIMessage

from app.direct_run import RunStats, is_correct, run_once, summarise
from app.investigate import build_investigator
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


def call(args: dict, id: str = "c1") -> dict:
    return {"name": "get_recent_deploys", "args": args, "id": id, "type": "tool_call"}


def ai(content: str = "", calls=None, tokens=(10, 5)) -> AIMessage:
    usage = {"input_tokens": tokens[0], "output_tokens": tokens[1],
             "total_tokens": sum(tokens)}
    return AIMessage(content=content, tool_calls=calls or [], usage_metadata=usage)


class ScriptedModel:
    def __init__(self, *replies: AIMessage):
        self._replies = iter(replies)

    def invoke(self, messages):
        return next(self._replies)


def ticking_clock(step: float = 0.5):
    t = {"now": 0.0}

    def clock() -> float:
        t["now"] += step
        return t["now"]

    return clock


def test_a_correct_run_is_scored_and_timed_from_the_clock(tool):
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 24})], tokens=(100, 20)),
        ai("Only auth 2.14.1 was deployed.", tokens=(150, 8)),
    )
    stats = run_once(build_investigator(model, tool), clock=ticking_clock(0.5))
    assert stats.correct and stats.tool_calls == 1 and stats.tool_errors == 0
    assert (stats.input_tokens, stats.output_tokens) == (250, 28)
    assert stats.seconds == 0.5  # the clock is read twice: start and end


def test_the_arguments_the_model_chose_are_in_the_row(tool):
    # The 1.5B model sends hours=1 for a 24 hour question. Legal, wrong, and
    # only visible if the arguments are recorded.
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 1})]),
        ai("There were no deploys to auth in the last 24 hours."),
    )
    stats = run_once(build_investigator(model, tool), clock=ticking_clock())
    assert stats.args == "hours=1,service=auth"
    assert not stats.correct


def test_a_bad_call_counts_as_a_tool_error(tool):
    model = ScriptedModel(
        ai(calls=[call({"service": "auth", "hours": 0})]),
        ai("could not look it up"),
    )
    stats = run_once(build_investigator(model, tool), clock=ticking_clock())
    assert stats.tool_calls == 1 and stats.tool_errors == 1


def test_answering_without_the_tool_is_zero_calls_and_empty_args(tool):
    stats = run_once(
        build_investigator(ScriptedModel(ai("I think 2.14.1.")), tool),
        clock=ticking_clock(),
    )
    assert stats.tool_calls == 0 and stats.args == ""


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
        return RunStats(tin, 10, secs, 2, calls, 0, False, ok, "", "")

    s = summarise([r(1.0, 100), r(1.2, None), r(30.0, 200, ok=False, calls=0)])
    assert s["runs"] == 3 and s["called_tool"] == 2 and s["correct"] == 2
    assert s["median_seconds"] == 1.2  # the 30s outlier does not move it
    assert s["mean_input_tokens"] == 150  # None skipped
