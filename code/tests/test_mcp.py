"""Tests for the MCP path (app/mcp_server.py, app/mcp_client.py, app/investigate.py).

What these tests cover
    That the MCP server shows the model the SAME tool the direct path does,
    that a real stdio round trip works and fails safely, that results reach the
    prompt as clean text, and that no server process is left running.

Why a real server process
    The transport is the thing under test. A mock would test the mock. Each
    test that needs the server starts it through the same adapter code the
    benchmark uses and stops it again. The tool-calling model is still a
    scripted fake, so there is no Ollama and no randomness.

Why `asyncio.run` and no async test plugin
    MCP tools are async-only. Each test runs one coroutine with `asyncio.run`.
    That avoids adding a package for something the standard library does.

What a staff+ reader should notice
    - The schema-parity test is a fairness guard for the Day 2 measurement. If
      the two tools showed the model different text, the token comparison
      would measure wording, not transport.
    - The error test checks the status, not only the text. The measurement's
      tool-error column depends on it.
"""

import asyncio
import subprocess

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.investigate import (
    analyse,
    build_investigator,
    content_text,
    format_evidence,
)
from app.mcp_client import persistent_tools
from app.tools import DEFAULT_DEPLOYS_PATH, get_recent_deploys, make_deploys_tool


def call(args: dict, id: str = "c1") -> dict:
    return {"name": "get_recent_deploys", "args": args, "id": id, "type": "tool_call"}


class ScriptedModel:
    def __init__(self, *replies):
        self._replies = iter(replies)

    def invoke(self, messages):
        return next(self._replies)


def server_count() -> int:
    """How many MCP server processes are running right now."""
    out = subprocess.run(["pgrep", "-f", "python.* -m app.mcp_server"],
                         capture_output=True, text=True)
    return len(out.stdout.split())


def run(coro):
    return asyncio.run(coro)


# ---- content_text and evidence (no server needed) ---------------------------


def test_content_text_passes_a_string_through():
    assert content_text("[1, 2]") == "[1, 2]"


def test_content_text_joins_text_blocks_and_drops_their_metadata():
    blocks = [{"type": "text", "text": "[1, ", "id": "x"}, {"type": "text", "text": "2]"}]
    assert content_text(blocks) == "[1, 2]"


def test_content_text_keeps_a_non_text_block_visible():
    assert "image" in content_text([{"type": "image", "data": "..."}])


def test_analyse_reports_clean_text_for_block_content():
    msgs = [
        HumanMessage("q"),
        AIMessage(content="", tool_calls=[call({"service": "auth", "hours": 24})]),
        ToolMessage(
            content=[{"type": "text", "text": "[]", "id": "lc_1"}], tool_call_id="c1"
        ),
        AIMessage(content="done"),
    ]
    inv = analyse(msgs)
    assert inv.calls[0].result == "[]"
    assert "'type'" not in format_evidence(inv.calls)


# ---- the real server, over stdio -------------------------------------------


def test_the_mcp_tool_shows_the_model_the_same_schema_as_the_direct_tool():
    direct = make_deploys_tool().args_schema.model_json_schema()
    direct_tool = make_deploys_tool()

    async def go():
        async with persistent_tools() as tools:
            return tools[0]

    # Read the schema inside the session: the tool object stays usable as data.
    async def schema_and_text():
        async with persistent_tools() as tools:
            return tools[0].args_schema, tools[0].description, tools[0].name

    mcp_schema, mcp_description, mcp_name = run(schema_and_text())

    def comparable(schema: dict) -> dict:
        """The parts the model reads. Titles are generated names, not content."""
        return {
            name: {k: v for k, v in prop.items() if k != "title"}
            for name, prop in schema["properties"].items()
        } | {"required": sorted(schema["required"])}

    assert comparable(mcp_schema) == comparable(direct)
    assert mcp_description == direct_tool.description
    assert mcp_name == direct_tool.name


def test_a_stdio_round_trip_returns_what_the_direct_function_returns():
    async def go():
        async with persistent_tools() as tools:
            return await tools[0].ainvoke({"service": "auth", "hours": 24})

    result = content_text(run(go()))
    assert result == get_recent_deploys("auth", 24, path=DEFAULT_DEPLOYS_PATH)


@pytest.mark.parametrize(
    "bad_args",
    [
        {"service": "auth", "hours": 0},
        {"service": "auth", "hours": 9999},
        {"service": "../../etc/passwd", "hours": 24},
    ],
)
def test_invalid_arguments_come_back_as_an_error_message_not_a_crash(bad_args):
    async def go():
        async with persistent_tools() as tools:
            model = ScriptedModel(AIMessage(content="", tool_calls=[call(bad_args)]),
                                  AIMessage(content="could not look it up"))
            final = await build_investigator(model, tools).ainvoke(
                {"messages": [HumanMessage("q")]}
            )
            return final["messages"]

    inv = analyse(run(go()))
    assert inv.calls[0].error and "Error" in inv.calls[0].result
    assert inv.calls[0].result.strip().startswith("Error")  # text, not a block list


def test_the_investigator_runs_with_mcp_tools_through_ainvoke():
    async def go():
        async with persistent_tools() as tools:
            model = ScriptedModel(
                AIMessage(content="", tool_calls=[call({"service": "auth", "hours": 24})]),
                AIMessage(content="Only auth 2.14.1."),
            )
            final = await build_investigator(model, tools).ainvoke(
                {"messages": [HumanMessage("q")]}
            )
            return final["messages"]

    inv = analyse(run(go()))
    assert not inv.calls[0].error and "2.14.1" in inv.calls[0].result
    assert "'type'" not in format_evidence(inv.calls)
    assert inv.answer == "Only auth 2.14.1."


def test_no_server_process_is_left_behind():
    before = server_count()

    async def go():
        async with persistent_tools() as tools:
            assert server_count() == before + 1  # it is running inside the block
            await tools[0].ainvoke({"service": "auth", "hours": 1})

    run(go())
    assert server_count() == before


def test_the_server_stops_even_when_the_caller_raises():
    before = server_count()

    async def go():
        async with persistent_tools():
            raise RuntimeError("caller failed")

    with pytest.raises(BaseException):
        run(go())
    assert server_count() == before
