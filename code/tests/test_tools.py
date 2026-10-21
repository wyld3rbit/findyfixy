"""Tests for the deploy tool (app/tools.py): behaviour and argument safety.

What these tests cover
    The filter and sort logic, argument validation, and that a ToolNode runs
    the tool from a model-style tool call and survives bad arguments.

Why no model is used
    A ToolNode is fed a hand-built AIMessage with `tool_calls`, exactly the
    shape a model emits. Whether a real 1.5B model chooses to call the tool
    is an empirical question for a live test. Whether the node executes it
    safely is a logic question, tested here.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import ValidationError

from app.tools import (
    DEFAULT_DEPLOYS_PATH,
    MAX_HOURS,
    get_recent_deploys,
    make_deploys_tool,
    make_tools_node,
)


@pytest.fixture
def deploys_file(tmp_path: Path) -> Path:
    """A tiny known data set, so assertions do not depend on the real file."""
    rows = [
        {"service": "auth", "version": "2", "hours_ago": 30},
        {"service": "auth", "version": "3", "hours_ago": 3},
        {"service": "auth", "version": "1", "hours_ago": 140},
        {"service": "billing", "version": "9", "hours_ago": 2},
    ]
    p = tmp_path / "deploys.json"
    p.write_text(json.dumps(rows))
    return p


def versions(text: str) -> list[str]:
    return [r["version"] for r in json.loads(text)]


def test_filters_by_service_and_window_newest_first(deploys_file):
    out = get_recent_deploys("auth", 48, path=deploys_file)
    assert versions(out) == ["3", "2"]


def test_window_boundary_is_inclusive(deploys_file):
    assert versions(get_recent_deploys("auth", 30, path=deploys_file)) == ["3", "2"]
    assert versions(get_recent_deploys("auth", 29, path=deploys_file)) == ["3"]


def test_service_with_no_deploys_in_window_returns_empty_list(deploys_file):
    assert get_recent_deploys("billing", 1, path=deploys_file) == "[]"


@pytest.mark.parametrize(
    ("service", "hours"),
    [
        ("auth", -5),
        ("auth", 0),
        ("auth", MAX_HOURS + 1),
        ("../etc/passwd", 24),
        ("unknown-service", 24),
        ("", 24),
    ],
    ids=["negative", "zero", "too-long", "path-traversal", "unknown", "empty"],
)
def test_bad_arguments_are_rejected(deploys_file, service, hours):
    with pytest.raises(ValidationError):
        get_recent_deploys(service, hours, path=deploys_file)


def test_hours_must_be_an_integer(deploys_file):
    with pytest.raises(ValidationError):
        get_recent_deploys("auth", "lots", path=deploys_file)


def test_shipped_data_file_is_valid_and_covers_every_service():
    # Guards the real file: every record must load and pass the same rules.
    rows = json.loads(DEFAULT_DEPLOYS_PATH.read_text())
    assert rows
    for r in rows:
        assert get_recent_deploys(r["service"], MAX_HOURS, path=DEFAULT_DEPLOYS_PATH)


def call(node, name: str, args: dict) -> str:
    """Run a ToolNode on a model-style tool call. Return the ToolMessage text.

    The node runs inside a one-node graph. Called on its own, this langgraph
    version raises "Missing required config key": the node expects the runtime
    context a graph supplies. That is also how it runs in production.
    """
    g = StateGraph(MessagesState)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    msg = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c1"}])
    out = g.compile().invoke({"messages": [msg]})
    return out["messages"][-1].content


def test_toolnode_runs_the_tool_from_a_tool_call(deploys_file):
    node = make_tools_node(make_deploys_tool(deploys_file))
    text = call(node, "get_recent_deploys", {"service": "auth", "hours": 48})
    assert versions(text) == ["3", "2"]


def test_toolnode_returns_validation_error_to_the_model_not_a_crash(deploys_file):
    node = make_tools_node(make_deploys_tool(deploys_file))
    text = call(node, "get_recent_deploys", {"service": "auth", "hours": -5})
    assert "hours" in text  # the error names the bad field


def test_toolnode_rejects_an_unknown_service_without_reading_files(deploys_file):
    node = make_tools_node(make_deploys_tool(deploys_file))
    text = call(node, "get_recent_deploys", {"service": "../secrets", "hours": 24})
    assert "service" in text
    assert not text.lstrip().startswith("[")  # no data came back


def test_tool_schema_shows_the_model_the_allowed_services():
    schema = make_deploys_tool().args_schema.model_json_schema()
    assert "voice-verification" in json.dumps(schema)
    assert schema["properties"]["hours"]["maximum"] == MAX_HOURS
