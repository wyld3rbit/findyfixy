"""Tests for run events (app/events.py) and where the graph emits them.

What these tests cover
    That `make_emit` stamps and forwards events, and that a run through the
    graph reports the right events in the right order for the three paths:
    stays small, retries then stays small, and escalates.

Why fakes
    A scripted `ask` and a fake clock. No Ollama, no real time. The event
    order IS the contract the console animation depends on, so it is
    asserted exactly, not loosely.
"""

import json

from app.events import Event, make_emit
from app.graph import build_graph


def reply(confidence: float) -> str:
    """A valid Triage JSON text with the given confidence."""
    return json.dumps(
        {
            "severity": "sev3",
            "component": "auth",
            "next_action": "check the logs",
            "confidence": confidence,
        }
    )


def run(replies_by_model: dict[str, list[str]]) -> list[Event]:
    """Run the graph with scripted replies per model and return the events."""
    events: list[Event] = []
    queues = {m: iter(r) for m, r in replies_by_model.items()}

    def ask(model: str, prompt: str) -> str:
        return next(queues[model])

    ticks = iter(range(1000))
    emit = make_emit(events.append, lambda: float(next(ticks)))
    graph = build_graph(ask, small_model="s", large_model="l", emit=emit)
    graph.invoke({"question": "q"})
    return events


def kinds(events: list[Event]) -> list[str]:
    return [e.kind for e in events]


def test_make_emit_stamps_elapsed_time_from_the_injected_clock():
    got: list[Event] = []
    ticks = iter([10.0, 12.5])
    emit = make_emit(got.append, lambda: next(ticks))
    emit("run_started", question="q")
    assert got == [Event("run_started", 2.5, {"question": "q"})]


def test_confident_small_answer_event_order():
    events = run({"s": [reply(0.9)]})
    assert kinds(events) == [
        "node_entered",
        "model_request",
        "model_reply",
        "validation",
        "node_exited",
        "route",
    ]
    route = events[-1].data
    assert route == {"confidence": 0.9, "threshold": 0.7, "target": "__end__"}


def test_retry_is_reported_between_two_requests():
    events = run({"s": [reply(75), reply(0.9)]})
    assert kinds(events) == [
        "node_entered",
        "model_request",
        "model_reply",
        "validation",
        "retry",
        "model_request",
        "model_reply",
        "validation",
        "node_exited",
        "route",
    ]
    validations = [e.data for e in events if e.kind == "validation"]
    assert [v["ok"] for v in validations] == [False, True]
    assert "confidence" in validations[0]["error"]


def test_escalation_reports_both_nodes_in_order():
    events = run({"s": [reply(0.3)], "l": [reply(0.9)]})
    entered = [e.data["node"] for e in events if e.kind == "node_entered"]
    assert entered == ["small", "large"]
    route = next(e for e in events if e.kind == "route")
    assert route.data["target"] == "large"
    assert kinds(events)[-1] == "node_exited"


def test_timestamps_never_go_backwards():
    events = run({"s": [reply(0.3)], "l": [reply(0.9)]})
    times = [e.t for e in events]
    assert times == sorted(times)
