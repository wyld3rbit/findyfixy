"""Tests for the router: does the right model run for a given triage reply?

What these tests cover
    The routing wiring and the routing rule. Not model quality.

Why they use a fake `ask`
    `build_graph` takes `ask` as an argument, so a test passes a function that
    returns a chosen confidence and records which models were called. No
    Ollama, no network, no GPU. The whole file runs in well under a second,
    so it can run on every commit. Model quality is a different test layer
    (Day 5 evals) because it needs real models and is slow and noisy.

What a staff+ reader should notice
    - The boundary case (exactly 0.7) is tested. Off-by-one on a threshold is
      the classic routing bug, and `>=` versus `>` changes cost.
    - The 75 case changed on purpose on Day 2. On Day 1 it passed straight
      through as "confident". Now it fails validation, is retried once, and
      then escalates. The diff on this file is the record of that change.
    - An invalid small reply is a routing input, not a crash.
    - The edge test guards the drawing. It fails if `route` loses its
      `Literal` return annotation.
"""

import pytest

import json

from app.graph import build_graph
from app.triage import TriageError

SMALL = "small-model"
LARGE = "large-model"


def reply(confidence: float) -> str:
    """A raw model reply with a chosen confidence. Other fields are legal."""
    return json.dumps(
        {
            "severity": "sev2",
            "component": "auth",
            "next_action": "check recent deploys",
            "confidence": confidence,
        }
    )


def run(small_confidence: float, large_confidence: float = 0.9):
    """Run the graph once with a fake `ask`. Return (models called, final state)."""
    calls: list[str] = []

    def fake_ask(model: str, prompt: str) -> str:
        calls.append(model)
        return reply(small_confidence if model == SMALL else large_confidence)

    graph = build_graph(fake_ask, small_model=SMALL, large_model=LARGE)
    return calls, graph.invoke({"question": "anything"})


@pytest.mark.parametrize(
    "small_confidence",
    [0.7, 0.9, 1.0],
    ids=["exactly-threshold", "confident", "max"],
)
def test_confident_small_answer_is_final(small_confidence):
    calls, state = run(small_confidence)
    assert calls == [SMALL]
    assert state["model_used"] == SMALL


@pytest.mark.parametrize(
    "small_confidence",
    [0.0, 0.2, 0.69],
    ids=["zero", "low", "just-below-threshold"],
)
def test_unsure_small_answer_escalates(small_confidence):
    calls, state = run(small_confidence)
    assert calls == [SMALL, LARGE]
    assert state["model_used"] == LARGE
    assert state["triage"].confidence == 0.9


def test_large_answer_is_final_even_if_unsure():
    # Documents a known gap: there is no third tier yet. The large model's
    # answer is returned however low its own confidence is.
    calls, state = run(small_confidence=0.1, large_confidence=0.1)
    assert calls == [SMALL, LARGE]
    assert state["triage"].confidence == 0.1


def test_percent_scale_confidence_retries_then_escalates():
    # The Day 1 "75" bug. Small is asked twice (first try + one retry), both
    # invalid, so the question moves to large.
    calls, state = run(small_confidence=75.0)
    assert calls == [SMALL, SMALL, LARGE]
    assert state["model_used"] == LARGE


def test_small_that_fixes_itself_on_retry_stays_small():
    replies = iter([reply(75.0), reply(0.9)])
    calls: list[str] = []

    def fake_ask(model: str, prompt: str) -> str:
        calls.append(model)
        return next(replies)

    graph = build_graph(fake_ask, small_model=SMALL, large_model=LARGE)
    state = graph.invoke({"question": "anything"})
    assert calls == [SMALL, SMALL]
    assert state["model_used"] == SMALL


def test_garbage_from_small_escalates_instead_of_crashing():
    calls: list[str] = []

    def fake_ask(model: str, prompt: str) -> str:
        calls.append(model)
        return "not json at all" if model == SMALL else reply(0.9)

    graph = build_graph(fake_ask, small_model=SMALL, large_model=LARGE)
    state = graph.invoke({"question": "anything"})
    assert calls == [SMALL, SMALL, LARGE]
    assert state["model_used"] == LARGE


def test_large_that_stays_invalid_raises():
    # No third tier: an invalid large answer is a loud failure, not a guess.
    graph = build_graph(lambda m, p: "{}", small_model=SMALL, large_model=LARGE)
    with pytest.raises(TriageError):
        graph.invoke({"question": "anything"})


def test_graph_drawing_shows_both_branches():
    # LangGraph infers a conditional edge's targets from `route`'s return
    # annotation. This fails if the Literal annotation is dropped.
    graph = build_graph(lambda m, p: reply(1), small_model=SMALL, large_model=LARGE)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("small", "large") in edges
    assert ("small", "__end__") in edges


@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.69, 0.95])
def test_one_model_never_escalates_to_itself(confidence: float) -> None:
    """Same name for both tiers means one model: asking it again adds latency, not information."""
    calls: list[str] = []

    def fake_ask(model: str, prompt: str) -> str:
        calls.append(model)
        return reply(confidence)

    graph = build_graph(fake_ask, small_model=SMALL, large_model=SMALL)
    state = graph.invoke({"question": "anything"})
    assert calls == [SMALL]
    assert state["model_used"] == SMALL

