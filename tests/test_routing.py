"""Tests for the Day 1 router: does the right model run for a given confidence?

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
    - The 75 case documents a known weakness instead of hiding it. If Day 2
      adds validation that rejects or rescales it, this test should change
      on purpose, and the diff will show that the behavior changed.
    - The edge test guards the drawing. It fails if `route` loses its
      `Literal` return annotation.
"""

import pytest

from app.graph import Answer, build_graph

SMALL = "small-model"
LARGE = "large-model"


def run(small_confidence: float, large_confidence: float = 0.9):
    """Run the graph once with a fake `ask`. Return (models called, final state)."""
    calls: list[str] = []

    def fake_ask(model: str, question: str) -> Answer:
        calls.append(model)
        conf = small_confidence if model == SMALL else large_confidence
        return Answer(answer=f"stub from {model}", confidence=conf)

    graph = build_graph(fake_ask, small_model=SMALL, large_model=LARGE)
    return calls, graph.invoke({"question": "anything"})


@pytest.mark.parametrize(
    "small_confidence",
    [0.7, 0.9, 1.0, 75.0],
    ids=["exactly-threshold", "confident", "max", "percent-scale-passes"],
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
    assert state["answer"] == f"stub from {LARGE}"


def test_large_answer_is_final_even_if_unsure():
    # Documents a known gap: there is no third tier yet. The large model's
    # answer is returned however low its own confidence is.
    calls, state = run(small_confidence=0.1, large_confidence=0.1)
    assert calls == [SMALL, LARGE]
    assert state["confidence"] == 0.1


def test_graph_drawing_shows_both_branches():
    # LangGraph infers a conditional edge's targets from `route`'s return
    # annotation. This fails if the Literal annotation is dropped.
    graph = build_graph(lambda m, q: Answer(answer="", confidence=1), small_model=SMALL, large_model=LARGE)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("small", "large") in edges
    assert ("small", "__end__") in edges
