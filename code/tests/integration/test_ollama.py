"""Integration tests: the real Ollama call, structured output and escalation.

Run with:  uv run pytest -m integration
Needs a running Ollama server and both models pulled. Otherwise every test in
this file is skipped with the reason (see conftest.py).

What these tests assert, and what they refuse to assert
    They assert STRUCTURE: a reply parsed into `Answer`, a non-empty answer, a
    finite confidence, the right model recorded in state, nodes run in order.
    They never assert on answer TEXT or on confidence VALUES.
      - Text is non-deterministic. Asserting on wording makes a flaky test.
      - Confidence is unreliable by design. Day 1 saw 0.95, 9.0 and 100.0 for
        similar answers, so a bound like `<= 1` would fail on a healthy system.
    Answer quality is measured by the Day 5 evaluation harness, not here. The
    division of labour: unit tests check logic, these check the plumbing, evals
    check quality.

Failure triage
    Timeout            server hung or a model is swapping in on a small machine
    OutputParserException  the model returned JSON that does not fit `Answer`
    Connection error   the server died mid-run (a skip covers "not running" at
                       the start; this covers it dying later)
"""

import math

import pytest

from app.graph import QUESTIONS, Answer, build_graph, ollama_ask

# Marks every test in this module. The default run deselects the marker.
pytestmark = pytest.mark.integration

# Generous on purpose. The first call to a model includes loading its weights
# into memory (seconds for 1.5B, longer for 7B on a 16 GB machine under load).
CALL_DEADLINE_S = 120
FULL_RUN_DEADLINE_S = 600


def assert_valid_answer(out: Answer) -> None:
    """Shared structural checks. See the module docstring for what is avoided."""
    assert isinstance(out, Answer)
    assert out.answer.strip(), "model returned an empty answer"
    assert math.isfinite(out.confidence), "confidence must be a finite number"


def test_small_model_returns_parseable_answer(ollama, deadline):
    """The 1.5B model, through constrained decoding, fills the `Answer` shape."""
    with deadline(CALL_DEADLINE_S):
        out = ollama_ask(ollama.small, "Reply with the word ok")
    assert_valid_answer(out)


def test_large_model_returns_parseable_answer(ollama, deadline):
    """Same contract for the 7B model. Catches a missing or broken large path."""
    with deadline(CALL_DEADLINE_S):
        out = ollama_ask(ollama.large, "Reply with the word ok")
    assert_valid_answer(out)


def test_forced_escalation_reaches_the_large_model(ollama, deadline):
    """With an unreachable threshold, one question must run small, then large.

    This is the end-to-end proof that the escalation edge works against real
    models. The wrapper records each call so we can assert on ORDER, not just
    on the final `model_used`, which would also be right if `large` ran alone.
    """
    calls: list[str] = []

    def recording_ask(model: str, question: str) -> Answer:
        calls.append(model)
        return ollama_ask(model, question)

    graph = build_graph(
        recording_ask,
        small_model=ollama.small,
        large_model=ollama.large,
        threshold=1e9,  # no confidence can reach this, so every run escalates
    )
    with deadline(2 * CALL_DEADLINE_S):
        state = graph.invoke({"question": QUESTIONS[0]})

    assert calls == [ollama.small, ollama.large]
    assert state["model_used"] == ollama.large
    assert state["answer"].strip()


def test_full_run_completes_on_the_lesson_questions(ollama, deadline):
    """All five lesson questions finish, each answered by one of the two models.

    We do not assert how many escalate. That depends on model self-grading,
    which is the weakest part of the design (see app/graph.py). The test guards
    against crashes: a validation error, a parse failure, a dead server.
    """
    graph = build_graph(
        ollama_ask, small_model=ollama.small, large_model=ollama.large
    )
    with deadline(FULL_RUN_DEADLINE_S):
        states = [graph.invoke({"question": q}) for q in QUESTIONS]

    assert len(states) == len(QUESTIONS)
    for state in states:
        assert state["model_used"] in {ollama.small, ollama.large}
        assert state["answer"].strip()
