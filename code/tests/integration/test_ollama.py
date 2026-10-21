"""Integration tests: the real Ollama call, structured output and escalation.

Run with:  uv run pytest -m integration
Needs a running Ollama server and both models pulled. Otherwise every test in
this file is skipped with the reason (see conftest.py).

What these tests assert, and what they refuse to assert
    They assert STRUCTURE: a reply that validates as `Triage`, non-empty fields,
    a finite confidence, the right model recorded in state, nodes run in order.
    They never assert on answer TEXT or on confidence VALUES.
      - Text is non-deterministic. Asserting on wording makes a flaky test.
      - Confidence is unreliable by design. Day 1 saw 0.95, 9.0 and 100.0 for
        similar answers, so a bound like `<= 1` would fail on a healthy system.
    Answer quality is measured by the Day 5 evaluation harness, not here. The
    division of labour: unit tests check logic, these check the plumbing, evals
    check quality.

Failure triage
    Timeout            server hung or a model is swapping in on a small machine
    TriageError        the model stayed invalid after its one retry (the small
                       model's case escalates; the large model's case raises)
    Connection error   the server died mid-run (a skip covers "not running" at
                       the start; this covers it dying later)
"""

import math

import pytest

from app.graph import QUESTIONS, build_graph, ollama_ask
from app.triage import Ask, Triage, ask_validated

# Marks every test in this module. The default run deselects the marker.
pytestmark = pytest.mark.integration

# Generous on purpose. The first call to a model includes loading its weights
# into memory (seconds for 1.5B, longer for 7B on a 16 GB machine under load).
CALL_DEADLINE_S = 120
FULL_RUN_DEADLINE_S = 600


def assert_valid_triage(out: Triage) -> None:
    """Shared structural checks. See the module docstring for what is avoided."""
    assert isinstance(out, Triage)
    assert out.component.strip(), "model returned an empty component"
    assert out.next_action.strip(), "model returned an empty next_action"
    assert math.isfinite(out.confidence), "confidence must be a finite number"


def test_small_model_returns_parseable_answer(ollama, deadline):
    """The 1.5B model, through constrained decoding, fills the `Triage` shape."""
    with deadline(CALL_DEADLINE_S):
        out = ask_validated(ollama_ask, ollama.small, "Reply with the word ok")
    assert_valid_triage(out)


def test_large_model_returns_parseable_answer(ollama, deadline):
    """Same contract for the 7B model. Catches a missing or broken large path."""
    with deadline(CALL_DEADLINE_S):
        out = ask_validated(ollama_ask, ollama.large, "Reply with the word ok")
    assert_valid_triage(out)


def test_forced_escalation_reaches_the_large_model(ollama, deadline):
    """With an unreachable threshold, one question must run small, then large.

    This is the end-to-end proof that the escalation edge works against real
    models. The wrapper records each call so we can assert on ORDER, not just
    on the final `model_used`, which would also be right if `large` ran alone.
    """
    calls: list[str] = []

    def recording_ask(model: str, prompt: str) -> str:
        calls.append(model)
        return ollama_ask(model, prompt)

    graph = build_graph(
        recording_ask,
        small_model=ollama.small,
        large_model=ollama.large,
        threshold=1e9,  # no confidence can reach this, so every run escalates
    )
    with deadline(2 * CALL_DEADLINE_S):
        state = graph.invoke({"question": QUESTIONS[0]})

    # A retry may add extra small calls, so check order by first and last call.
    assert calls[0] == ollama.small
    assert calls[-1] == ollama.large
    assert state["model_used"] == ollama.large
    assert_valid_triage(state["triage"])


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
        assert_valid_triage(state["triage"])


# A reply that is well-formed JSON but breaks the 0-to-1 rule: the Day 1 "75".
_BAD_SMALL = '{"severity": "sev1", "component": "x", "next_action": "y", "confidence": 75}'
_GOOD_SMALL = '{"severity": "sev1", "component": "x", "next_action": "y", "confidence": 0.9}'


def _forced_small(small_reply: str, ollama, prompts: list[tuple[str, str]]) -> Ask:
    """An `ask` that fakes the SMALL model's reply and calls the real LARGE model.

    Faking only the small side forces one trigger at a time. The large model is
    real, so the escalation path is proven against real constrained decoding.
    Every call is logged as (model, prompt) so a test can assert on ORDER and on
    what the retry prompt contained.
    """

    def ask(model: str, prompt: str) -> str:
        prompts.append((model, prompt))
        return small_reply if model == ollama.small else ollama_ask(model, prompt)

    return ask


def test_invalid_small_output_retries_then_escalates_to_real_large(ollama, deadline):
    """Trigger A, live: small stays invalid after its retry, so the 7B answers.

    Expected order: small, small (retry with the error text), large.
    """
    prompts: list[tuple[str, str]] = []
    graph = build_graph(
        _forced_small(_BAD_SMALL, ollama, prompts),
        small_model=ollama.small,
        large_model=ollama.large,
    )
    with deadline(2 * CALL_DEADLINE_S):
        state = graph.invoke({"question": QUESTIONS[0]})

    assert [m for m, _ in prompts][:3] == [ollama.small, ollama.small, ollama.large]
    retry_prompt = prompts[1][1]
    assert "confidence" in retry_prompt and "less than or equal to 1" in retry_prompt
    assert state["model_used"] == ollama.large
    assert_valid_triage(state["triage"])
    assert 0 <= state["triage"].confidence <= 1


def test_valid_confident_small_output_does_not_escalate(ollama, deadline):
    """Control for the test above: a valid 0.9 reply must stay on the small model.

    Without this, a router that always escalated would pass the test above.
    """
    prompts: list[tuple[str, str]] = []
    graph = build_graph(
        _forced_small(_GOOD_SMALL, ollama, prompts),
        small_model=ollama.small,
        large_model=ollama.large,
    )
    with deadline(CALL_DEADLINE_S):
        state = graph.invoke({"question": QUESTIONS[0]})

    assert [m for m, _ in prompts] == [ollama.small]
    assert state["model_used"] == ollama.small
