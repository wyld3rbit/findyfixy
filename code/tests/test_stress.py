"""Tests for the stress settings (app/stress.py).

What these tests cover
    That each stress changes exactly what it claims and nothing else: the
    prompt text for the two prompt stresses, the request options for the other
    two, and that "none" is the unchanged request. Also that a typo cannot
    silently run unstressed.

Why a fake chat client
    `make_ollama_ask` takes the chat constructor as an argument. The fake
    records the options and the prompt it was given and returns a canned reply.
    No Ollama. Whether the REAL model then fails is a live measurement (the
    figures are in the module docstring), not a unit test.

What a staff+ reader should notice
    - The vague-prompt test pins that the sentence it removes is the one in the
      real prompt. If someone rewords the prompt, the stress would silently do
      nothing, and this test fails instead.
    - The percentage note must land where the report ends, not inside the
      instructions. It models untrusted text, which lives in the report.
"""

import pytest
from langchain_core.messages import AIMessage

from app.stress import (
    DESCRIPTIONS,
    PERCENT_NOTE,
    SMALL_CONTEXT_TOKENS,
    STRESSES,
    make_ollama_ask,
    vague,
    with_percent_note,
)
from app.triage import (
    CONFIDENCE_RULE,
    EVIDENCE_MARKER,
    PROMPT,
    RETRY_MARKER,
    build_prompt,
)


class FakeChat:
    """Records how it was built and what it was asked. Returns a canned reply."""

    built: list[dict] = []
    prompts: list[str] = []

    def __init__(self, model, **kwargs):
        FakeChat.built.append({"model": model, **kwargs})

    def invoke(self, prompt):
        FakeChat.prompts.append(prompt)
        return AIMessage(content="{}", response_metadata={"model": "served-name"})


@pytest.fixture(autouse=True)
def reset():
    FakeChat.built, FakeChat.prompts = [], []


def run(stress, prompt="PROMPT"):
    ask = make_ollama_ask(stress, chat=FakeChat)
    return ask("m", prompt), FakeChat.built[-1], FakeChat.prompts[-1]


def test_every_stress_has_a_description_for_the_page():
    assert set(DESCRIPTIONS) == set(STRESSES)


def test_the_stress_removes_the_sentence_that_is_really_in_the_prompt():
    # If the prompt is reworded, `vague` would silently do nothing.
    assert CONFIDENCE_RULE in PROMPT
    assert CONFIDENCE_RULE in build_prompt("q")
    assert CONFIDENCE_RULE not in vague(build_prompt("q"))
    assert vague(build_prompt("q")).endswith("Report: q")  # nothing else was touched


def test_vague_prompt_still_asks_for_a_confidence_field():
    assert "confidence" in vague(build_prompt("q"))


def test_the_percent_note_lands_at_the_end_of_the_report():
    out = with_percent_note(build_prompt("the report"))
    assert out == build_prompt("the report") + PERCENT_NOTE
    assert out.index("Report:") < out.index(PERCENT_NOTE)   # after the instructions


def test_the_percent_note_goes_before_evidence_and_before_a_retry_section():
    with_evidence = build_prompt("r", "get_recent_deploys(...) result: []")
    out = with_percent_note(with_evidence)
    assert out.index(PERCENT_NOTE) < out.index(EVIDENCE_MARKER)
    retry = with_evidence + RETRY_MARKER + "\nPrevious reply: x"
    out = with_percent_note(retry)
    assert out.index(PERCENT_NOTE) < out.index(EVIDENCE_MARKER) < out.index(RETRY_MARKER)


def test_none_is_the_plain_request_with_the_json_constraint():
    reply, built, prompt = run("none")
    assert "format" in built and "num_ctx" not in built and prompt == "PROMPT"


def test_no_format_drops_the_json_constraint_and_nothing_else():
    _, built, prompt = run("no_format")
    assert "format" not in built and "num_ctx" not in built and prompt == "PROMPT"


def test_small_context_sets_a_tiny_window_and_keeps_the_constraint():
    _, built, prompt = run("small_context")
    assert built["num_ctx"] == SMALL_CONTEXT_TOKENS == 256
    assert "format" in built and prompt == "PROMPT"


def test_prompt_stresses_change_the_prompt_and_keep_the_constraint():
    _, built, prompt = run("vague_prompt", build_prompt("q"))
    assert "format" in built and CONFIDENCE_RULE not in prompt
    _, built, prompt = run("percent_log", build_prompt("q"))
    assert "format" in built and PERCENT_NOTE in prompt and CONFIDENCE_RULE in prompt


def test_the_reply_still_reports_which_model_the_server_says_answered():
    reply, _, _ = run("no_format")
    assert reply == "{}" and reply.served_by == "served-name"


@pytest.mark.parametrize("typo", ["vague", "", "NONE", "small-context"])
def test_a_typo_raises_instead_of_silently_running_unstressed(typo):
    with pytest.raises(ValueError):
        make_ollama_ask(typo, chat=FakeChat)


@pytest.mark.parametrize("stress", STRESSES)
def test_every_stress_keeps_temperature_zero_and_the_timeout(stress) -> None:
    """Only the stress may differ from the normal path, never the sampling."""
    make_ollama_ask(stress, chat=FakeChat)("m", "prompt")
    built = FakeChat.built[-1]
    assert built["temperature"] == 0
    assert built["reasoning"] is False
    assert built["client_kwargs"]["timeout"] > 0


def test_no_format_removes_only_the_format() -> None:
    make_ollama_ask("no_format", chat=FakeChat)("m", "prompt")
    assert "format" not in FakeChat.built[-1]
    make_ollama_ask("none", chat=FakeChat)("m", "prompt")
    assert "format" in FakeChat.built[-1]

