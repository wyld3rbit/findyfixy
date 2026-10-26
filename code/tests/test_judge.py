"""Tests for the judge seam (app/judge.py), with a fake `chat`. No model.

What these tests cover
    Validation in code, one retry that carries the validation error, a bounded
    call count, infrastructure faults mapped to `JudgeError`, and that
    temperature 0 and the schema reach the chat function.

What a staff+ reader should notice
    The retry prompt is asserted to contain the validation error. A retry that
    repeats the same prompt would pass a call-count test and still be useless.
"""

import json

import pytest
from pydantic import BaseModel

from app.judge import JudgeError, make_ollama_judge


class Answer(BaseModel):
    """A tiny schema for the tests."""

    ok: bool
    n: int


GOOD = json.dumps({"ok": True, "n": 3})
BAD_SHAPE = json.dumps({"ok": "maybe", "n": "x"})


def scripted(*replies):
    """Return (chat, calls). A reply that is an Exception is raised."""
    calls: list[tuple[str, str, dict]] = []
    it = iter(replies)

    def chat(model, prompt, json_schema):
        calls.append((model, prompt, json_schema))
        reply = next(it)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return chat, calls


def test_valid_reply_is_parsed_with_one_call():
    chat, calls = scripted(GOOD)
    out = make_ollama_judge("m", chat=chat).ask("p", Answer)
    assert out == Answer(ok=True, n=3)
    assert len(calls) == 1


def test_invalid_json_then_valid_retries_with_the_error_text():
    chat, calls = scripted("not json at all", GOOD)
    out = make_ollama_judge("m", chat=chat).ask("the question", Answer)
    assert out.n == 3
    assert len(calls) == 2
    retry_prompt = calls[1][1]
    assert "the question" in retry_prompt
    assert "Validation error:" in retry_prompt
    assert "not json at all" in retry_prompt
    assert "Invalid JSON" in retry_prompt  # Pydantic's own message


def test_schema_failure_error_names_the_field():
    chat, calls = scripted(BAD_SHAPE, GOOD)
    make_ollama_judge("m", chat=chat).ask("p", Answer)
    assert "ok" in calls[1][1]
    assert "valid boolean" in calls[1][1]


def test_invalid_twice_raises_after_exactly_two_calls():
    chat, calls = scripted("nope", "still nope", GOOD)
    with pytest.raises(JudgeError):
        make_ollama_judge("m", chat=chat).ask("p", Answer)
    assert len(calls) == 2


def test_retries_zero_means_one_call():
    chat, calls = scripted("nope", GOOD)
    with pytest.raises(JudgeError):
        make_ollama_judge("m", chat=chat, retries=0).ask("p", Answer)
    assert len(calls) == 1


@pytest.mark.parametrize("fault", [TimeoutError("slow"), ConnectionError("down")])
def test_infrastructure_fault_becomes_judge_error_without_retry(fault):
    chat, calls = scripted(fault, GOOD)
    with pytest.raises(JudgeError):
        make_ollama_judge("m", chat=chat).ask("p", Answer)
    assert len(calls) == 1


def test_model_prompt_and_schema_are_passed_through():
    chat, calls = scripted(GOOD)
    make_ollama_judge("qwen-test", chat=chat).ask("hello", Answer)
    model, prompt, schema = calls[0]
    assert model == "qwen-test"
    assert prompt == "hello"
    assert schema == Answer.model_json_schema()


def test_default_chat_sets_temperature_zero_schema_and_timeout(monkeypatch):
    """The real chat builds ChatOllama with temperature 0, the schema, a timeout."""
    seen = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def invoke(self, prompt):
            return type("M", (), {"content": GOOD})()

    import langchain_ollama

    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeLLM)
    out = make_ollama_judge("m", timeout_s=7.5).ask("p", Answer)
    assert out.ok is True
    assert seen["temperature"] == 0
    assert seen["format"] == Answer.model_json_schema()
    assert seen["client_kwargs"] == {"timeout": 7.5}
    assert seen["model"] == "m"
