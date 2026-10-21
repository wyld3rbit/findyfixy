"""Tests for the Triage schema and the validate-retry loop (app/triage.py).

What these tests cover
    That a bad reply is rejected, re-asked once with the error text, and
    turned into `TriageError` when it stays bad. Not model quality.

Why they use a fake `ask`
    `ask_validated` takes `ask` as an argument. A test passes a function that
    plays back scripted replies and records every prompt it was given. No
    Ollama, no network. Whether a real 1.5B model fixes itself when shown its
    error is an empirical question for the integration tests, not for here.

What a staff+ reader should notice
    - The retry prompt is asserted to contain the validation error. A retry
      that repeats the same prompt would pass a call-count test and still be
      useless.
    - The call count is bounded. A stubborn model must cost at most 2 calls.
    - Infrastructure errors are not swallowed by the retry.
"""

import json

import pytest

from app.triage import Triage, TriageError, ask_validated, parse_triage

GOOD = json.dumps(
    {
        "severity": "sev3",
        "component": "billing",
        "next_action": "check the queue depth",
        "confidence": 0.8,
    }
)


def scripted(*replies: str):
    """Return (ask, prompts). `ask` plays `replies` in order, logging prompts."""
    prompts: list[str] = []
    it = iter(replies)

    def ask(model: str, prompt: str) -> str:
        prompts.append(prompt)
        return next(it)

    return ask, prompts


def test_valid_reply_is_parsed_with_one_call():
    ask, prompts = scripted(GOOD)
    out = ask_validated(ask, "m", "q")
    assert isinstance(out, Triage)
    assert out.severity == "sev3"
    assert len(prompts) == 1


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({"severity": "sev9", "component": "x", "next_action": "y", "confidence": 0.5}),
        json.dumps({"severity": "sev1", "component": "x", "next_action": "y", "confidence": 75}),
        json.dumps({"severity": "sev1", "component": "x", "next_action": "y", "confidence": -0.1}),
        json.dumps({"severity": "sev1", "component": "x"}),
        "plain prose, no json",
        "",
    ],
    ids=["bad-enum", "percent-scale", "negative", "missing-field", "prose", "empty"],
)
def test_invalid_replies_are_rejected(reply):
    with pytest.raises(Exception) as info:
        parse_triage(reply)
    assert info.type.__name__ == "ValidationError"


def test_boundaries_are_inclusive():
    for c in (0, 1):
        reply = json.dumps({"severity": "sev1", "component": "x", "next_action": "y", "confidence": c})
        assert parse_triage(reply).confidence == c


def test_retry_prompt_carries_the_error_and_the_bad_reply():
    bad = json.dumps({"severity": "sev1", "component": "x", "next_action": "y", "confidence": 75})
    ask, prompts = scripted(bad, GOOD)
    out = ask_validated(ask, "m", "the report")
    assert out.confidence == 0.8
    assert len(prompts) == 2
    assert "the report" in prompts[1]
    assert "confidence" in prompts[1]  # the validation error names the field
    assert "less than or equal to 1" in prompts[1]
    assert '"confidence": 75' in prompts[1]  # the bad reply is echoed


def test_stubborn_model_costs_exactly_two_calls_then_raises():
    ask, prompts = scripted("nope", "still nope")
    with pytest.raises(TriageError) as info:
        ask_validated(ask, "m", "q")
    assert len(prompts) == 2
    assert info.value.attempts == 2
    assert info.value.model == "m"


def test_max_retries_zero_means_no_retry():
    ask, prompts = scripted("nope")
    with pytest.raises(TriageError):
        ask_validated(ask, "m", "q", max_retries=0)
    assert len(prompts) == 1


def test_infrastructure_errors_are_not_retried_or_swallowed():
    def ask(model: str, prompt: str) -> str:
        raise ConnectionError("ollama is down")

    with pytest.raises(ConnectionError):
        ask_validated(ask, "m", "q")


def test_long_bad_reply_is_truncated_in_the_retry_prompt():
    ask, prompts = scripted("x" * 5000, GOOD)
    ask_validated(ask, "m", "q")
    assert len(prompts[1]) < 2500


def test_evidence_is_in_the_first_prompt_and_in_the_retry_prompt():
    ask, prompts = scripted(json.dumps({"severity": "sev3", "component": "x",
                                        "next_action": "y", "confidence": 75}), GOOD)
    ask_validated(ask, "m", "q", evidence="get_recent_deploys(hours=24,service=auth) result: []")
    assert len(prompts) == 2
    assert all("hours=24,service=auth" in p for p in prompts)


def test_no_evidence_leaves_the_prompt_exactly_as_before():
    from app.triage import build_prompt

    assert build_prompt("q") == build_prompt("q", "")
    assert "Tool results" not in build_prompt("q")


# ---- which model the server says answered -----------------------------------


def run_with(reply_obj):
    """Run ask_validated once with `reply_obj` as the reply. Return the events."""
    events: list = []

    def emit(kind, /, **data):
        events.append((kind, data))

    ask_validated(lambda m, p: reply_obj, "qwen2.5:7b", "q", emit=emit)
    return [d for k, d in events if k == "model_reply"][0]


def test_a_reply_is_still_a_string_to_everything_that_reads_it():
    from app.triage import Reply

    r = Reply(GOOD, served_by="qwen2.5:7b")
    assert isinstance(r, str) and r == GOOD and len(r) == len(GOOD)
    assert parse_triage(r).severity == "sev3"
    assert r[:5] == GOOD[:5] and r.served_by == "qwen2.5:7b"


def test_the_served_model_is_recorded_and_a_match_is_not_flagged():
    from app.triage import Reply

    d = run_with(Reply(GOOD, served_by="qwen2.5:7b"))
    assert d["served_by"] == "qwen2.5:7b" and d["mismatch"] is False


def test_a_different_served_model_is_flagged_as_a_mismatch():
    from app.triage import Reply

    d = run_with(Reply(GOOD, served_by="qwen2.5:1.5b"))
    assert d["served_by"] == "qwen2.5:1.5b" and d["mismatch"] is True


def test_an_unreported_model_is_unknown_not_a_match_and_not_a_mismatch():
    d = run_with(GOOD)  # a plain string: a fake, an injected reply
    assert d["served_by"] is None and d["mismatch"] is False
