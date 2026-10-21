"""Tests for the realistic incident bank (app/incidents.py, data/incidents.json).

What these tests cover
    That the shipped file loads, that every incident fits the model's context
    window, and that it is synthetic. They do not judge whether a model solves
    the incidents: that is a live measurement.

Why the size test matters
    Ollama silently cuts a prompt that overflows its 4096-token window from
    the start. No error, just a worse answer. The cap keeps the retry prompt
    (about double) inside the window.
"""

import re
from collections import Counter

import pytest

from app.incidents import MAX_TEXT_CHARS, Incident, load_incidents

BANK = load_incidents()

# Documentation ranges (RFC 5737). Safe to publish, never routable.
_ALLOWED_IP_PREFIXES = ("192.0.2.", "198.51.100.", "203.0.113.")
_IPV4 = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def test_bank_has_eight_unique_incidents():
    assert len(BANK) == 8
    assert not [i for i, n in Counter(b.id for b in BANK).items() if n > 1]


@pytest.mark.parametrize("incident", BANK, ids=lambda i: i.id)
def test_incident_fits_the_context_window(incident: Incident):
    assert len(incident.text) <= MAX_TEXT_CHARS


@pytest.mark.parametrize("incident", BANK, ids=lambda i: i.id)
def test_incident_is_realistic_not_a_one_liner(incident: Incident):
    # At least 10 lines and some timestamps: real evidence to work through.
    assert incident.text.count("\n") >= 10
    assert len(re.findall(r"2026-\d\d-\d\dT", incident.text)) >= 5


@pytest.mark.parametrize("incident", BANK, ids=lambda i: i.id)
def test_incident_uses_only_documentation_ip_addresses(incident: Incident):
    stray = [
        ip for ip in _IPV4.findall(incident.text)
        if not ip.startswith(_ALLOWED_IP_PREFIXES)
    ]
    assert stray == []


@pytest.mark.parametrize("incident", BANK, ids=lambda i: i.id)
def test_answer_key_never_leaks_into_the_text(incident: Incident):
    # The model sees only `text`. A line like 'expected severity' there would
    # make every measurement meaningless.
    lowered = incident.text.lower()
    assert "expected" not in lowered
    assert incident.trap.lower() not in lowered


def test_severity_mix_is_not_all_one_value():
    # An answer key of one value would let "always say sev2" score full marks.
    assert len({b.expected.severity for b in BANK}) >= 3


def test_an_oversized_incident_is_rejected():
    with pytest.raises(ValueError):
        Incident.model_validate(
            {
                "id": "x", "title": "t", "trap": "t",
                "expected": {"severity": "sev3", "components": ["a"]},
                "text": "x" * (MAX_TEXT_CHARS + 1),
            }
        )
