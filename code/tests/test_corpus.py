"""Tests for the synthetic corpus generator. No model, no database, no network.

They pin the properties later days rely on: the size, the fields, the twin
answer key, and determinism. A twin must carry the same facts as its original
but different words, or the Day 4 recall number would measure nothing.
"""

import json
import re
from pathlib import Path

import pytest

from app.corpus import (
    SERVICES,
    Incident,
    build_corpus,
    clock,
    write_jsonl,
)

FIELDS = {
    "id", "service", "severity", "date", "summary", "timeline",
    "root_cause", "remediation", "synthetic", "twin_of",
}


@pytest.fixture(scope="module")
def corpus() -> list[Incident]:
    """The default corpus, built once. It is pure, so sharing it is safe."""
    return build_corpus()


def _numbers(incident: Incident) -> set[str]:
    """Every number in the incident text. Facts are numbers, so this compares facts."""
    text = " ".join([incident.summary, incident.timeline, incident.root_cause])
    return set(re.findall(r"\d+(?:\.\d+)*", text))


def test_size_and_ids(corpus: list[Incident]) -> None:
    """200 rows, unique ids, ids in date order."""
    assert len(corpus) == 200
    assert len({i.id for i in corpus}) == 200
    assert [i.date for i in corpus] == sorted(i.date for i in corpus)


def test_every_row_has_the_fields_and_is_labelled_synthetic(corpus: list[Incident]) -> None:
    """The file must say it is synthetic, on every row."""
    for incident in corpus:
        assert set(incident.__dict__) == FIELDS
        assert incident.synthetic is True
        assert incident.summary and incident.timeline
        assert incident.root_cause and incident.remediation


def test_all_eight_services_appear(corpus: list[Incident]) -> None:
    """No service is missing from the corpus."""
    assert len(SERVICES) == 8
    assert {i.service for i in corpus} == set(SERVICES)


def test_no_template_placeholder_leaks(corpus: list[Incident]) -> None:
    """A literal `{` in the text means a placeholder was never filled."""
    for incident in corpus:
        text = incident.summary + incident.timeline + incident.root_cause + incident.remediation
        assert "{" not in text and "}" not in text


def test_clock_wraps_past_midnight() -> None:
    """Timeline times wrap at 24 hours instead of printing 25:10."""
    assert clock(0) == "00:00"
    assert clock(23 * 60 + 70) == "00:10"


def test_twenty_twin_pairs(corpus: list[Incident]) -> None:
    """Exactly 20 twins. Each points at an original that is not itself a twin."""
    by_id = {i.id: i for i in corpus}
    twins = [i for i in corpus if i.twin_of is not None]
    assert len(twins) == 20
    assert len({t.twin_of for t in twins}) == 20
    for twin in twins:
        assert by_id[twin.twin_of].twin_of is None


def test_a_twin_is_the_same_incident_in_different_words(corpus: list[Incident]) -> None:
    """Same service, severity and numbers. Different text. Dated later."""
    by_id = {i.id: i for i in corpus}
    for twin in (i for i in corpus if i.twin_of is not None):
        original = by_id[twin.twin_of]
        assert twin.service == original.service
        assert twin.severity == original.severity
        assert _numbers(twin) == _numbers(original)
        assert twin.summary != original.summary
        assert twin.root_cause != original.root_cause
        assert twin.date > original.date


def test_same_seed_same_corpus_different_seed_different_corpus() -> None:
    """Determinism is the point of generating the file instead of asking a model."""
    assert build_corpus(seed=1) == build_corpus(seed=1)
    assert build_corpus(seed=1) != build_corpus(seed=2)


def test_impossible_pair_count_is_rejected() -> None:
    """More twins than originals cannot be built."""
    with pytest.raises(ValueError):
        build_corpus(total=10, pairs=9)


def test_jsonl_round_trip(tmp_path: Path, corpus: list[Incident]) -> None:
    """One JSON object per line, readable back, with the answer key intact."""
    path = tmp_path / "nested" / "incidents.jsonl"
    assert write_jsonl(corpus, path) == 200
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 200
    assert set(rows[0]) == FIELDS
    assert sum(r["twin_of"] is not None for r in rows) == 20
