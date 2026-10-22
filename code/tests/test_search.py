"""Fast tests for search. No model, no database, no network.

The SQL itself is covered by tests/integration/test_search_db.py.
"""

from pathlib import Path

import pytest

from app.corpus import build_corpus, write_jsonl
from app.ingest import DIMENSIONS, chunk_incident
from app.search import QUERY_PREFIX, Hit, embed_query, format_hits, load_twin_pairs, search


def hit(incident: str, section: str = "summary", distance: float = 0.1) -> Hit:
    """A hit with filler text, for the formatting tests."""
    return Hit(f"{incident}:{section}", incident, section, "auth", "sev1", "Some body text.", distance)


def test_queries_get_the_instruction_and_documents_do_not() -> None:
    """The model sees the prefix on queries only. Mixing this up lowers recall silently."""
    seen: list[str] = []

    def spy(texts):
        seen.extend(texts)
        return [[0.0] * DIMENSIONS for _ in texts]

    embed_query(spy, "  certificate rotation  ")
    assert seen == [QUERY_PREFIX + "certificate rotation"]
    row = {
        "id": "INC-1", "service": "auth", "severity": "sev1", "date": "2026-01-01",
        "summary": "s", "timeline": "t", "root_cause": "r", "remediation": "m",
    }
    assert not any(c.embed_text.startswith(QUERY_PREFIX) for c in chunk_incident(row))


def test_empty_query_and_bad_k_are_rejected_before_any_work() -> None:
    """Fail early, with no connection or model touched."""
    with pytest.raises(ValueError, match="empty"):
        embed_query(lambda t: [[0.0]], "   ")
    with pytest.raises(ValueError, match="at least 1"):
        search(None, lambda t: [[0.0]], "q", k=0)


def test_twin_pairs_are_read_in_both_directions(tmp_path: Path) -> None:
    """20 pairs give 40 entries, and each points back to its partner."""
    path = tmp_path / "incidents.jsonl"
    write_jsonl(build_corpus(), path)
    pairs = load_twin_pairs(path)
    assert len(pairs) == 40
    assert all(pairs[pairs[i]] == i for i in pairs)


def test_format_marks_pairs_and_counts_both_members() -> None:
    """A pair with one member present is a weak hit. Both present is the strong one."""
    pairs = {"INC-1": "INC-2", "INC-2": "INC-1", "INC-7": "INC-8", "INC-8": "INC-7"}
    text = format_hits([hit("INC-1"), hit("INC-2"), hit("INC-7"), hit("INC-9")], pairs)
    assert "hits that belong to a seeded pair: 3 of 4" in text
    assert "pairs with both members in the results: 1" in text
    assert "pair INC-2 (both here)" in text
    assert "pair INC-8" in text and "pair INC-8 (both here)" not in text


def test_format_with_no_pairs_says_zero() -> None:
    """No pair among the hits is reported plainly, not hidden."""
    text = format_hits([hit("INC-5")], {"INC-1": "INC-2", "INC-2": "INC-1"})
    assert "belong to a seeded pair: 0 of 1" in text
