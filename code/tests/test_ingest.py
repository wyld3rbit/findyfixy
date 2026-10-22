"""Fast tests for the ingestion logic. No model, no database, no network.

They cover the decisions that keep the index correct: how an incident is cut
into chunks, what the hash covers, and what the diff writes, skips and deletes.
The database part is covered by tests/integration/test_ingest_db.py.
"""

import json
from pathlib import Path

import pytest

from app.corpus import build_corpus, write_jsonl
from app.ingest import (
    DIMENSIONS,
    SECTIONS,
    chunk_incident,
    content_hash,
    embed_in_batches,
    plan_changes,
    read_chunks,
)

ROW = {
    "id": "INC-0001",
    "service": "auth",
    "severity": "sev1",
    "date": "2026-01-02",
    "summary": "Auth was slow.",
    "timeline": "09:00 Alert.",
    "root_cause": "Bad certificate.",
    "remediation": "Test before cutover.",
    "synthetic": True,
    "twin_of": "INC-0099",
}


def fake_embed(texts):
    """A fake model: right size, content-dependent, no weights."""
    return [[float(len(t) % 11)] * DIMENSIONS for t in texts]


def test_one_chunk_per_section_with_metadata() -> None:
    """Four chunks per incident, each carrying the filter fields."""
    chunks = chunk_incident(ROW)
    assert [c.section for c in chunks] == list(SECTIONS)
    assert [c.id for c in chunks][0] == "INC-0001:summary"
    for chunk in chunks:
        assert (chunk.service, chunk.severity, chunk.incident_date) == ("auth", "sev1", "2026-01-02")
        assert chunk.body in chunk.embed_text
        assert chunk.embed_text.startswith("auth | sev1 | ")


def test_the_answer_key_never_reaches_a_chunk() -> None:
    """`twin_of` must not leak into anything that is stored or embedded."""
    for chunk in chunk_incident(ROW):
        assert "INC-0099" not in repr(chunk)


def test_missing_field_stops_the_job() -> None:
    """A malformed record raises. It must not become an empty chunk."""
    broken = {k: v for k, v in ROW.items() if k != "root_cause"}
    with pytest.raises(KeyError):
        chunk_incident(broken)


def test_hash_covers_model_text_and_date() -> None:
    """Changing any stored input changes the hash. Changing none keeps it."""
    base = content_hash("m", "text", "2026-01-01")
    assert content_hash("m", "text", "2026-01-01") == base
    assert content_hash("other", "text", "2026-01-01") != base
    assert content_hash("m", "text2", "2026-01-01") != base
    assert content_hash("m", "text", "2026-01-02") != base


def test_hash_fields_are_unambiguous() -> None:
    """Moving a character across a field boundary must change the hash."""
    assert content_hash("ab", "c", "d") != content_hash("a", "bc", "d")


def test_switching_model_rewrites_every_chunk() -> None:
    """Vectors from two models are not comparable, so a new model means a full redo."""
    old = chunk_incident(ROW, model="model-a")
    existing = {c.id: c.hash for c in old}
    plan = plan_changes(chunk_incident(ROW, model="model-b"), existing)
    assert len(plan.to_write) == 4
    assert plan.unchanged == 0


def test_unchanged_source_writes_nothing() -> None:
    """The re-run guarantee: same source, same table, zero writes."""
    chunks = chunk_incident(ROW)
    plan = plan_changes(chunks, {c.id: c.hash for c in chunks})
    assert plan.to_write == () and plan.to_delete == ()
    assert plan.unchanged == 4


def test_changed_text_rewrites_only_that_chunk() -> None:
    """One edited section means one write."""
    chunks = chunk_incident(ROW)
    existing = {c.id: c.hash for c in chunks}
    edited = chunk_incident({**ROW, "root_cause": "A different cause."})
    plan = plan_changes(edited, existing)
    assert [c.section for c in plan.to_write] == ["root_cause"]
    assert plan.unchanged == 3


def test_retracted_incident_is_deleted() -> None:
    """An id in the table but gone from the source is removed from the index."""
    chunks = chunk_incident(ROW)
    existing = {c.id: c.hash for c in chunks} | {"INC-0042:summary": "x"}
    plan = plan_changes(chunks, existing)
    assert plan.to_delete == ("INC-0042:summary",)


def test_read_chunks_gives_800_from_the_corpus(tmp_path: Path) -> None:
    """200 incidents become 800 chunks with unique ids."""
    path = tmp_path / "incidents.jsonl"
    write_jsonl(build_corpus(), path)
    chunks = read_chunks(path)
    assert len(chunks) == 800
    assert len({c.id for c in chunks}) == 800


def test_duplicate_incident_ids_are_rejected(tmp_path: Path) -> None:
    """Two rows with one id would overwrite each other silently."""
    path = tmp_path / "dup.jsonl"
    path.write_text((json.dumps(ROW) + "\n") * 2)
    with pytest.raises(ValueError):
        read_chunks(path)


def test_batches_cover_every_text_in_order() -> None:
    """Batching must not drop, repeat or reorder texts."""
    seen: list[list[str]] = []

    def spy(texts):
        seen.append(list(texts))
        return fake_embed(texts)

    texts = [f"t{i}" for i in range(7)]
    vectors = embed_in_batches(spy, texts, batch_size=3)
    assert [len(b) for b in seen] == [3, 3, 1]
    assert [t for b in seen for t in b] == texts
    assert len(vectors) == 7


def test_wrong_vector_size_is_caught() -> None:
    """A model with the wrong dimension fails with a clear message."""
    with pytest.raises(ValueError, match="dimensions"):
        embed_in_batches(lambda texts: [[0.0] * 3 for _ in texts], ["a"], batch_size=8)


def test_wrong_vector_count_is_caught() -> None:
    """An embedder that loses a text must not shift every later vector."""
    with pytest.raises(ValueError, match="vectors"):
        embed_in_batches(lambda texts: fake_embed(texts)[:-1], ["a", "b"], batch_size=8)
