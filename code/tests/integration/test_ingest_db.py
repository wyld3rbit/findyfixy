"""Ingestion against the real Postgres from docker-compose.

The fast tests prove the diff logic. This proves the SQL: the table and HNSW
index are created, the upsert and delete work, and a re-run writes nothing.

It runs in a throwaway schema, so it cannot touch the real `chunks` table, and
it is dropped afterwards. It skips, not fails, when the database is down: a red
result should mean the code is wrong, not that `docker compose up` was not run.
The `conn` fixture, with its throwaway schema, is in conftest.py.
Run with: uv run pytest -m integration tests/integration/test_ingest_db.py
"""

import psycopg
import pytest

from app.ingest import DIMENSIONS, chunk_incident, ingest

pytestmark = pytest.mark.integration

BASE = {
    "service": "auth", "severity": "sev1", "date": "2026-01-02",
    "summary": "Auth was slow.", "timeline": "09:00 Alert.",
    "root_cause": "Bad certificate.", "remediation": "Test before cutover.",
}


class SpyEmbedder:
    """A fake model that counts how many texts it was asked to embed."""

    def __init__(self) -> None:
        self.texts = 0

    def __call__(self, texts):
        self.texts += len(texts)
        return [[0.5] * DIMENSIONS for _ in texts]


def count(conn: psycopg.Connection) -> int:
    """Rows in the test schema's chunks table."""
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def test_first_run_writes_second_run_writes_nothing(conn: psycopg.Connection) -> None:
    """The self-check promise: the re-run reports zero writes and embeds nothing."""
    chunks = chunk_incident({"id": "INC-1", **BASE}) + chunk_incident({"id": "INC-2", **BASE})
    embed = SpyEmbedder()

    first = ingest(conn, chunks, embed)
    assert (first.written, first.deleted, first.unchanged) == (8, 0, 0)
    assert count(conn) == 8
    assert embed.texts == 8

    second = ingest(conn, chunks, embed)
    assert (second.written, second.deleted, second.unchanged) == (0, 0, 8)
    assert embed.texts == 8, "the model must not be called when nothing changed"


def test_edit_rewrites_one_chunk_and_retraction_deletes(conn: psycopg.Connection) -> None:
    """An edited section is one upsert. A removed incident disappears from the table."""
    both = chunk_incident({"id": "INC-1", **BASE}) + chunk_incident({"id": "INC-2", **BASE})
    ingest(conn, both, SpyEmbedder())

    edited = chunk_incident({"id": "INC-1", **BASE, "root_cause": "Something else."})
    report = ingest(conn, edited, SpyEmbedder())
    assert (report.written, report.deleted, report.unchanged) == (1, 4, 3)
    assert count(conn) == 4
    body = conn.execute("SELECT body FROM chunks WHERE id = 'INC-1:root_cause'").fetchone()[0]
    assert body == "Something else."


def test_stored_rows_carry_model_and_dimension(conn: psycopg.Connection) -> None:
    """Every row records its model, and the vector has the column's size."""
    ingest(conn, chunk_incident({"id": "INC-1", **BASE}), SpyEmbedder())
    model, dims = conn.execute(
        "SELECT embedding_model, vector_dims(embedding) FROM chunks LIMIT 1"
    ).fetchone()
    assert model == "BAAI/bge-small-en-v1.5"
    assert dims == DIMENSIONS


def test_hnsw_index_exists(conn: psycopg.Connection) -> None:
    """The index is part of the schema, not an afterthought."""
    ingest(conn, chunk_incident({"id": "INC-1", **BASE}), SpyEmbedder())
    definition = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'"
    ).fetchone()[0]
    assert "hnsw" in definition and "vector_cosine_ops" in definition
