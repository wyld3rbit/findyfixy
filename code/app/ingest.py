"""Ingestion job: incident postmortems in, embedded chunks out (Day 3).

What: reads `data/incidents.jsonl`, splits each incident into its four
sections, embeds each section, and stores it in Postgres with pgvector. It is
safe to re-run: unchanged chunks are not touched.

Pipeline: source -> chunk -> hash -> diff against the table -> embed only what
changed -> one transaction (upsert changed, delete retracted).

Design decisions and what changes at scale:
- Chunk by section. A postmortem has structure. One section is one topic, so one
  vector means one thing. A fixed token window would cut across topics.
- Idempotence by content hash. The hash covers the embedding model, the exact
  text that is embedded, and the stored metadata. Change any of them and the
  chunk is rewritten. Change none and it is skipped. Switching the model
  therefore re-embeds everything, which is the correct outcome: vectors from two
  models are not comparable, and mixing them makes distances meaningless.
- Deletes are handled. A chunk whose id is no longer in the source is removed.
  Without this the index keeps serving incidents that were retracted.
- One transaction. A crash half way leaves the previous state, never a mix.
  At millions of chunks you would write in bounded batches and make each batch
  idempotent, because one long transaction holds locks and bloats the WAL.
- The embedder is injected. Tests pass a fake, so the logic is checked without a
  model. The real one is built only in `main`, never at import time.
- `twin_of` in the source file is the Day 4 answer key. This module never reads
  it. If retrieval could see its own answer key, the recall number would mean
  nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import psycopg
from pgvector import Vector

if TYPE_CHECKING:
    from psycopg import Connection

# The model is pinned by revision, not just by name. A model repository can be
# edited after you read it. A commit hash cannot. Weights are `safetensors`, a
# format that cannot run code on load, unlike pickle-based `.bin` files.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIMENSIONS = 384

Section = Literal["summary", "timeline", "root_cause", "remediation"]
SECTIONS: tuple[Section, ...] = get_args(Section)

# Takes a batch of texts, returns one vector per text.
Embedder = Callable[[Sequence[str]], list[list[float]]]

# The local compose database, in psycopg's keyword form. It matches the
# compose file: localhost only, the dev password, never a real credential.
# DATABASE_URL overrides it, as a URL or in this form.
DEFAULT_DSN = "host=127.0.0.1 port=5432 dbname=triage user=postgres password=dev"

# `embedding vector(384)` fixes the dimension in the column type, so a vector
# from the wrong model is rejected by the database instead of stored silently.
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,
    incident_id     text NOT NULL,
    section         text NOT NULL,
    service         text NOT NULL,
    severity        text NOT NULL,
    incident_date   date NOT NULL,
    body            text NOT NULL,
    hash            text NOT NULL,
    embedding_model text NOT NULL,
    embedding       vector({DIMENSIONS}) NOT NULL
)
"""
# HNSW with pgvector's defaults (m = 16, ef_construction = 64). Cosine
# operator class, because the vectors are normalised and queries use cosine.
_INDEX = (
    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
    "ON chunks USING hnsw (embedding vector_cosine_ops)"
)

_UPSERT = """
INSERT INTO chunks (id, incident_id, section, service, severity, incident_date,
                    body, hash, embedding_model, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    incident_id = EXCLUDED.incident_id,
    section = EXCLUDED.section,
    service = EXCLUDED.service,
    severity = EXCLUDED.severity,
    incident_date = EXCLUDED.incident_date,
    body = EXCLUDED.body,
    hash = EXCLUDED.hash,
    embedding_model = EXCLUDED.embedding_model,
    embedding = EXCLUDED.embedding
"""


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: a single section of a single incident.

    `body` is what a reader sees. `embed_text` is what the model sees: the body
    with a short header, because a section like "Disk at 100 percent" says
    little without the service and the section name.
    """

    id: str
    incident_id: str
    section: Section
    service: str
    severity: str
    incident_date: str
    body: str
    embed_text: str
    hash: str


@dataclass(frozen=True)
class Plan:
    """What a run has to do, decided before any model or write is used."""

    to_write: tuple[Chunk, ...]
    to_delete: tuple[str, ...]
    unchanged: int


@dataclass(frozen=True)
class Report:
    """What a run did. `written` and `deleted` are zero on an unchanged re-run."""

    total: int
    written: int
    deleted: int
    unchanged: int

    def __str__(self) -> str:
        return (
            f"chunks {self.total}: written {self.written}, "
            f"deleted {self.deleted}, unchanged {self.unchanged}"
        )


def content_hash(model: str, embed_text: str, incident_date: str) -> str:
    """Hash everything that decides what is stored for a chunk.

    JSON encoding keeps the fields unambiguous: with plain concatenation,
    ("ab", "c") and ("a", "bc") would hash the same.
    """
    payload = json.dumps([model, embed_text, incident_date])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_incident(row: Mapping[str, object], model: str = MODEL_NAME) -> list[Chunk]:
    """Split one incident into one chunk per section.

    Reads named fields only. A missing field raises KeyError on purpose: a
    malformed record should stop the job, not produce an empty chunk that
    embeds to noise. Metadata (service, severity, date) is stored on every chunk
    so retrieval can filter on it before it compares vectors.
    """
    incident_id = str(row["id"])
    service = str(row["service"])
    severity = str(row["severity"])
    incident_date = str(row["date"])
    chunks = []
    for section in SECTIONS:
        body = str(row[section])
        embed_text = f"{service} | {severity} | {section.replace('_', ' ')}: {body}"
        chunks.append(
            Chunk(
                id=f"{incident_id}:{section}",
                incident_id=incident_id,
                section=section,
                service=service,
                severity=severity,
                incident_date=incident_date,
                body=body,
                embed_text=embed_text,
                hash=content_hash(model, embed_text, incident_date),
            )
        )
    return chunks


def read_chunks(path: Path, model: str = MODEL_NAME) -> list[Chunk]:
    """Read a JSON Lines file of incidents and chunk every row.

    Blank lines are skipped. Duplicate chunk ids raise: two rows with the same
    incident id would silently overwrite each other in the table.
    """
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.extend(chunk_incident(json.loads(line), model))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate chunk ids in {path.name}")
    return chunks


def plan_changes(chunks: Iterable[Chunk], existing: Mapping[str, str]) -> Plan:
    """Compare the source with the table, by hash.

    `existing` maps chunk id to stored hash. A chunk is written when it is new
    or its hash differs. An id in the table but not in the source is deleted.
    """
    chunks = list(chunks)
    to_write = tuple(c for c in chunks if existing.get(c.id) != c.hash)
    seen = {c.id for c in chunks}
    to_delete = tuple(sorted(i for i in existing if i not in seen))
    return Plan(to_write, to_delete, unchanged=len(chunks) - len(to_write))


def ensure_schema(conn: Connection) -> None:
    """Create the extension, table and HNSW index if they are missing.

    Safe to run every time. The connection is in autocommit mode, so each
    statement stands alone and the table exists before `Vector` values are sent.
    """
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(_SCHEMA)
    conn.execute(_INDEX)


def existing_hashes(conn: Connection) -> dict[str, str]:
    """Chunk id to stored hash, for the diff. One row per chunk, no vectors."""
    return dict(conn.execute("SELECT id, hash FROM chunks").fetchall())


def embed_in_batches(
    embed: Embedder, texts: Sequence[str], batch_size: int
) -> list[list[float]]:
    """Embed in bounded batches and check every vector's size.

    Batching bounds memory: a batch is padded to its longest text, so one giant
    batch wastes compute on padding. The size check catches a wrong model before
    the database does, with a clearer message.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embed(texts[start : start + batch_size]))
    if len(vectors) != len(texts):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
    for vector in vectors:
        if len(vector) != DIMENSIONS:
            raise ValueError(f"expected {DIMENSIONS} dimensions, got {len(vector)}")
    return vectors


def apply_plan(
    conn: Connection,
    plan: Plan,
    vectors: Sequence[Sequence[float]],
    model: str = MODEL_NAME,
) -> None:
    """Write the plan in one transaction: upsert changed chunks, delete retracted."""
    with conn.transaction():
        if plan.to_write:
            rows = [
                (
                    c.id, c.incident_id, c.section, c.service, c.severity,
                    c.incident_date, c.body, c.hash, model, Vector(v),
                )
                for c, v in zip(plan.to_write, vectors, strict=True)
            ]
            conn.cursor().executemany(_UPSERT, rows)
        if plan.to_delete:
            conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (list(plan.to_delete),))


def ingest(
    conn: Connection,
    chunks: Sequence[Chunk],
    embed: Embedder,
    *,
    model: str = MODEL_NAME,
    batch_size: int = 64,
) -> Report:
    """Bring the table in line with `chunks`, doing only the work that is needed.

    Embedding is the slow step, so it runs after the diff and only for chunks
    that are new or changed. On an unchanged re-run the model is never called.
    """
    ensure_schema(conn)
    plan = plan_changes(chunks, existing_hashes(conn))
    vectors = embed_in_batches(embed, [c.embed_text for c in plan.to_write], batch_size)
    apply_plan(conn, plan, vectors, model)
    return Report(
        total=len(chunks),
        written=len(plan.to_write),
        deleted=len(plan.to_delete),
        unchanged=plan.unchanged,
    )


def make_embedder() -> Embedder:
    """Load the real model on the CPU and return an embedding function.

    Imported inside the function so importing this module stays cheap and the
    tests never load torch. Vectors are normalised to unit length: cosine
    similarity then equals the dot product, which is what the index is built for.
    The first call downloads the pinned revision (about 130 MB).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
    if model.get_embedding_dimension() != DIMENSIONS:
        raise ValueError("model dimension does not match the table column")

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    return embed


def database_url() -> str:
    """Connection string from DATABASE_URL, else the local compose database.

    psycopg accepts both a URL and the keyword form, so either works in the
    environment variable.
    """
    return os.environ.get("DATABASE_URL", DEFAULT_DSN)


def main() -> None:
    """Ingest `data/incidents.jsonl` into the local database and print a report.

    The data path is resolved from this file, not the working directory.
    """
    root = Path(__file__).resolve().parents[1]
    chunks = read_chunks(root / "data" / "incidents.jsonl")
    with psycopg.connect(database_url(), autocommit=True) as conn:
        ensure_schema(conn)
        # Registering after the extension exists lets psycopg send `Vector` values.
        from pgvector.psycopg import register_vector

        register_vector(conn)
        embed = make_embedder() if _needs_model(conn, chunks) else _never_called
        print(ingest(conn, chunks, embed))


def _needs_model(conn: Connection, chunks: Sequence[Chunk]) -> bool:
    """True when the plan has work. Skips the slow model load on a no-op re-run."""
    return bool(plan_changes(chunks, existing_hashes(conn)).to_write)


def _never_called(texts: Sequence[str]) -> list[list[float]]:
    """Stand-in embedder for a run with nothing to embed."""
    raise AssertionError("embedder called although the plan has nothing to write")


if __name__ == "__main__":
    main()
