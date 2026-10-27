"""Nearest-chunk search over the ingested incidents (Day 3, step 4).

What: embeds a question and returns the closest chunks by cosine distance.

Run it:  uv run python -m app.search "auth latency spike after certificate rotation"

Mechanics worth knowing:
- The query must be embedded by the same model and revision as the documents.
  A different model gives vectors in a different space, and distances between
  the two spaces mean nothing.
- bge models are trained to expect an instruction in front of a short search
  query. Documents get no instruction. Skipping it still returns results, only
  worse ones, with no error to warn you. That is why the prefix is tested.
- `<=>` is pgvector's cosine distance, 1 minus cosine similarity. Smaller is
  closer. Results are ordered by it, ascending.
- At 800 rows the planner usually ignores the HNSW index and scans every row.
  That is the right call at this size and it is exact. The index starts to pay
  at tens of thousands of rows. At that size it is approximate: it can miss a
  true neighbour, and `hnsw.ef_search` trades recall against speed.
- Filters (service, severity) are not applied here. With HNSW a WHERE clause
  runs after the index scan, so a selective filter can return fewer than `k`
  rows. Day 4 deals with that.

Separation: `search` never sees `twin_of`. Only the display code reads the
answer key, to mark hits for the self-check. Retrieval that could read its own
answer key would make the Day 4 recall number meaningless.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.ingest import Embedder, database_url, make_embedder
from app.paths import DATA

if TYPE_CHECKING:
    from psycopg import Connection

# The instruction bge-small-en-v1.5 was trained with for search queries.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_QUERY = "auth latency spike after certificate rotation"

_SEARCH = """
SELECT id, incident_id, section, service, severity, body, embedding <=> %s AS distance
FROM chunks
ORDER BY embedding <=> %s
LIMIT %s
"""


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. `distance` is cosine distance: smaller is closer."""

    id: str
    incident_id: str
    section: str
    service: str
    severity: str
    body: str
    distance: float


def embed_query(embed: Embedder, text: str) -> list[float]:
    """Embed one search query, with the instruction bge expects in front of it."""
    if not text.strip():
        raise ValueError("empty query")
    (vector,) = embed([QUERY_PREFIX + text.strip()])
    return vector


def search(conn: Connection, embed: Embedder, query: str, k: int = 5) -> list[Hit]:
    """Return the `k` chunks nearest to `query`, closest first.

    The connection needs `register_vector` applied so a `Vector` can be sent.
    The query vector is passed twice because the distance is both selected, so
    it can be shown, and used to order. Postgres computes it once per row.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    vector = Vector(embed_query(embed, query))
    rows = conn.execute(_SEARCH, (vector, vector, k)).fetchall()
    return [Hit(*row) for row in rows]


def load_twin_pairs(path: Path) -> dict[str, str]:
    """Map each incident in a seeded pair to its partner, in both directions.

    Display only. This reads the Day 3 answer key from the source file, and is
    never called by `search`.
    """
    pairs: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("twin_of"):
                pairs[row["id"]] = row["twin_of"]
                pairs[row["twin_of"]] = row["id"]
    return pairs


def format_hits(hits: Sequence[Hit], pairs: Mapping[str, str]) -> str:
    """Render hits as a table and a one-line verdict on the seeded pairs.

    A hit "belongs to a pair" when its incident has a twin. The verdict also
    counts pairs where BOTH incidents appear, which is the stronger result.
    """
    present = {h.incident_id for h in hits}
    lines = []
    for rank, hit in enumerate(hits, 1):
        partner = pairs.get(hit.incident_id)
        note = ""
        if partner:
            note = f"pair {partner}" + (" (both here)" if partner in present else "")
        body = hit.body if len(hit.body) <= 60 else hit.body[:57] + "..."
        lines.append(
            f"{rank}. {hit.distance:.3f}  {hit.id:<22} {hit.service:<18} "
            f"{hit.section:<11} {note:<26} {body}"
        )
    in_pair = sum(1 for h in hits if h.incident_id in pairs)
    both = sum(1 for i in present if pairs.get(i) in present) // 2
    lines.append(
        f"hits that belong to a seeded pair: {in_pair} of {len(hits)}; "
        f"pairs with both members in the results: {both}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Search the local database and print the top 5 for the query in `argv`."""
    words = sys.argv[1:] if argv is None else argv
    query = " ".join(words) or DEFAULT_QUERY
    pairs = load_twin_pairs(DATA / "incidents.jsonl")
    print(f"query: {query}\n")
    with psycopg.connect(database_url(), autocommit=True) as conn:
        register_vector(conn)
        print(format_hits(search(conn, make_embedder(), query), pairs))


if __name__ == "__main__":
    main()
