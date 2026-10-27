"""Day 4: three retrievers over the incident chunks, behind one interface.

What it does
    `Retriever.vector`, `.hybrid` and `.hybrid_rerank` each take a query, a
    count `k` and an optional `service`, and return the top `k` INCIDENTS,
    best first. One interface lets `app/eval_retrieval.py` score all three
    the same way, and lets the graph swap one for another.

Why three, in the order they fix each other's failures
    vector: embeds the query and finds the nearest chunks (pgvector, Day 3).
        Strong on meaning ("auth is slow" finds "login latency"). Weak on
        exact strings: a dense model smooths `scheduler-3de` into its
        neighbours, and it ranked "14 times normal" next to "41 times normal".
    hybrid: adds BM25, a keyword score. A term counts more when it is rare in
        the corpus (inverse document frequency) and its count saturates
        (`k1`), with long documents normalised down (`b`). A hostname is the
        rarest term there is, so BM25 finds it exactly and misses every
        paraphrase. The two lists are then merged.
    hybrid_rerank: takes the best 30 chunks from the merge and re-scores each
        with a cross-encoder. A bi-encoder embeds query and chunk separately
        and never sees them together. A cross-encoder reads both as one input
        and attends across them, which judges relevance far better, at one
        forward pass per candidate per query. Hence two stages: cheap recall
        first, expensive precision on a short list.

Why reciprocal-rank fusion (RRF)
    A BM25 score of 12 and a cosine distance of 0.17 are on different scales
    and shapes. Normalising and weighting them is brittle and needs re-tuning
    when the corpus changes. RRF ignores scores: score(d) = sum over lists of
    1 / (60 + rank). A chunk high in both lists wins. The 60 flattens the
    curve so rank 1 does not swamp rank 5. No training, few failure modes.

Why results are incidents, not chunks
    The labels name incidents, and one incident has four chunks. Without
    collapsing, one incident could fill all five slots and recall@5 would
    measure nothing. An incident ranks by its best chunk.

Seams (dependency injection, nothing at import)
    `VectorSearch` and `Rerank` are plain functions passed in. Tests pass
    fakes, so BM25, fusion, collapsing and filters are tested with no
    database and no model. `postgres_vector_search` and `make_reranker` are
    the real ones.

What changes at scale
    - BM25 here is built in memory from every chunk when the retriever is
      made. Fine for 800 chunks. At millions it moves into the database
      (Postgres full-text search) or a search engine, and it is updated, not
      rebuilt.
    - A service filter is applied with the vector query's WHERE clause. With
      an HNSW index the filter runs after the index scan, so a selective
      filter can return fewer than `k` rows. At 800 rows Postgres scans every
      row, so it is exact. At scale, raise `hnsw.ef_search`, use iterative
      scans, or partition by service.
    - The reranker is usually the biggest slice of retrieval latency. The
      pool size (30) is the lever: halve it and the cost halves.

Separation
    Nothing here reads `twin_of` or the eval labels. A retriever that could
    read its own answer key would make recall meaningless.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args

from pgvector import Vector
from rank_bm25 import BM25Okapi

from app.ingest import Embedder
from app.search import embed_query
from app.triage import Prior, Severity

if TYPE_CHECKING:
    from psycopg import Connection

# Four sections per incident (summary, timeline, root cause, remediation).
# Asking for k * SECTIONS chunks guarantees k distinct incidents, even when one
# incident takes every chunk.
SECTIONS = 4
BM25_POOL = 50
VECTOR_POOL = 50
RERANK_POOL = 30
RRF_K = 60

RERANKER_NAME = "BAAI/bge-reranker-base"
# Pinned like the embedding model: a silent weight swap would change every
# number without a code change.
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"

# (query, limit, service) -> [(chunk id, cosine distance)], nearest first.
VectorSearch = Callable[[str, int, str | None], list[tuple[str, float]]]
# (query, texts) -> one relevance score per text, higher is better.
Rerank = Callable[[str, Sequence[str]], list[float]]

# A hostname-like token with parts: `scheduler-3de`. Matched whole, and its
# parts are added too, so `scheduler-3de` finds the exact host (rare, high
# weight) and `dead-letter` still matches "dead letter".
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class Passage:
    """One chunk as the retrievers see it. No embedding, no label."""

    chunk_id: str
    incident_id: str
    section: str
    service: str
    severity: Severity
    body: str

    def __post_init__(self) -> None:
        """Reject an unknown severity from the database. The column is plain
        text, so a bad row must stop loading, not reach a prompt."""
        if self.severity not in get_args(Severity):
            raise ValueError(f"unknown severity {self.severity!r} in {self.chunk_id}")

    @property
    def text(self) -> str:
        """What BM25 and the reranker read: the body with its service and section.

        The embedding text carries the same header (Day 3), because a section
        such as "Disk at 100 percent" says little without it.
        """
        return f"{self.service} {self.section}: {self.body}"


@dataclass(frozen=True)
class Found:
    """One retrieved incident. `score` is higher-is-better but its scale
    depends on the retriever (1 - distance, an RRF sum, a reranker logit), so
    compare ranks, never scores across retrievers."""

    incident_id: str
    chunk_id: str
    service: str
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase terms for BM25: whole hyphenated tokens, then their parts."""
    tokens: list[str] = []
    for word in _TOKEN.findall(text.lower()):
        tokens.append(word)
        if "-" in word:
            tokens.extend(word.split("-"))
    return tokens


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids by reciprocal rank. Best first.

    Ties break by first appearance, so the output is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    order = {item: i for i, item in enumerate(scores)}
    return sorted(scores.items(), key=lambda pair: (-pair[1], order[pair[0]]))


class Retriever:
    """Three retrieval strategies over one set of passages.

    Building one indexes every passage for BM25 (a pass over 800 short
    texts). It is made once and reused, never at import.
    """

    def __init__(
        self,
        passages: Sequence[Passage],
        vector_search: VectorSearch,
        rerank: Rerank | None = None,
    ) -> None:
        if not passages:
            raise ValueError("no passages to retrieve from")
        self._passages = list(passages)
        self._by_chunk = {p.chunk_id: p for p in self._passages}
        self._by_incident: dict[str, dict[str, Passage]] = {}
        for p in self._passages:
            self._by_incident.setdefault(p.incident_id, {})[p.section] = p
        self._bm25 = BM25Okapi([tokenize(p.text) for p in self._passages])
        self._vector_search = vector_search
        self._rerank = rerank

    # -- the three retrievers -------------------------------------------------

    def vector(self, query: str, k: int = 5, service: str | None = None) -> list[Found]:
        """Nearest chunks by meaning, collapsed to incidents."""
        _check(query, k)
        hits = self._vector_search(query, k * SECTIONS, service)
        return self._collapse([(cid, 1.0 - dist) for cid, dist in hits], k)

    def hybrid(self, query: str, k: int = 5, service: str | None = None) -> list[Found]:
        """BM25 top 50 and vector top 50, fused by RRF, collapsed to incidents."""
        _check(query, k)
        return self._collapse(self._fused(query, service), k)

    def hybrid_rerank(
        self, query: str, k: int = 5, service: str | None = None
    ) -> list[Found]:
        """Hybrid's best 30 chunks, re-scored by the cross-encoder."""
        _check(query, k)
        if self._rerank is None:
            raise ValueError("this retriever was built without a reranker")
        pool = self._fused(query, service)[:RERANK_POOL]
        if not pool:
            return []
        scores = self._rerank(query, [self._by_chunk[cid].text for cid, _ in pool])
        if len(scores) != len(pool):
            raise ValueError("the reranker returned a different number of scores")
        # sorted() is stable: equal scores keep the fused order.
        ranked = sorted(zip(pool, scores), key=lambda pair: -pair[1])
        return self._collapse([(cid, float(score)) for (cid, _), score in ranked], k)

    # -- what the graph needs -------------------------------------------------

    def brief(self, incident_id: str) -> Prior:
        """The severity, summary and root cause of one incident, for a prompt."""
        sections = self._by_incident[incident_id]
        any_passage = next(iter(sections.values()))
        return Prior(
            incident_id=incident_id,
            service=any_passage.service,
            severity=any_passage.severity,
            summary=sections["summary"].body if "summary" in sections else "",
            root_cause=sections["root_cause"].body if "root_cause" in sections else "",
            remediation=sections["remediation"].body if "remediation" in sections else "",
        )

    # -- internals ------------------------------------------------------------

    def _bm25_ranked(self, query: str, limit: int, service: str | None) -> list[str]:
        """Chunk ids by BM25 score, best first. Zero-score chunks are left out:
        no shared term means no evidence, and RRF would otherwise reward them."""
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self._passages)), key=lambda i: (-scores[i], i))
        out = []
        for i in order:
            passage = self._passages[i]
            if scores[i] <= 0 or (service is not None and passage.service != service):
                continue
            out.append(passage.chunk_id)
            if len(out) == limit:
                break
        return out

    def _fused(self, query: str, service: str | None) -> list[tuple[str, float]]:
        """The RRF merge of the BM25 list and the vector list, chunk level."""
        lexical = self._bm25_ranked(query, BM25_POOL, service)
        dense = [cid for cid, _ in self._vector_search(query, VECTOR_POOL, service)]
        return rrf([lexical, dense])

    def _collapse(self, ranked: Sequence[tuple[str, float]], k: int) -> list[Found]:
        """Keep each incident's best chunk, in rank order, and stop at `k`."""
        seen: set[str] = set()
        found: list[Found] = []
        for chunk_id, score in ranked:
            passage = self._by_chunk[chunk_id]
            if passage.incident_id in seen:
                continue
            seen.add(passage.incident_id)
            found.append(Found(passage.incident_id, chunk_id, passage.service, score))
            if len(found) == k:
                break
        return found


def _check(query: str, k: int) -> None:
    """Reject inputs that would silently return nothing useful."""
    if not query.strip():
        raise ValueError("empty query")
    if k < 1:
        raise ValueError("k must be at least 1")


# -- the real seams -----------------------------------------------------------

_PASSAGES_SQL = "SELECT id, incident_id, section, service, severity, body FROM chunks ORDER BY id"

# `%s::text IS NULL OR service = %s` makes the filter optional in one query.
_VECTOR_SQL = """
SELECT id, embedding <=> %s AS distance
FROM chunks
WHERE (%s::text IS NULL OR service = %s)
ORDER BY embedding <=> %s
LIMIT %s
"""


def load_passages(conn: Connection) -> list[Passage]:
    """Read every chunk. The connection needs no vector support for this."""
    return [Passage(*row) for row in conn.execute(_PASSAGES_SQL).fetchall()]


def postgres_vector_search(conn: Connection, embed: Embedder) -> VectorSearch:
    """Build the real vector search: embed the query, then pgvector cosine.

    The connection needs `register_vector` applied so a `Vector` can be sent.
    """

    def search(query: str, limit: int, service: str | None) -> list[tuple[str, float]]:
        vector = Vector(embed_query(embed, query))
        rows = conn.execute(_VECTOR_SQL, (vector, service, service, vector, limit))
        return [(row[0], float(row[1])) for row in rows.fetchall()]

    return search


def make_reranker() -> Rerank:
    """Load the cross-encoder on the CPU. Imported inside, so importing this
    module never loads torch. The first call downloads about 1.1 GB, pinned to
    `RERANKER_REVISION`."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(
        RERANKER_NAME, revision=RERANKER_REVISION, device="cpu", max_length=512
    )

    def rerank(query: str, texts: Sequence[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        return [float(s) for s in model.predict(pairs, show_progress_bar=False)]

    return rerank
