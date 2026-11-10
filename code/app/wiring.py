"""The composition root: where the real database, models and retriever are built.

Why: the retriever was assembled by hand in three places (the console, the
answer evaluation and the retrieval evaluation), each with its own copy of
"connect, register the vector type, load every passage, build the embedder and
the reranker". A change to how recall is built had to be made three times.
Everything that needs the real thing now builds it here.

The rest of the code takes its dependencies as arguments (a `Recall`, an `Ask`,
a `Judge`) and never imports this module. Only entry points do. That keeps the
domain code testable with fakes.

Heavy imports (psycopg, the embedding model, the reranker) happen inside the
functions, so importing this module costs nothing.
"""

import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from app.triage import RECALL_K, Prior

if TYPE_CHECKING:
    from psycopg import Connection

    from app.graph import Recall
    from app.ingest import Embedder
    from contextlib import AbstractContextManager

    from app.retrieve import Retriever, Stage
    from harness.trace.tracing import Tracing


def connect() -> "Connection":
    """Open the project database with the vector type registered.

    Autocommit, because the app only reads here. The caller closes it (use it as
    a context manager).
    """
    import psycopg
    from pgvector.psycopg import register_vector

    from app.ingest import database_url

    conn = psycopg.connect(database_url(), autocommit=True)
    register_vector(conn)
    return conn


def build_retriever(
    conn: "Connection", embedder: "Embedder | None" = None, stage: "Stage | None" = None
) -> "Retriever":
    """Hybrid retriever over every passage in the database.

    Loads the embedding model and the reranker (about 1.3 GB of weights). Pass an
    `embedder` you already have to share one loaded model. `stage` wraps each
    retrieval stage (`stage_spans` makes them tracing spans).
    """
    from app.ingest import make_embedder
    from app.retrieve import Retriever, load_passages, make_reranker, no_stages, postgres_vector_search

    embedder = embedder or make_embedder()
    return Retriever(load_passages(conn), postgres_vector_search(conn, embedder), make_reranker(),
                     stage=stage or no_stages)


def stage_spans(tracing: "Tracing | None") -> "Stage | None":
    """A retriever `Stage` that makes each stage a span under the current one, or None without tracing.

    The recall node's span is current while recall runs (app/trace_adapter.py),
    so the stages appear inside it. Outside any span they would each start a
    trace of their own, which is why this is only built when tracing is on.
    """
    if tracing is None:
        return None
    tracer = tracing.tracer()

    def stage(name: str) -> "AbstractContextManager[object]":
        """One span named for the stage, current while it runs."""
        return tracer.start_as_current_span(f"recall {name}", attributes={"langfuse.observation.type": "span"})

    return stage


class SharedRetrieval:
    """One database connection, embedder and retriever per process, built on first use.

    Why shared: the console runs triage (recall) and, from Day 5, eval runs
    whose judge also needs the embedder. Built separately, they held two copies
    of the embedding model and the reranker (about 1.3 GB) and two database
    connections, on a 16 GB machine that also holds two answering models and a
    judge. One instance serves both.

    Why lazy: building it loads two models and opens the database. Doing that
    at startup would slow every start and fail it when the database is off,
    though fake mode needs neither. On first use a failure propagates (the
    console turns it into a failed run) and the next call tries again. What was
    built is kept.

    Thread safety: one lock covers every use. Neither a psycopg connection nor
    a loaded model is safe to call from two threads at once, and the lock also
    keeps two first calls from building twice. It is held only for retrieval
    and embedding (well under a second), never across a model call.
    """

    def __init__(
        self, *, make_embedder: "Callable[[], Embedder] | None" = None, stage: "Stage | None" = None
    ) -> None:
        """Hold nothing yet. `make_embedder` is for tests; `stage` traces the retrieval stages."""
        self._make_embedder = make_embedder
        self._stage = stage
        self._lock = threading.Lock()
        self._embedder: "Embedder | None" = None
        self._conn: "Connection | None" = None
        self._retriever: "Retriever | None" = None

    def _load_embedder(self) -> "Embedder":
        """The embedder, loaded once. The caller holds the lock."""
        if self._embedder is None:
            if self._make_embedder is None:
                from app.ingest import make_embedder

                self._make_embedder = make_embedder
            self._embedder = self._make_embedder()
        return self._embedder

    def _load_retriever(self) -> "Retriever":
        """The retriever over the shared embedder, built once. The caller holds the lock.

        A failed build closes the connection it opened, so a retry starts clean.
        """
        if self._retriever is None:
            embedder = self._load_embedder()
            conn = connect()
            try:
                self._retriever = build_retriever(conn, embedder, self._stage)
            except BaseException:
                conn.close()
                raise
            self._conn = conn
        return self._retriever

    def candidates(self, question: str, n: int) -> list[Prior]:
        """Top `n` incidents by hybrid search and rerank, as prompt-ready briefs."""
        with self._lock:
            retriever = self._load_retriever()
            return [retriever.brief(f.incident_id) for f in retriever.hybrid_rerank(question, n)]

    def recall(self, k: int = RECALL_K) -> "Recall":
        """The graph's recall step: the top `k` candidates for a question."""

        def recall(question: str) -> list[Prior]:
            """Similar past incidents for one question."""
            return self.candidates(question, k)

        return recall

    def embed(self, texts: "Sequence[str]") -> list[list[float]]:
        """Embed texts with the shared model (the judge's relevancy metric needs it)."""
        with self._lock:
            return self._load_embedder()(texts)

    def close(self) -> None:
        """Close the database connection and drop what was built. Safe to call twice."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
            self._conn = self._retriever = None
