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
from typing import TYPE_CHECKING

from app.triage import Prior

if TYPE_CHECKING:
    from psycopg import Connection

    from app.graph import Recall
    from app.ingest import Embedder
    from app.retrieve import Retriever

# Past incidents shown to the model. Three keeps the prompt small: each one costs
# prompt tokens and, with a small model, starts to distract it.
RECALL_K = 3


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


def build_retriever(conn: "Connection", embedder: "Embedder | None" = None) -> "Retriever":
    """Hybrid retriever over every passage in the database.

    Loads the embedding model and the reranker (about 1.3 GB of weights). Pass an
    `embedder` you already have to share one loaded model.
    """
    from app.ingest import make_embedder
    from app.retrieve import Retriever, load_passages, make_reranker, postgres_vector_search

    embedder = embedder or make_embedder()
    return Retriever(load_passages(conn), postgres_vector_search(conn, embedder), make_reranker())


def lazy_recall(k: int = RECALL_K) -> "Recall":
    """The real recall step, built on first use instead of at startup.

    Building it loads two models and opens the database. Doing that at startup
    would slow every start and fail it when the database is off, though fake mode
    needs neither. On first use a failure becomes a `run_failed` event, which the
    page shows. The retriever is kept after the first build. One lock keeps two
    runs from building it twice, and from sharing the connection and the models
    at the same moment (neither is safe across threads).
    """
    lock = threading.Lock()
    built: list["Retriever"] = []

    def recall(question: str) -> list[Prior]:
        with lock:
            if not built:
                built.append(build_retriever(connect()))
            retriever = built[0]
            found = retriever.hybrid_rerank(question, k)
            return [retriever.brief(f.incident_id) for f in found]

    return recall
