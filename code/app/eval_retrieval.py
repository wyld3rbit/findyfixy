"""Day 4, step 3: score the three retrievers on the labelled queries.

What it does
    Runs every query in `data/retrieval_eval.jsonl` through `vector`, `hybrid`
    and `hybrid_rerank`, and prints one table: recall@5, all@5, MRR and p50
    latency, for all queries and split by kind (exact-token and meaning).

Run it:  uv run python -m app.eval_retrieval

The metrics
    recall@5: the share of queries where at least one correct incident is in
        the top 5. "Did the right thing reach the prompt at all?"
    all@5: the share where EVERY correct incident is in the top 5. Stricter.
        For a seeded pair it asks whether both voices of the incident came back.
    MRR (mean reciprocal rank): 1 / rank of the first correct incident,
        averaged. Rank 1 scores 1, rank 4 scores 0.25, a miss scores 0. It
        rewards putting the answer first, which recall@5 ignores.
    p50 latency: the median wall time of one call, including embedding the
        query and, for `hybrid_rerank`, the cross-encoder. The first call of
        each retriever is run and thrown away: it pays model loading.

Reading the numbers honestly
    The corpus is templated, so absolute scores flatter. Only the gap between
    retrievers is informative. 40 queries is a small set: one query is 2.5
    points, so a gap of a few points is noise. A median over 40 calls on a
    laptop CPU also moves with whatever else the machine is doing.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from app.eval_queries import EvalQuery, load_queries

K = 5

# (query, k) -> incident ids, best first. The eval sees ids only, never scores.
Search = Callable[[str, int], list[str]]


@dataclass(frozen=True)
class Scores:
    """The four numbers for one retriever over one group of queries."""

    queries: int
    recall: float
    all_found: float
    mrr: float
    p50_ms: float


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first correct id, or 0 when none is in the list."""
    wanted = set(relevant)
    for rank, incident_id in enumerate(ranked, start=1):
        if incident_id in wanted:
            return 1.0 / rank
    return 0.0


def evaluate(
    search: Search,
    queries: Sequence[EvalQuery],
    *,
    k: int = K,
    clock: Callable[[], float] = time.perf_counter,
) -> Scores:
    """Score one retriever. The clock is injected so tests need no real time."""
    if not queries:
        raise ValueError("no queries to score")
    recalled = complete = 0
    ranks: list[float] = []
    millis: list[float] = []
    for query in queries:
        start = clock()
        ranked = search(query.query, k)
        millis.append((clock() - start) * 1000)
        wanted = set(query.relevant)
        top = set(ranked[:k])
        recalled += bool(wanted & top)
        complete += wanted <= top
        ranks.append(reciprocal_rank(ranked[:k], query.relevant))
    n = len(queries)
    return Scores(n, recalled / n, complete / n, sum(ranks) / n, statistics.median(millis))


def evaluate_all(
    searches: Mapping[str, Search],
    queries: Sequence[EvalQuery],
    *,
    k: int = K,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, dict[str, Scores]]:
    """Score every retriever on all queries and on each kind.

    The first query is run once per retriever and discarded, so model loading
    does not land in the latency of the first measured call.
    """
    out: dict[str, dict[str, Scores]] = {}
    kinds = ("all", "token", "meaning")
    for name, search in searches.items():
        search(queries[0].query, k)  # warm-up, not measured
        out[name] = {
            kind: evaluate(
                search, [q for q in queries if kind in ("all", q.kind)], k=k, clock=clock
            )
            for kind in kinds
        }
    return out


def format_table(results: Mapping[str, Mapping[str, Scores]]) -> str:
    """A Markdown table, one row per retriever and group."""
    lines = [
        "| Retriever | Queries | n | recall@5 | all@5 | MRR | p50 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, groups in results.items():
        for kind, s in groups.items():
            lines.append(
                f"| {name} | {kind} | {s.queries} | {s.recall:.2f} | {s.all_found:.2f} "
                f"| {s.mrr:.3f} | {s.p50_ms:.0f} |"
            )
    return "\n".join(lines)


def main() -> None:
    """Build the real retrievers and print the table.

    Loads the embedding model and the reranker (about 1.3 GB of weights on the
    first run), and needs the database from `make up` with Day 3 ingested.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    from app.ingest import database_url, make_embedder
    from app.retrieve import Retriever, load_passages, make_reranker, postgres_vector_search

    queries = load_queries()
    with psycopg.connect(database_url(), autocommit=True) as conn:
        register_vector(conn)
        retriever = Retriever(
            load_passages(conn),
            postgres_vector_search(conn, make_embedder()),
            make_reranker(),
        )

        def ids(method: Callable[..., list]) -> Search:
            return lambda query, k: [f.incident_id for f in method(query, k)]

        results = evaluate_all(
            {
                "vector": ids(retriever.vector),
                "hybrid": ids(retriever.hybrid),
                "hybrid_rerank": ids(retriever.hybrid_rerank),
            },
            queries,
        )
    print(format_table(results))


if __name__ == "__main__":
    main()
