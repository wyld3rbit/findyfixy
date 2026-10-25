"""Fast tests for the retrievers. Real BM25, fake vector search and reranker.

No database, no model, no network. The SQL and the real models are covered by
the eval run and the integration tests.
"""

import pytest

from app.retrieve import (
    RERANK_POOL,
    SECTIONS,
    Passage,
    Retriever,
    rrf,
    tokenize,
)


def passage(incident: str, section: str, body: str, service: str = "auth") -> Passage:
    return Passage(f"{incident}:{section}", incident, section, service, body)


# Four incidents. A is the host-name incident, B and C are about meaning, D is
# another service's twin of B's wording.
PASSAGES = [
    passage("A", "summary", "Clock drift on scheduler-3de rejected logins.", "scheduler"),
    passage("A", "root_cause", "NTP lost its upstream on scheduler-3de.", "scheduler"),
    passage("B", "summary", "Login latency regression after certificate rotation."),
    passage("B", "root_cause", "Peers lacked the intermediate authority."),
    passage("C", "summary", "Queue depth grew because consumers stalled."),
    passage("D", "summary", "Login latency regression after certificate rotation.", "billing"),
]


def fake_vector(order: list[str]):
    """A vector search that returns a fixed order, honouring the service filter."""
    by_id = {p.chunk_id: p for p in PASSAGES}

    def search(query, limit, service):
        ids = [c for c in order if service is None or by_id[c].service == service]
        return [(cid, 0.1 * i) for i, cid in enumerate(ids[:limit])]

    return search


def retriever(order: list[str], rerank=None) -> Retriever:
    return Retriever(PASSAGES, fake_vector(order), rerank)


def test_tokenize_keeps_a_hostname_whole_and_adds_its_parts() -> None:
    assert tokenize("Down on scheduler-3de!") == ["down", "on", "scheduler-3de", "scheduler", "3de"]


def test_rrf_rewards_a_chunk_that_is_high_in_both_lists() -> None:
    fused = rrf([["x", "y", "z"], ["y", "q", "x"]])
    assert [item for item, _ in fused][:2] == ["y", "x"]


def test_rrf_scores_follow_the_formula() -> None:
    (item, score), = rrf([["x"], ["x"]], k=60)
    assert item == "x" and score == pytest.approx(2 / 61)


def test_vector_returns_incidents_not_chunks() -> None:
    """Two chunks of one incident must not take two slots."""
    out = retriever(["A:summary", "A:root_cause", "B:summary", "C:summary"]).vector("q", k=3)
    assert [f.incident_id for f in out] == ["A", "B", "C"]


def test_vector_asks_for_enough_chunks_to_fill_k_incidents() -> None:
    seen = []

    def spy(query, limit, service):
        seen.append(limit)
        return []

    Retriever(PASSAGES, spy).vector("q", k=5)
    assert seen == [5 * SECTIONS]


def test_hybrid_finds_an_exact_host_that_vector_ranks_last() -> None:
    """The point of BM25: a vector list that buries the host does not bury it."""
    burying = ["B:summary", "C:summary", "D:summary", "B:root_cause", "A:summary", "A:root_cause"]
    r = retriever(burying)
    assert r.vector("what is wrong on scheduler-3de", k=1)[0].incident_id == "B"
    assert r.hybrid("what is wrong on scheduler-3de", k=1)[0].incident_id == "A"


def test_hybrid_still_finds_a_paraphrase_that_has_no_shared_word() -> None:
    """With no keyword match, the vector list alone carries the answer."""
    out = retriever(["C:summary", "B:summary"]).hybrid("zzz qqq", k=1)
    assert out[0].incident_id == "C"


def test_bm25_leaves_out_chunks_with_no_matching_term() -> None:
    r = retriever(["C:summary"])
    assert r._bm25_ranked("zzz qqq", 10, None) == []


@pytest.mark.parametrize("method", ["vector", "hybrid"])
def test_service_filter_never_returns_another_service(method) -> None:
    out = getattr(retriever([p.chunk_id for p in PASSAGES]), method)(
        "login latency regression certificate rotation", k=5, service="billing"
    )
    assert out and {f.service for f in out} == {"billing"}


def test_rerank_reorders_the_fused_pool_and_uses_its_scores() -> None:
    seen = []

    def prefer_stalled(query, texts):
        seen.append(len(texts))
        return [10.0 if "stalled" in t else 0.0 for t in texts]

    r = retriever(["B:summary", "C:summary"], prefer_stalled)
    out = r.hybrid_rerank("login latency", k=2)
    assert out[0].incident_id == "C" and out[0].score == 10.0
    assert seen and seen[0] <= RERANK_POOL


def test_rerank_pool_is_capped() -> None:
    many = [passage(f"I{n}", "summary", "common words here") for n in range(60)]
    seen = []

    def count(query, texts):
        seen.append(len(texts))
        return [0.0] * len(texts)

    ids = [p.chunk_id for p in many]
    Retriever(many, lambda q, n, s: [(c, 0.0) for c in ids[:n]], count).hybrid_rerank("common words", k=3)
    assert seen == [RERANK_POOL]


def test_rerank_without_a_reranker_is_an_error_not_a_silent_fallback() -> None:
    with pytest.raises(ValueError, match="reranker"):
        retriever(["B:summary"]).hybrid_rerank("q", k=1)


def test_a_reranker_that_returns_the_wrong_count_is_refused() -> None:
    with pytest.raises(ValueError, match="number of scores"):
        retriever(["B:summary"], lambda q, texts: [1.0]).hybrid_rerank("login latency", k=1)


@pytest.mark.parametrize("query,k", [("", 3), ("   ", 3), ("ok", 0)])
def test_bad_inputs_are_rejected(query, k) -> None:
    with pytest.raises(ValueError):
        retriever(["B:summary"]).hybrid(query, k)


def test_brief_gives_the_summary_and_the_root_cause() -> None:
    prior = retriever([]).brief("B")
    assert (prior.incident_id, prior.service) == ("B", "auth")
    assert "certificate rotation" in prior.summary and "intermediate" in prior.root_cause


def test_an_empty_corpus_is_refused() -> None:
    with pytest.raises(ValueError):
        Retriever([], lambda q, n, s: [])
