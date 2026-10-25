"""Fast tests for the scoring code. Fake retrievers and a fake clock."""

import pytest

from app.eval_queries import EvalQuery
from app.eval_retrieval import evaluate, evaluate_all, format_table, reciprocal_rank


def q(qid: str, kind: str, *relevant: str) -> EvalQuery:
    return EvalQuery(qid, kind, f"query {qid}", relevant)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "y", "z"], ["y"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["q"]) == 0.0
    assert reciprocal_rank(["x", "y", "z", "w"], ["w", "z"]) == pytest.approx(1 / 3)


def test_recall_all_and_mrr_on_a_known_case() -> None:
    queries = [q("a", "meaning", "1", "2"), q("b", "meaning", "9")]
    answers = {"query a": ["1", "x", "x", "x", "x"], "query b": ["x", "x", "x", "x", "x"]}
    s = evaluate(lambda text, k: answers[text], queries, clock=iter(range(100)).__next__)
    assert s.recall == 0.5  # query a found one of its two, query b found none
    assert s.all_found == 0.0  # query a is missing "2"
    assert s.mrr == 0.5  # (1/1 + 0) / 2


def test_results_beyond_k_do_not_count() -> None:
    s = evaluate(lambda text, k: ["x"] * 5 + ["1"], [q("a", "meaning", "1")], k=5)
    assert s.recall == 0.0


def test_latency_is_the_median_of_the_calls_in_milliseconds() -> None:
    ticks = iter([0.0, 0.010, 1.0, 1.030, 2.0, 2.020])  # calls take 10, 30 and 20 ms
    s = evaluate(lambda text, k: [], [q(str(i), "meaning", "1") for i in range(3)], clock=ticks.__next__)
    assert s.p50_ms == pytest.approx(20.0)


def test_each_retriever_is_warmed_up_once_and_the_warm_up_is_not_scored() -> None:
    calls: list[str] = []

    def search(text, k):
        calls.append(text)
        return ["1"]

    queries = [q("a", "token", "1"), q("b", "meaning", "1")]
    results = evaluate_all({"only": search}, queries)
    assert calls[0] == "query a" and len(calls) == 1 + 2 + 1 + 1  # warm-up, all, token, meaning
    assert results["only"]["all"].queries == 2
    assert results["only"]["token"].queries == 1 and results["only"]["meaning"].queries == 1


def test_the_table_has_a_row_per_retriever_and_group() -> None:
    results = evaluate_all({"a": lambda t, k: ["1"], "b": lambda t, k: []}, [q("x", "token", "1"), q("y", "meaning", "1")])
    table = format_table(results)
    assert table.count("\n") + 1 == 2 + 2 * 3
    assert "| a | all | 2 | 1.00 |" in table and "| b | token | 1 | 0.00 |" in table


def test_no_queries_is_an_error() -> None:
    with pytest.raises(ValueError):
        evaluate(lambda t, k: [], [])
