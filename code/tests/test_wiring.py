"""Tests for app/wiring.py: the composition root, with fakes instead of a database."""

import threading
import time

import pytest

import app.wiring as wiring
from app.triage import Prior


class FakeFound:
    def __init__(self, incident_id: str) -> None:
        self.incident_id = incident_id


class FakeRetriever:
    def __init__(self) -> None:
        self.asked: list[tuple[str, int]] = []

    def hybrid_rerank(self, question: str, k: int):
        self.asked.append((question, k))
        return [FakeFound("INC-1"), FakeFound("INC-2")]

    def brief(self, incident_id: str) -> Prior:
        return Prior(incident_id, "auth", "sev2", "summary", "cause")


def install(monkeypatch, builds: list[int], fail_first: bool = False, delay: float = 0.0) -> FakeRetriever:
    retriever = FakeRetriever()

    def fake_connect():
        return object()

    def fake_build(conn, embedder=None):
        builds.append(1)
        time.sleep(delay)
        if fail_first and len(builds) == 1:
            raise OSError("database is down")
        return retriever

    monkeypatch.setattr(wiring, "connect", fake_connect)
    monkeypatch.setattr(wiring, "build_retriever", fake_build)
    return retriever


def test_recall_builds_nothing_until_it_is_first_used(monkeypatch) -> None:
    builds: list[int] = []
    install(monkeypatch, builds)
    recall = wiring.lazy_recall(3)
    assert builds == []
    recall("q")
    assert builds == [1]


def test_recall_asks_for_k_and_returns_briefs(monkeypatch) -> None:
    retriever = install(monkeypatch, [])
    result = wiring.lazy_recall(5)("what broke?")
    assert retriever.asked == [("what broke?", 5)]
    assert [p.incident_id for p in result] == ["INC-1", "INC-2"]


def test_the_retriever_is_built_once_even_when_runs_overlap(monkeypatch) -> None:
    builds: list[int] = []
    install(monkeypatch, builds, delay=0.05)
    recall = wiring.lazy_recall(3)
    threads = [threading.Thread(target=recall, args=("q",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert builds == [1]  # eight runs, one build


def test_a_failed_first_build_is_retried_on_the_next_call(monkeypatch) -> None:
    builds: list[int] = []
    install(monkeypatch, builds, fail_first=True)
    recall = wiring.lazy_recall(3)
    with pytest.raises(OSError):
        recall("q")
    assert len(recall("q")) == 2  # the database came back
    assert builds == [1, 1]


def test_the_default_k_is_the_shared_recall_k() -> None:
    import inspect

    assert inspect.signature(wiring.lazy_recall).parameters["k"].default == wiring.RECALL_K == 3
