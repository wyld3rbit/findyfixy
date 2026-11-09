"""Tests for app/wiring.py: the composition root, with fakes instead of a database."""

import threading
import time

import pytest

import app.wiring as wiring
from app.triage import Prior


class FakeFound:
    """A retrieval hit: only the incident id is read."""

    def __init__(self, incident_id: str) -> None:
        self.incident_id = incident_id


class FakeRetriever:
    """Records what it was asked and returns two fixed hits."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, int]] = []

    def hybrid_rerank(self, question: str, k: int):
        self.asked.append((question, k))
        return [FakeFound("INC-1"), FakeFound("INC-2")]

    def brief(self, incident_id: str) -> Prior:
        return Prior(incident_id, "auth", "sev2", "summary", "cause", "fix")


class FakeConnection:
    """Counts closes, so a failed build can be shown to clean up."""

    closed = 0

    def close(self) -> None:
        """Record the close."""
        FakeConnection.closed += 1


def install(monkeypatch, builds: list[int], fail_first: bool = False, delay: float = 0.0) -> FakeRetriever:
    """Replace the database and the retriever build with fakes. Returns the fake retriever."""
    retriever = FakeRetriever()
    FakeConnection.closed = 0

    def fake_connect():
        return FakeConnection()

    def fake_build(conn, embedder=None):
        builds.append(1)
        time.sleep(delay)
        if fail_first and len(builds) == 1:
            raise OSError("database is down")
        return retriever

    monkeypatch.setattr(wiring, "connect", fake_connect)
    monkeypatch.setattr(wiring, "build_retriever", fake_build)
    return retriever


def shared() -> wiring.SharedRetrieval:
    """A shared retrieval whose embedder is a fake: no model is loaded."""
    return wiring.SharedRetrieval(make_embedder=lambda: lambda texts: [[float(len(t))] for t in texts])


def test_recall_builds_nothing_until_it_is_first_used(monkeypatch) -> None:
    builds: list[int] = []
    install(monkeypatch, builds)
    recall = shared().recall(3)
    assert builds == []
    recall("q")
    assert builds == [1]


def test_recall_asks_for_k_and_returns_briefs(monkeypatch) -> None:
    retriever = install(monkeypatch, [])
    result = shared().recall(5)("what broke?")
    assert retriever.asked == [("what broke?", 5)]
    assert [p.incident_id for p in result] == ["INC-1", "INC-2"]


def test_the_retriever_is_built_once_even_when_runs_overlap(monkeypatch) -> None:
    builds: list[int] = []
    install(monkeypatch, builds, delay=0.05)
    recall = shared().recall(3)
    threads = [threading.Thread(target=recall, args=("q",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert builds == [1]  # eight runs, one build


def test_a_failed_first_build_is_retried_on_the_next_call(monkeypatch) -> None:
    builds: list[int] = []
    install(monkeypatch, builds, fail_first=True)
    recall = shared().recall(3)
    with pytest.raises(OSError):
        recall("q")
    assert FakeConnection.closed == 1  # the failed build closed its connection
    assert len(recall("q")) == 2  # the database came back
    assert builds == [1, 1]


def test_the_default_k_is_the_shared_recall_k() -> None:
    import inspect

    assert inspect.signature(wiring.SharedRetrieval.recall).parameters["k"].default == wiring.RECALL_K == 3


def test_recall_and_the_judge_share_one_embedder_and_one_retriever(monkeypatch) -> None:
    """The point of sharing: triage recall and eval candidates reuse one build."""
    builds: list[int] = []
    install(monkeypatch, builds)
    made: list[int] = []

    def make_embedder():
        made.append(1)
        return lambda texts: [[1.0] for _ in texts]

    retrieval = wiring.SharedRetrieval(make_embedder=make_embedder)
    retrieval.recall(3)("q")
    retrieval.candidates("q", 4)
    assert retrieval.embed(["a", "b"]) == [[1.0], [1.0]]
    assert builds == [1] and made == [1]


def test_close_closes_the_connection_and_a_later_use_rebuilds(monkeypatch) -> None:
    """Closing releases the database; the next call builds again."""
    builds: list[int] = []
    install(monkeypatch, builds)
    retrieval = shared()
    retrieval.candidates("q", 1)
    retrieval.close()
    retrieval.close()  # harmless twice
    assert FakeConnection.closed == 1
    retrieval.candidates("q", 1)
    assert builds == [1, 1]
