"""Search against the real Postgres, with hand-made vectors.

The fake model puts text on two axes: "alpha" on the first, "beta" on the
second, "alpha beta" halfway between. Distances are then known in advance, so
the test can check the real SQL: ordering, the limit and the distance values.
It runs in a throwaway schema (the `conn` fixture in conftest.py).
Run with: uv run pytest -m integration tests/integration/test_search_db.py
"""

import math

import psycopg
import pytest

from app.ingest import DIMENSIONS, chunk_incident, ingest
from app.search import search

pytestmark = pytest.mark.integration

META = {"service": "auth", "severity": "sev1", "date": "2026-01-02"}


def fake_embed(texts):
    """Two-axis fake model. Unit length, like the real normalised vectors."""
    vectors = []
    for text in texts:
        x = 1.0 if "alpha" in text else 0.0
        y = 1.0 if "beta" in text else 0.0
        norm = math.hypot(x, y) or 1.0
        vectors.append([x / norm, y / norm] + [0.0] * (DIMENSIONS - 2))
    return vectors


def incident(incident_id: str, body: str) -> list:
    """An incident whose four sections all say `body`."""
    row = {"id": incident_id, **META}
    row.update({s: body for s in ("summary", "timeline", "root_cause", "remediation")})
    return chunk_incident(row)


@pytest.fixture
def filled(conn: psycopg.Connection) -> psycopg.Connection:
    """Three incidents: pure alpha, alpha with beta, pure beta."""
    chunks = incident("INC-A", "alpha") + incident("INC-AB", "alpha beta") + incident("INC-B", "beta")
    ingest(conn, chunks, fake_embed)
    return conn


def test_nearest_comes_first_and_distances_ascend(filled: psycopg.Connection) -> None:
    """Query "alpha": identical vectors first, then the halfway ones, then the opposite."""
    hits = search(filled, fake_embed, "alpha", k=12)
    assert [h.incident_id for h in hits] == ["INC-A"] * 4 + ["INC-AB"] * 4 + ["INC-B"] * 4
    distances = [h.distance for h in hits]
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(0.0, abs=1e-6)
    assert distances[4] == pytest.approx(1 - 1 / math.sqrt(2), abs=1e-6)
    assert distances[8] == pytest.approx(1.0, abs=1e-6)


def test_k_limits_the_rows(filled: psycopg.Connection) -> None:
    """Asking for 3 returns 3, all from the closest incident."""
    hits = search(filled, fake_embed, "alpha", k=3)
    assert len(hits) == 3
    assert {h.incident_id for h in hits} == {"INC-A"}


def test_hits_carry_the_stored_fields(filled: psycopg.Connection) -> None:
    """A hit has what a reader needs: where it came from and its text."""
    hit = search(filled, fake_embed, "beta", k=1)[0]
    assert (hit.incident_id, hit.service, hit.severity) == ("INC-B", "auth", "sev1")
    assert hit.section in {"summary", "timeline", "root_cause", "remediation"}
    assert hit.body == "beta"
