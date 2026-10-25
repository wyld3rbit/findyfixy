"""Fast tests for the labelled retrieval queries. No model, no database.

They run on the real corpus, because the labels are only worth anything if
they match the real data.
"""

import json

import pytest

from app.eval_queries import (
    EVAL_PATH,
    HOSTNAME,
    MAX_TOKEN_INCIDENTS,
    TEXT_FIELDS,
    build_queries,
    load_queries,
    meaning_queries,
    partners,
    read_rows,
    token_queries,
)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    """The real corpus, read once."""
    return read_rows()


def test_forty_queries_ten_of_them_exact_token(rows) -> None:
    """The lesson asks for 40, with 10 that hinge on an exact token."""
    queries = build_queries(rows)
    assert len(queries) == 40
    assert sum(q.kind == "token" for q in queries) == 10
    assert len({q.id for q in queries}) == 40


def test_every_label_is_a_real_incident_and_none_is_empty(rows) -> None:
    """A label that names a missing incident would make recall unreachable."""
    known = {row["id"] for row in rows}
    for query in build_queries(rows):
        assert query.relevant, query.id
        assert set(query.relevant) <= known, query.id


def test_meaning_labels_are_the_anchor_and_its_twin(rows) -> None:
    """The answer to a meaning query is both voices of the incident."""
    twin = partners(rows)
    by_id = {q.id: q for q in meaning_queries(rows)}
    # m01 is written against INC-0009, whose twin is INC-0015.
    assert by_id["m01"].relevant == tuple(sorted(("INC-0009", twin["INC-0009"])))
    # The gateway query covers both identical pairs, so four incidents.
    assert len(by_id["m02"].relevant) == 4


def test_meaning_queries_hold_no_hostname(rows) -> None:
    """A hostname in a meaning query would let keyword search win it cheaply."""
    for query in meaning_queries(rows):
        assert not HOSTNAME.search(query.query), query.id


def test_token_labels_are_exactly_the_incidents_that_contain_the_token(rows) -> None:
    """Recomputed independently of the module's own helper."""
    for query in token_queries(rows):
        token = query.query.rsplit(" ", 1)[-1]
        containing = sorted(
            row["id"] for row in rows if any(token in row[field] for field in TEXT_FIELDS)
        )
        assert list(query.relevant) == containing, query.id
        assert 1 <= len(containing) <= MAX_TOKEN_INCIDENTS


def test_an_anchor_that_is_not_seeded_is_refused(rows, monkeypatch) -> None:
    """A typo in an anchor ID must stop the build, not label the wrong thing."""
    from app import eval_queries

    monkeypatch.setattr(eval_queries, "MEANING", ((("INC-0001",), "x"),))
    with pytest.raises(ValueError):
        eval_queries.meaning_queries(rows)


def test_hostname_pattern_skips_ordinary_hyphenated_words() -> None:
    """`re-add` has only hex letters after the dash. It must not count as a host."""
    assert HOSTNAME.findall("re-add and dead-letter on scheduler-3de") == ["scheduler-3de"]


def test_the_committed_file_matches_what_the_code_builds(rows) -> None:
    """Catches a hand edit of the data file, or a corpus change with no rebuild."""
    assert [json.loads(line) for line in EVAL_PATH.read_text().splitlines()] == [
        {"id": q.id, "kind": q.kind, "query": q.query, "relevant": list(q.relevant)}
        for q in build_queries(rows)
    ]
    assert load_queries() == build_queries(rows)
