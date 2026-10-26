"""Fast tests for the golden set. No model, no database, no network.

They run on the real corpus, because the labels are only worth something if
they match the real data. Bad-input tests use small variations of it.
"""

from dataclasses import replace

import pytest

from app.golden import (
    DEFAULT_PATH,
    MAX_QUESTION_CHARS,
    VALID_SEVERITIES,
    GoldenCase,
    build_golden,
    load_golden,
    main,
    read_incidents,
    seeded_pairs,
    write_golden,
)


@pytest.fixture(scope="module")
def incidents():
    """The real corpus, read once."""
    return read_incidents()


@pytest.fixture(scope="module")
def cases(incidents) -> list[GoldenCase]:
    """The built cases, shared. Building is pure, so sharing is safe."""
    return build_golden(incidents)


def test_thirty_cases_split_twenty_ten(cases) -> None:
    """20 twin and 10 no_twin, ids g01..g30, all unique."""
    assert len(cases) == 30
    assert sum(c.group == "twin" for c in cases) == 20
    assert sum(c.group == "no_twin" for c in cases) == 10
    assert [c.id for c in cases] == [f"g{n:02d}" for n in range(1, 31)]


def test_twin_cases_link_to_the_source_through_twin_of(cases, incidents) -> None:
    """twin_id exists, and the corpus links the two in one direction or the other."""
    by_id = {i.id: i for i in incidents}
    twins = [c for c in cases if c.group == "twin"]
    assert len({c.source_id for c in twins}) == 20
    for case in twins:
        assert case.twin_id in by_id
        assert case.twin_id != case.source_id
        linked = (
            by_id[case.source_id].twin_of == case.twin_id
            or by_id[case.twin_id].twin_of == case.source_id
        )
        assert linked, case.id


def test_both_directions_of_a_pair_are_used_as_source(cases, incidents) -> None:
    """Some sources are the original and some the later twin row."""
    by_id = {i.id: i for i in incidents}
    twins = [c for c in cases if c.group == "twin"]
    later_rows = sum(by_id[c.source_id].twin_of is not None for c in twins)
    assert 0 < later_rows < 20


def test_no_twin_cases_have_no_twin_id_and_no_pair(cases, incidents) -> None:
    """A no_twin source is in no pair, in either direction."""
    in_pair = {x for pair in seeded_pairs(incidents) for x in pair}
    for case in (c for c in cases if c.group == "no_twin"):
        assert case.twin_id is None
        assert case.source_id not in in_pair


def test_no_twin_cases_spread_over_services_and_severities(cases, incidents) -> None:
    """The control group must not sit in one corner of the corpus."""
    by_id = {i.id: i for i in incidents}
    control = [by_id[c.source_id] for c in cases if c.group == "no_twin"]
    assert len({i.service for i in control}) >= 6
    assert len({i.severity for i in control}) >= 3


def test_question_never_contains_the_answer(cases, incidents) -> None:
    """The question is the alert line alone: no summary, no later steps, no answer."""
    by_id = {i.id: i for i in incidents}
    for case in cases:
        source = by_id[case.source_id]
        assert source.root_cause not in case.question, case.id
        assert source.remediation not in case.question, case.id
        assert case.question == source.timeline.splitlines()[0].strip(), case.id
        assert source.summary not in case.question, case.id
        for later in source.timeline.splitlines()[1:]:
            assert later.strip() not in case.question, case.id
        assert source.root_cause in case.reference
        assert source.remediation in case.reference


def test_questions_stay_under_the_cap(cases) -> None:
    """The triage prompt caps input at 3500 characters."""
    assert all(0 < len(c.question) < MAX_QUESTION_CHARS for c in cases)


def test_expected_severity_is_valid_and_matches_the_source(cases, incidents) -> None:
    """The label is the incident's own severity."""
    by_id = {i.id: i for i in incidents}
    for case in cases:
        assert case.expected_severity in VALID_SEVERITIES
        assert case.expected_severity == by_id[case.source_id].severity


def test_build_is_deterministic(incidents) -> None:
    """Two builds are equal."""
    assert build_golden(incidents) == build_golden(incidents)


def test_load_round_trips_and_bytes_are_stable(cases, tmp_path) -> None:
    """Write, load, equal. Writing twice gives identical bytes."""
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_golden(cases, first)
    write_golden(cases, second)
    assert first.read_bytes() == second.read_bytes()
    assert load_golden(first) == cases


def test_the_committed_file_matches_a_fresh_build(cases, tmp_path) -> None:
    """Catches a hand edit of the data file, or a corpus change with no rebuild."""
    fresh = tmp_path / "fresh.jsonl"
    write_golden(cases, fresh)
    assert DEFAULT_PATH.read_bytes() == fresh.read_bytes()
    assert load_golden() == cases


def test_too_few_pairs_is_refused(incidents) -> None:
    """With the pair labels stripped, the build must stop, not shrink the set."""
    stripped = [replace(i, twin_of=None) for i in incidents]
    with pytest.raises(ValueError, match="seeded pairs"):
        build_golden(stripped)


def test_too_few_unpaired_incidents_is_refused(incidents) -> None:
    """A corpus that is nearly all pairs cannot fill the control group."""
    pairs = {x for pair in seeded_pairs(incidents) for x in pair}
    only_pairs = [i for i in incidents if i.id in pairs]
    with pytest.raises(ValueError, match="unpaired"):
        build_golden(only_pairs)


def test_show_prints_every_case_with_its_question_and_reference(cases, capsys) -> None:
    """The review view lists all 30 cases and changes nothing on disk."""
    before = DEFAULT_PATH.read_bytes()
    main(["--show"])
    out = capsys.readouterr().out
    assert out.count("QUESTION") == 30 and out.count("REFERENCE") == 30
    assert cases[0].question in out and cases[0].id in out
    assert DEFAULT_PATH.read_bytes() == before


def test_unknown_argument_is_refused() -> None:
    with pytest.raises(SystemExit):
        main(["--nope"])
