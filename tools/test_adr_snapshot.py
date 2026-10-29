"""Tests for tools/adr_snapshot.py.

The live page is the source of the ADR pictures. These tests keep the pictures true to it:
the latest phase must match what the page marks built, the committed images must equal a
fresh render, and each phase must draw only what existed then.

Run from the training root: `make check`.
"""

import xml.dom.minidom

import adr_snapshot as snap

PAGE = snap.PAGE.read_text()


def test_the_latest_phase_matches_the_live_page() -> None:
    assert snap.built_in_phase(max(snap.PHASES)) == snap.built_on_page(PAGE)


def test_flipping_a_node_on_the_page_without_a_new_phase_is_caught() -> None:
    flipped = PAGE.replace('<g class="node" data-status="planned">\n        <title>Planned for Day 6.', '<g class="node" data-status="built">\n        <title>Planned for Day 6.')
    assert flipped != PAGE
    assert snap.built_in_phase(max(snap.PHASES)) != snap.built_on_page(flipped)


def test_phase_one_draws_only_the_router() -> None:
    svg = snap.snapshot(1, PAGE)
    for name in ("Question", "Small model", "Route", "Large model", "Triage record"):
        assert f">{name}<" in svg
    for name in ("Investigate", "Validate", "Recall", "Deploy tool"):
        assert f">{name}<" not in svg


def test_phase_two_adds_validation_and_the_tool() -> None:
    svg = snap.snapshot(2, PAGE)
    for name in ("Investigate", "Validate", "Deploy tool"):
        assert f">{name}<" in svg
    assert ">Recall<" not in svg


def test_a_node_built_later_is_drawn_planned_with_its_day() -> None:
    svg = snap.snapshot(1, PAGE)
    evals = svg.split(">Evals<")[0].rsplit("<g ", 1)[1]
    assert 'data-status="planned"' in evals
    assert "Day 5" in svg.split(">Evals<")[1].split("</g>")[0]


def test_every_snapshot_is_well_formed() -> None:
    for phase in snap.PHASES:
        xml.dom.minidom.parseString(snap.snapshot(phase, PAGE))


def test_the_committed_images_equal_a_fresh_render() -> None:
    for phase in snap.PHASES:
        path = snap.OUT / f"{phase:04d}-architecture.svg"
        assert path.read_text() == snap.snapshot(phase, PAGE), f"{path.name} is stale: run make adr-snapshot"
