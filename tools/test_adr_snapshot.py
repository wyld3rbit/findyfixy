"""Tests for tools/adr_snapshot.py.

The live page is the source of the ADR pictures from phase 5 on, and phases 1
to 4 are frozen files. These tests keep both true: the latest phase must match
what the page marks built, a frozen picture must keep its recorded hash, a
drawn picture must equal a fresh render, and a phase draws only what existed then.

Run from the training root: `make check`.
"""

import hashlib
import xml.dom.minidom

import pytest

import adr_snapshot as snap

PAGE = snap.PAGE.read_text()
DRAWN = [p for p in snap.PHASES if p not in snap.FROZEN]


def test_the_latest_phase_matches_the_live_page() -> None:
    assert snap.built_in_phase(max(snap.PHASES)) == snap.built_on_page(PAGE)


def test_flipping_a_node_on_the_page_without_a_new_phase_is_caught() -> None:
    old = '<g class="node" data-status="planned" data-layer="harness" data-code="none">\n        <title>Planned for Day 6.'
    flipped = PAGE.replace(old, old.replace('"planned"', '"built"'))
    assert flipped != PAGE
    assert snap.built_in_phase(max(snap.PHASES)) != snap.built_on_page(flipped)


def test_the_frozen_pictures_keep_their_recorded_hashes() -> None:
    for phase, digest in snap.FROZEN.items():
        assert hashlib.sha256(snap.image_path(phase).read_bytes()).hexdigest() == digest, phase


def test_a_changed_frozen_picture_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(snap, "OUT", tmp_path)
    snap.image_path(1).write_text("<svg/>")
    with pytest.raises(snap.FrozenImageChanged):
        snap.check_frozen(1)


def test_a_frozen_phase_is_never_drawn_from_the_page() -> None:
    with pytest.raises(ValueError, match="frozen"):
        snap.snapshot(min(snap.FROZEN), PAGE)


def test_phase_five_draws_both_lanes_and_the_loop_between_them() -> None:
    svg = snap.snapshot(5, PAGE)
    for needle in (">Harness<", ">FindyFixy<", ">cases in<", ">answers scored<",
                   'id="e_eval_q"', 'id="e_rec_eval"'):
        assert needle in svg, needle
    for name in ("Question", "Recall", "Investigate", "Deploy tool", "Small model", "Validate", "Route",
                 "Large model", "Triage record", "Eval gate", "CI gate"):
        block = svg.split(f">{name}<")[0].rsplit("<g ", 1)[1]
        assert 'data-status="built"' in block, name
    for name in ("Tracing", "Policy", "Gateway", "Human review"):
        block = svg.split(f">{name}<")[0].rsplit("<g ", 1)[1]
        assert 'data-status="planned"' in block, name
    assert "data-code" not in svg and "<title>" not in svg and "tool step off" not in svg


def test_a_node_built_later_is_drawn_planned_with_its_day(monkeypatch) -> None:
    # Pretend Tracing was built on Day 6: the Day 5 picture must still show it planned.
    monkeypatch.setitem(snap.FIRST_BUILT, "Tracing", 6)
    page = PAGE.replace('data-status="planned" data-layer="harness" data-code="none">\n        <title>Planned for Day 6.',
                        'data-status="built" data-layer="harness" data-code="none">\n        <title>Planned for Day 6.')
    svg = snap.snapshot(5, page)
    block = svg.split(">Tracing<")[0].rsplit("<g ", 1)[1]
    assert 'data-status="planned"' in block
    assert ">Day 6<" in svg.split(">Tracing<")[1].split("</g>")[0]


def test_every_drawn_snapshot_is_well_formed() -> None:
    for phase in DRAWN:
        xml.dom.minidom.parseString(snap.snapshot(phase, PAGE))


def test_the_committed_drawn_images_equal_a_fresh_render() -> None:
    for phase in DRAWN:
        path = snap.image_path(phase)
        assert path.read_text() == snap.snapshot(phase, PAGE), f"{path.name} is stale: run make adr-snapshot"
