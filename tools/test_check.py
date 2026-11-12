"""Tests for the board rules in tools/check.py.

`card_problems` is pure, so each test hands it one card as a plain dict. The
guard test runs the real record, so a bad status or date fails `make check`.

Run from the training root:
    uv run --no-project --with markdown --with pytest pytest tools
"""

import pytest

import check


@pytest.mark.parametrize(
    "card",
    [
        {"status": "deck"},
        {"status": "deck", "due": "2026-11-12"},
        {"status": "doing", "started": "2026-11-12"},
        {"status": "done", "started": "2026-11-09", "finished": "2026-11-10"},
    ],
)
def test_valid_cards_pass(card: dict) -> None:
    """Each column's minimal valid card has no problems."""
    assert check.card_problems(card, "c") == []


@pytest.mark.parametrize(
    ("card", "expected"),
    [
        ({"status": "todo"}, "is not one of"),
        ({"status": "deck", "due": "Nov 12"}, "not a YYYY-MM-DD date"),
        ({"status": "deck", "due": "2026-02-30"}, "not a YYYY-MM-DD date"),
        ({"status": "deck", "due": "2026-W46-3"}, "not a YYYY-MM-DD date"),
        ({"status": "deck", "date": "11/12/2026"}, "not a YYYY-MM-DD date"),
        ({"status": "deck", "started": "2026-11-10"}, "has a started date but is deck"),
        ({"status": "doing"}, "doing needs a started date"),
        ({"status": "done", "started": "2026-11-09"}, "done needs a finished date"),
        ({"status": "deck", "finished": "2026-11-10"}, "has a finished date but is deck"),
        ({"status": "done", "started": "2026-11-10", "finished": "2026-11-09"}, "finished before it started"),
    ],
)
def test_bad_cards_are_named(card: dict, expected: str) -> None:
    """Each broken rule produces one message that says which rule."""
    found = check.card_problems(card, "c")
    assert len(found) == 1 and expected in found[0], found


@pytest.mark.parametrize("bad_id", ['a"><img src=x>', "has space", ["x"], "Upper"])
def test_task_ids_must_be_slugs(bad_id: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """A task id that is not a lowercase slug is reported, never a crash."""
    record = check.load_progress()
    record["tasks"] = [{"id": bad_id, "title": "t", "status": "deck", "note": "n"}]
    monkeypatch.setattr(check, "load_progress", lambda: record)
    assert any("lowercase slug" in p for p in check.problems())


def test_the_real_record_is_valid() -> None:
    """The committed record passes every check, board rules included."""
    assert check.problems() == []
