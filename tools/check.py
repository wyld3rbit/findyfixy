"""Validate the tracker's data before committing.

Run from the training root:
    uv run --no-project --with markdown python tools/check.py

Checks:
1. record/progress.json parses as JSON.
2. Every day has the required fields, and every `done` index points at a real check.
   Every card on the board (each day and each task) has a valid status and dates.
3. Every lesson file renders to non-empty HTML.
4. Decisions and log entries have all fields.
5. The README and CLAUDE.md name only make targets and files that exist (tools/check_docs.py).
6. No em-dashes in lessons or progress data.

Exits non-zero on the first class of failure, with a one-line reason per problem.
"""

import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_docs  # noqa: E402
from lessons import LESSON_NUMBERS, ROOT, lesson_file, render  # noqa: E402

REQUIRED_DAY = {"n", "week", "date", "title", "file", "outcome", "checks", "done", "status"}
REQUIRED_DECISION = {"date", "day", "decision", "rejected", "why"}
REQUIRED_TOP = {"updated", "plannedStart", "plannedEnd", "countdownStart", "countdownEnd", "log", "days", "tasks"}
REQUIRED_TASK = {"id", "title", "status", "note"}
# The board's three columns. The status is set by hand as work moves, never
# derived from self-checks: from Day 5 the checks are ticked only in the final
# review, so a built day would otherwise sit in "doing" for weeks.
STATUSES = ("deck", "doing", "done")
# A day's `date` is its due date on the board, so it is checked like `due`.
DATE_FIELDS = ("date", "started", "finished", "due")
# Only YYYY-MM-DD. date.fromisoformat alone also accepts ISO week dates
# such as 2026-W46-3, which the page cannot format or compare.
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# A task id becomes an HTML attribute and the URL hash on the page, so it is
# a lowercase slug: no quotes, spaces or "#" that could break either.
TASK_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


PROGRESS = ROOT / "record" / "progress.json"


def load_progress() -> dict:
    """Parse record/progress.json into a dict."""
    return json.loads(PROGRESS.read_text())


def _is_date(value: object) -> bool:
    """True for an ISO calendar date such as 2026-11-11."""
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        return False
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def card_problems(card: dict, label: str) -> list[str]:
    """Problems with one board card's status and dates. `label` names it in messages.

    A card in doing must say when it started, and one in done when it finished,
    because the board shows exactly that date on the card. Dates must be real
    and in order: a finish before its start is a typo the page would show as fact.
    """
    out: list[str] = []
    status = card.get("status")
    if status not in STATUSES:
        return [f"{label}: status {status!r} is not one of {list(STATUSES)}"]
    bad = [f for f in DATE_FIELDS if f in card and not _is_date(card[f])]
    if bad:
        return [f"{label}: {', '.join(bad)} not a YYYY-MM-DD date"]
    if status in ("doing", "done") and "started" not in card:
        out.append(f"{label}: {status} needs a started date")
    if status == "done" and "finished" not in card:
        out.append(f"{label}: done needs a finished date")
    if status == "deck" and "started" in card:
        out.append(f"{label}: has a started date but is deck")
    if status != "done" and "finished" in card:
        out.append(f"{label}: has a finished date but is {status}")
    if "started" in card and "finished" in card and card["finished"] < card["started"]:
        out.append(f"{label}: finished before it started")
    return out


def problems() -> list[str]:
    """Return a list of human-readable problems. Empty means all good."""
    out: list[str] = []
    try:
        data = load_progress()
    except (ValueError, json.JSONDecodeError) as e:
        return [f"progress.json: {e}"]

    missing = REQUIRED_TOP - data.keys()
    if missing:
        out.append(f"progress.json: missing keys {sorted(missing)}")
    for d in data.get("days", []):
        gaps = REQUIRED_DAY - d.keys()
        if gaps:
            out.append(f"day {d.get('n')}: missing {sorted(gaps)}")
            continue
        bad = [i for i in d["done"] if not (isinstance(i, int) and 0 <= i < len(d["checks"]))]
        if bad:
            out.append(f"day {d['n']}: done indexes out of range {bad}")
        if not (ROOT / d["file"]).exists():
            out.append(f"day {d['n']}: lesson file {d['file']} not found")
        out.extend(card_problems(d, f"day {d['n']}"))
    ids: set[str] = set()
    for t in data.get("tasks", []):
        gaps = REQUIRED_TASK - t.keys()
        if gaps:
            out.append(f"task {t.get('id')}: missing {sorted(gaps)}")
            continue
        if not (isinstance(t["id"], str) and TASK_ID_RE.fullmatch(t["id"])):
            out.append(f"task {t['id']!r}: id must be a lowercase slug (a-z, 0-9, -)")
            continue
        if t["id"] in ids:
            out.append(f"task {t['id']}: id used twice")
        ids.add(t["id"])
        out.extend(card_problems(t, f"task {t['id']}"))
    for dec in data.get("decisions", []):
        if not REQUIRED_DECISION <= dec.keys():
            out.append(f"decision malformed: {dec}")
    for entry in data.get("log", []):
        if not {"date", "entry"} <= entry.keys():
            out.append(f"log entry malformed: {entry}")

    out.extend(check_docs.problems())

    for n in LESSON_NUMBERS:
        if not render(n).strip():
            out.append(f"{lesson_file(n)}: renders to nothing")

    for name in ["record/progress.json", *[f"lessons/{lesson_file(n)}" for n in LESSON_NUMBERS]]:
        if "—" in (ROOT / name).read_text():
            out.append(f"{name}: contains an em-dash")
    return out


if __name__ == "__main__":
    found = problems()
    for p in found:
        print("FAIL", p)
    print("OK" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
