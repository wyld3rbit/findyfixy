"""Validate the tracker's data before committing.

Run from the training root:
    uv run --no-project --with markdown python tools/check.py

Checks:
1. record/progress.json parses as JSON.
2. Every day has the required fields, and every `done` index points at a real check.
3. Every lesson file renders to non-empty HTML.
4. Decisions have all fields; a day's `plan` has a valid status and an existing file.
5. No em-dashes in lessons, plan or progress data.

Exits non-zero on the first class of failure, with a one-line reason per problem.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lessons import LESSON_NUMBERS, ROOT, lesson_file, render  # noqa: E402

REQUIRED_DAY = {"n", "week", "date", "title", "file", "outcome", "checks", "done"}
REQUIRED_DECISION = {"date", "day", "decision", "rejected", "why"}
PLAN_STATUSES = {"draft", "approved", "built", "gated"}
REQUIRED_TOP = {"updated", "plannedStart", "plannedEnd", "countdownStart", "countdownEnd", "log", "days"}


PROGRESS = ROOT / "record" / "progress.json"


def load_progress() -> dict:
    """Parse record/progress.json into a dict."""
    return json.loads(PROGRESS.read_text())


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
        plan = d.get("plan")
        if plan is not None:
            if plan.get("status") not in PLAN_STATUSES:
                out.append(f"day {d['n']}: plan status must be one of {sorted(PLAN_STATUSES)}")
            if not (ROOT / plan.get("file", "")).is_file():
                out.append(f"day {d['n']}: plan file {plan.get('file')} not found")
    for dec in data.get("decisions", []):
        if not REQUIRED_DECISION <= dec.keys():
            out.append(f"decision malformed: {dec}")
    for entry in data.get("log", []):
        if not {"date", "entry"} <= entry.keys():
            out.append(f"log entry malformed: {entry}")

    for n in LESSON_NUMBERS:
        if not render(n).strip():
            out.append(f"{lesson_file(n)}: renders to nothing")

    for name in ["record/progress.json", *[lesson_file(n) for n in LESSON_NUMBERS]]:
        if "—" in (ROOT / name).read_text():
            out.append(f"{name}: contains an em-dash")
    return out


if __name__ == "__main__":
    found = problems()
    for p in found:
        print("FAIL", p)
    print("OK" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
