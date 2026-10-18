"""Validate the tracker's data before committing.

Run from the training root:
    uv run --no-project --with markdown python tools/check.py

Checks:
1. progress.js parses as JSON after the `window.DATA = ` prefix.
2. Every day has the required fields, and every `done` index points at a real check.
3. lessons.js matches what tools/build_lessons.py would generate (no stale lessons).
4. No em-dashes in lessons, plan or progress data.

Exits non-zero on the first class of failure, with a one-line reason per problem.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lessons import ROOT, build  # noqa: E402

REQUIRED_DAY = {"n", "week", "date", "title", "file", "outcome", "checks", "done"}
REQUIRED_TOP = {"updated", "plannedStart", "plannedEnd", "countdownStart", "countdownEnd", "log", "days"}


def load_progress() -> dict:
    """Parse progress.js into a dict."""
    raw = (ROOT / "progress.js").read_text().strip()
    prefix = "window.DATA = "
    if not raw.startswith(prefix):
        raise ValueError("progress.js must start with 'window.DATA = '")
    return json.loads(raw[len(prefix):].rstrip(";"))


def problems() -> list[str]:
    """Return a list of human-readable problems. Empty means all good."""
    out: list[str] = []
    try:
        data = load_progress()
    except (ValueError, json.JSONDecodeError) as e:
        return [f"progress.js: {e}"]

    missing = REQUIRED_TOP - data.keys()
    if missing:
        out.append(f"progress.js: missing keys {sorted(missing)}")
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
    for entry in data.get("log", []):
        if not {"date", "entry"} <= entry.keys():
            out.append(f"log entry malformed: {entry}")

    if (ROOT / "lessons.js").read_text() != build():
        out.append("lessons.js is stale: run tools/build_lessons.py")

    for name in ["00-plan.md", "progress.js", *[f"d{n:02d}.md" for n in range(1, 11)]]:
        if "—" in (ROOT / name).read_text():
            out.append(f"{name}: contains an em-dash")
    return out


if __name__ == "__main__":
    found = problems()
    for p in found:
        print("FAIL", p)
    print("OK" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
