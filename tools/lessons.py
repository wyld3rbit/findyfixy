"""Render a lesson's markdown to HTML for the tracker.

Why: the lessons are the .md files in lessons/ and nothing else. The server (tools/serve.py)
renders one on request, so no generated copy exists to go stale.

Needs the `markdown` package, which is not a project dependency. Run tools with:
    uv run --no-project --with markdown python tools/serve.py

The Self-check section of each lesson is dropped, because the tracker renders
the checks itself from record/progress.json.
"""

import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]

LESSON_NUMBERS = range(0, 11)
LESSONS_DIR = ROOT / "lessons"


def lesson_file(n: int) -> str:
    """Name of the markdown file for lesson n (0 is the plan)."""
    if n not in LESSON_NUMBERS:
        raise ValueError(f"no lesson {n}")
    return "00-plan.md" if n == 0 else f"d{n:02d}.md"


def render(n: int) -> str:
    """Return the HTML for lesson n. Raises ValueError for an unknown number."""
    text = (LESSONS_DIR / lesson_file(n)).read_text()
    text = re.sub(r"^# .*\n", "", text, count=1)
    text = re.sub(r"\n## Self-check\n.*?(?=\n## |\Z)", "\n", text, flags=re.S)
    return markdown.markdown(text, extensions=["fenced_code", "tables"])
