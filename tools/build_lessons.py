"""Regenerate lessons.js from 00-plan.md and d01.md to d10.md.

Why: the tracker is opened from disk (file://), where browsers block reading
other files with fetch. So lesson text is compiled into a script the page can
load with a plain <script> tag.

Run from the training root:
    uv run --no-project --with markdown python tools/build_lessons.py

The Self-check section of each lesson is dropped, because the tracker renders
the checks itself from progress.js.
"""

import json
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]


def render(n: int) -> str:
    """Return the HTML for lesson n (0 is the plan)."""
    name = "00-plan.md" if n == 0 else f"d{n:02d}.md"
    text = (ROOT / name).read_text()
    text = re.sub(r"^# .*\n", "", text, count=1)
    text = re.sub(r"\n## Self-check\n.*?(?=\n## |\Z)", "\n", text, flags=re.S)
    return markdown.markdown(text, extensions=["fenced_code", "tables"])


def build() -> str:
    """Return the full contents of lessons.js."""
    parts = [
        "// Generated from 00-plan.md and d01.md to d10.md. Edit those, then regenerate.",
        "window.LESSONS = {};",
    ]
    for n in range(0, 11):
        lines = ",\n  ".join(json.dumps(line, ensure_ascii=False) for line in render(n).split("\n"))
        parts.append(f'window.LESSONS[{n}] = [\n  {lines}\n].join("\\n");')
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    out = ROOT / "lessons.js"
    out.write_text(build())
    print(f"wrote {out.name} ({out.stat().st_size} bytes)")
