"""List and render the architecture decision records for the tracker.

Why: the records live in code/docs/adr/ as markdown, one per phase. The tracker
shows each one under its day, rendered on request like the lessons, so no
generated copy exists to go stale.

Which day a record belongs to is read from the record itself: its "Phase:" line
("Phase: Day 5, rebuilt." belongs to day 5). The diagram is served from the
records' own img/ folder; other relative links (to lessons, to other records)
point at files the browser cannot open from this server, so they are shown as
plain text.
"""

import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
ADR_DIR = ROOT / "code" / "docs" / "adr"
IMG_DIR = ADR_DIR / "img"

_FILE = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")
_TITLE = re.compile(r"^# ADR \d{4}: (.+)$", re.M)
_PHASE = re.compile(r"^Phase: Day (\d+)\b", re.M)
_IMG = re.compile(r'src="img/(\d{4}-architecture\.svg)"')
_LOCAL_LINK = re.compile(r'<a href="(?!https?://)[^"]*">(.*?)</a>', re.S)


def records(adr_dir: Path = ADR_DIR) -> dict[int, Path]:
    """Every record by number. Files that do not look like `NNNN-slug.md` are ignored."""
    found = {}
    for path in sorted(adr_dir.glob("*.md")):
        match = _FILE.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def index(adr_dir: Path = ADR_DIR) -> list[dict[str, object]]:
    """Number, title and day of each record, in number order. A record without a Phase line has day None."""
    out = []
    for number, path in records(adr_dir).items():
        text = path.read_text(encoding="utf-8")
        title, phase = _TITLE.search(text), _PHASE.search(text)
        out.append({"n": number, "title": title.group(1) if title else path.stem,
                    "day": int(phase.group(1)) if phase else None})
    return out


def render(number: int, adr_dir: Path = ADR_DIR) -> str:
    """The HTML for record `number`. Raises KeyError for an unknown number."""
    text = records(adr_dir)[number].read_text(encoding="utf-8")
    text = re.sub(r"^# .*\n", "", text, count=1)  # the tracker shows the title itself
    html = markdown.markdown(text, extensions=["fenced_code", "tables"])
    html = _IMG.sub(r'src="/api/adr-img/\1"', html)
    return _LOCAL_LINK.sub(r"\1", html)


def image_path(name: str, img_dir: Path = IMG_DIR) -> Path | None:
    """The diagram file `name`, or None unless it is a `NNNN-architecture.svg` that exists.

    The name must match exactly, so no path, `..` or other file type can be asked for.
    """
    if not re.fullmatch(r"\d{4}-architecture\.svg", name):
        return None
    path = img_dir / name
    return path if path.is_file() else None
