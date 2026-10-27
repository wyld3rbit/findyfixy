"""Documentation drift check: the docs may not name things that do not exist.

Why: a doc that names a deleted file, a renamed command or a stale count is
worse than no doc, because the reader trusts it. These errors are mechanical,
so a machine finds them. Judgment calls (is the prose still true?) stay with
the person who changes the code; this only catches what can be proven wrong.

Checked in README.md and CLAUDE.md (code/CHANGELOG.md is history, and the
lessons are checked separately when they are revised):
1. Every `make <target>` named in code is a target in the root Makefile.
2. Every repository path named in code exists, except the few that are not in
   git on purpose (see ALLOWED_MISSING).
3. No count that changes, such as "502 tests", is written in prose. It would be
   wrong within a day. Point at the command that prints it instead.

Run through tools/check.py (`make check`). Standard library only.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "CLAUDE.md")

# Paths a doc may name that are absent from a fresh clone on purpose: secrets
# the user creates, and generated output that is git-ignored.
ALLOWED_MISSING = ("code/.env", "reports/", "code/reports/", "code/evals/baseline.json")

# Targets the docs name before they exist, on purpose. Remove an entry when the
# target is built; the check then covers it like any other.
PLANNED_TARGETS = ("demo",)

_SPAN = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"```.*?\n(.*?)```", re.S)
_MAKE = re.compile(r"\bmake\s+([a-z][a-z0-9-]*)")
_FILE_SUFFIXES = (
    ".md", ".py", ".js", ".css", ".txt", ".json", ".jsonl", ".html", ".toml", ".lock", ".yml", ".yaml", ".sh",
)
# Placeholders and things that are not repository paths: templates (dNN.md,
# dayN.md), slash commands (/code-review), URLs, and a bare suffix (.md).
_PLACEHOLDER = re.compile(r"[<>*$~{}=]|^https?:|^\.\.|^/|NN|dayN")
_COUNT = re.compile(r"\b\d{2,}\s+(?:tests|passed)\b")


def make_targets(makefile_text: str) -> set[str]:
    """Targets defined in a Makefile, including several names before one colon."""
    found: set[str] = set()
    for line in makefile_text.splitlines():
        if not line or line[0] in "\t# " or ":=" in line or "?=" in line:
            continue
        head, sep, _ = line.partition(":")
        if not sep:
            continue
        for name in head.split():
            if not name.startswith((".", "%", "$")):
                found.add(name)
    return found


def code_text(markdown: str) -> list[str]:
    """The inline code spans and fenced code blocks of a document."""
    blocks = _FENCE.findall(markdown)
    without_fences = _FENCE.sub("", markdown)
    return blocks + _SPAN.findall(without_fences)


def path_refs(snippets: list[str]) -> set[str]:
    """Tokens in code that look like repository paths."""
    refs = set()
    for snippet in snippets:
        for token in snippet.split():
            token = token.strip(",;:)(").rstrip(".")
            if not token or _PLACEHOLDER.search(token) or token in _FILE_SUFFIXES:
                continue
            if "/" in token or token.endswith(_FILE_SUFFIXES):
                refs.add(token)
    return refs


def problems(root: Path = ROOT, docs: tuple[str, ...] = DOCS) -> list[str]:
    """Human-readable problems across the checked docs. Empty means all good."""
    out: list[str] = []
    makefile = root / "Makefile"
    targets = make_targets(makefile.read_text()) if makefile.is_file() else set()
    for name in docs:
        path = root / name
        if not path.is_file():
            out.append(f"{name}: file is missing")
            continue
        text = path.read_text()
        snippets = code_text(text)
        for snippet in snippets:
            for target in _MAKE.findall(snippet):
                if target not in targets and target not in PLANNED_TARGETS:
                    out.append(f"{name}: `make {target}` is not a target in the Makefile")
        for ref in sorted(path_refs(snippets)):
            if ref.startswith(ALLOWED_MISSING) or ref.endswith(ALLOWED_MISSING):
                continue
            # A bare file name such as pyproject.toml lives under code/.
            if not ((root / ref).exists() or ("/" not in ref and (root / "code" / ref).exists())):
                out.append(f"{name}: `{ref}` does not exist")
        for match in _COUNT.finditer(text):
            out.append(f"{name}: '{match.group(0)}' is a count that goes stale; name the command instead")
    return sorted(set(out))


if __name__ == "__main__":
    found = problems()
    for line in found:
        print("FAIL", line)
    print("OK" if not found else f"{len(found)} problem(s)")
    raise SystemExit(1 if found else 0)
