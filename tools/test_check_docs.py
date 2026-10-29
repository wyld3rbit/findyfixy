"""Tests for tools/check_docs.py.

Each test builds a tiny repository in a temporary folder, so the checker is
proven to fail on the kinds of staleness this project actually had: a deleted
file still named, a missing make target, a volatile test count. One guard test
runs the real repository.

Run from the training root:
    uv run --no-project --with markdown --with pytest pytest tools
"""

from pathlib import Path

import pytest

import check_docs

# The throwaway repositories hold only these two docs.
TMP_DOCS = ("README.md", "CLAUDE.md")

MAKEFILE = """\
.DEFAULT_GOAL := help
.PHONY: help test ui
COLIMA ?= 6
help test ui:
\t@echo hi
eval-gate:
\t@echo gate
%.o: %.c
\tcc
"""


def repo(tmp_path: Path, readme: str, claude: str = "ok") -> Path:
    (tmp_path / "Makefile").write_text(MAKEFILE)
    (tmp_path / "README.md").write_text(readme)
    (tmp_path / "CLAUDE.md").write_text(claude)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "models.md").write_text("m")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "pyproject.toml").write_text("p")
    return tmp_path


def test_make_targets_reads_names_before_the_colon_and_skips_recipes_and_variables() -> None:
    assert check_docs.make_targets(MAKEFILE) == {"help", "test", "ui", "eval-gate"}


def test_a_clean_repo_passes(tmp_path: Path) -> None:
    root = repo(tmp_path, "Run `make test` then `make ui`. See `notes/models.md` and `pyproject.toml`.")
    assert check_docs.problems(root, TMP_DOCS) == []


def test_a_make_target_that_does_not_exist_is_flagged(tmp_path: Path) -> None:
    root = repo(tmp_path, "```bash\nmake setup\n```")
    assert check_docs.problems(root, TMP_DOCS) == ["README.md: `make setup` is not a target in the Makefile"]


def test_prose_that_merely_says_make_is_not_a_command(tmp_path: Path) -> None:
    root = repo(tmp_path, "We make sure to keep it short. make a decision.")
    assert check_docs.problems(root, TMP_DOCS) == []


def test_a_planned_target_is_allowed(tmp_path: Path) -> None:
    root = repo(tmp_path, "Everything starts with `make demo` by Day 10.")
    assert check_docs.problems(root, TMP_DOCS) == []


def test_a_deleted_file_still_named_is_flagged(tmp_path: Path) -> None:
    root = repo(tmp_path, "The record is `progress.js`.")
    assert check_docs.problems(root, TMP_DOCS) == ["README.md: `progress.js` does not exist"]


def test_a_path_in_a_code_block_is_checked_too(tmp_path: Path) -> None:
    root = repo(tmp_path, "```\nuv run python tools/gone.py\n```")
    assert check_docs.problems(root, TMP_DOCS) == ["README.md: `tools/gone.py` does not exist"]


@pytest.mark.parametrize("ref", ["code/.env", "reports/eval-x.json", "dNN.md", "notes/dayN.md", "/code-review", ".md", "https://example.org/a.md", "<file>.md"])
def test_things_that_are_not_checked_paths_are_ignored(tmp_path: Path, ref: str) -> None:
    root = repo(tmp_path, f"See `{ref}`.")
    assert check_docs.problems(root, TMP_DOCS) == []


def test_a_bare_name_is_found_under_code(tmp_path: Path) -> None:
    root = repo(tmp_path, "Edit `pyproject.toml`.")
    assert check_docs.problems(root, TMP_DOCS) == []


@pytest.mark.parametrize("text", ["```\nmake test   # 371 tests\n```", "All 502 tests pass.", "498 passed."])
def test_a_volatile_count_is_flagged_in_prose_and_in_code(tmp_path: Path, text: str) -> None:
    root = repo(tmp_path, text)
    (found,) = check_docs.problems(root, TMP_DOCS)
    assert "goes stale" in found


def test_a_missing_doc_is_reported(tmp_path: Path) -> None:
    root = repo(tmp_path, "x")
    (root / "CLAUDE.md").unlink()
    assert check_docs.problems(root, TMP_DOCS) == ["CLAUDE.md: file is missing"]


def test_the_real_repository_has_no_drift() -> None:
    assert check_docs.problems() == []


def test_every_readme_in_the_repository_is_checked() -> None:
    """A new README must be added to DOCS, so it cannot drift unnoticed."""
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "README.md", "*/README.md"], cwd=check_docs.ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    assert set(tracked) <= set(check_docs.DOCS)
