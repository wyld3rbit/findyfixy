"""Tests for the CI gate (harness/eval/attest.py). Temp trees and fake reports only.

What these tests cover
    The fingerprint (computed from imports, stable, order-independent, a
    declared directory listed again at each check), what `verify` reports
    for a changed, added or removed file, every refusal of
    `write_attestation`, the type checks on a loaded attestation, the
    sanitised log lines, the command line's exit codes when run alone with
    no site-packages, and the `--gate` wiring in harness/eval/cli.py:
    preconditions, and an attestation only after a command that succeeded.

What a staff+ reader should notice
    A console-only change leaves the fingerprint alone. That is the point of
    computing the set from imports: the gate asks for a rerun exactly when a
    file on the eval path changed, and never for a file off it.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from harness.eval import attest
from harness.eval.attest import (
    AttestationRefused,
    FingerprintError,
    Inputs,
    fingerprint,
    import_closure,
    printable,
    verify,
    write_attestation,
)
from harness.eval.cli import AttestationSpec, main
from harness.eval.report import write_baseline
from tests.harness_fakes import NOW, PROV, case, run

ATTEST_SCRIPT = Path(attest.__file__)
ENTRY = ("app.entry",)
EXTRA = ("data/", "uv.lock")
INPUTS = Inputs(ENTRY, EXTRA)
CREATED = datetime(2026, 11, 9, 8, 0, 0, tzinfo=timezone.utc)
STALE = "the database is stale: run make ingest (1 chunk(s) to write, 0 to delete)"

# A small app: an entry module with a top-level, a parent-package and a lazy
# import, a module only the console uses, a harness package with a
# third-party import that must be ignored, and a data directory whose README
# and hidden OS file are not inputs.
SOURCES = {
    "app/__init__.py": "",
    "app/entry.py": "import app.core\nfrom harness import lib\n\ndef late():\n    from app import helper\n",
    "app/core.py": "import json\nPROMPT = 'v1'\n",
    "app/helper.py": "VALUE = 1\n",
    "app/console.py": "import app.core\nPORT = 8775\n",
    "harness/__init__.py": "",
    "harness/lib.py": "import numpy\nfrom collections.abc import Mapping\n",
    "data/golden.jsonl": '{"id": "g1"}\n{"id": "g2"}\n',
    "data/README.md": "What the data is.\n",
    "data/.DS_Store": "finder state\n",
    "uv.lock": "version = 1\n",
}
INCLUDED = [
    "app/__init__.py", "app/core.py", "app/entry.py", "app/helper.py",
    "harness/__init__.py", "harness/lib.py",
]
FILES = sorted([*INCLUDED, "data/golden.jsonl", "uv.lock"])


def make_tree(root: Path) -> Path:
    """Write SOURCES under `root` and return it."""
    for rel, text in SOURCES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A fresh app tree in a temp directory."""
    return make_tree(tmp_path / "code")


def passing(**provenance: Any) -> dict[str, Any]:
    """A minimal report of a full, real, clean run. Keyword arguments override provenance."""
    base = {
        "judge_model": "judge:7b", "judge_digest": "d1", "git_sha": "abc1234", "git_dirty": False,
        "timestamp_utc": "2026-11-09T07:00:00+00:00", "negative_control": False,
        "small_model": "small:4b", "large_model": "large:4b", "threshold": 0.7,
    }
    return {"provenance": {**base, **provenance}, "cases": [{"id": "g1"}, {"id": "g2"}]}


def attest_now(root: Path, report: dict[str, Any] | None = None, problems: list[str] | None = None,
               **kw: Any) -> Path:
    """Write an attestation for `root` from a fake report; return its path."""
    report_file = root / "reports" / "eval-x.json"
    report_file.parent.mkdir(exist_ok=True)
    report_file.write_text("{}\n")
    path = root / "evals" / "attestation.json"
    write_attestation(path, report or passing(), problems or [], fingerprint(root, ENTRY, EXTRA),
                      inputs=INPUTS, report_file=report_file, expected_case_ids={"g1", "g2"},
                      now=lambda: CREATED, **kw)
    return path


def current(root: Path) -> attest.Fingerprint:
    """The fingerprint of the test tree now."""
    return fingerprint(root, ENTRY, EXTRA)


def rewrite(path: Path, **fields: Any) -> None:
    """Replace top-level fields of the attestation at `path`."""
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


# ---- Fingerprint ------------------------------------------------------------------


def test_the_closure_follows_imports_at_any_depth_and_skips_what_is_not_first_party(root: Path) -> None:
    """Parent packages and a lazy import count; the console and numpy do not."""
    assert [p.relative_to(root).as_posix() for p in import_closure(root, ENTRY)] == INCLUDED


def test_the_fingerprint_is_stable_and_order_independent(root: Path) -> None:
    """Same tree, same digest, whatever order the extras are declared in."""
    once, twice = current(root), fingerprint(root, ENTRY, tuple(reversed(EXTRA)))
    assert once == twice and once.digest == current(root).digest
    assert sorted(once.files) == FILES  # the directory's README and hidden file are left out
    assert attest.digest_of(dict(reversed(list(once.files.items())))) == once.digest


def test_changing_an_included_file_changes_the_digest_and_verify_names_it(root: Path) -> None:
    """One edited byte in a module on the eval path fails verification, by path."""
    path = attest_now(root)
    assert verify(path, current(root)) == []
    (root / "app/core.py").write_text("import json\nPROMPT = 'v2'\n")
    assert verify(path, current(root)) == ["changed: app/core.py"]


def test_a_declared_directory_catches_a_changed_and_a_new_file(root: Path) -> None:
    """Extras count like code, and a file added to `data/` is covered without being listed."""
    path = attest_now(root)
    (root / "data/golden.jsonl").write_text('{"id": "g1"}\n')
    (root / "data/new.json").write_text("{}\n")
    (root / "data/README.md").write_text("Edited docs.\n")
    assert verify(path, current(root)) == ["changed: data/golden.jsonl", "added: data/new.json"]


def test_a_console_only_change_needs_no_new_run(root: Path) -> None:
    """A module off the eval path can change freely."""
    path = attest_now(root)
    (root / "app/console.py").write_text("import app.core\nPORT = 9999\n")
    assert verify(path, current(root)) == []


def test_added_and_removed_files_are_reported(root: Path) -> None:
    """A new import adds a file to the set; dropping one removes it."""
    path = attest_now(root)
    (root / "app/extra.py").write_text("")
    (root / "app/entry.py").write_text("import app.core\nimport app.extra\nfrom harness import lib\n")
    assert verify(path, current(root)) == ["changed: app/entry.py", "added: app/extra.py", "removed: app/helper.py"]


def test_a_relative_import_a_missing_module_and_a_bad_extra_are_errors(root: Path) -> None:
    """Each would silently shrink the set, so each raises instead."""
    (root / "app/core.py").write_text("from . import helper\n")
    with pytest.raises(FingerprintError, match="relative import"):
        current(root)
    (root / "app/core.py").write_text("import app.gone\n")
    with pytest.raises(FingerprintError, match="app.gone"):
        current(root)
    (root / "app/core.py").write_text(SOURCES["app/core.py"])
    for bad in ("../outside.txt", "/etc/hosts", "data/missing.jsonl", "gone/", "data/\x1b[2J"):
        with pytest.raises(FingerprintError):
            fingerprint(root, ENTRY, (bad,))
    for bad in ("numpy", "", "app.x/../../y"):
        with pytest.raises(FingerprintError):
            fingerprint(root, (bad,), EXTRA)


def test_a_symlink_out_of_the_root_is_refused(root: Path, tmp_path: Path) -> None:
    """A fork could link a module to a file elsewhere on the runner."""
    (tmp_path / "secret.py").write_text("X = 1\n")
    (root / "app/helper.py").unlink()
    (root / "app/helper.py").symlink_to(tmp_path / "secret.py")
    with pytest.raises(FingerprintError, match="outside the root"):
        current(root)


# ---- Writing ----------------------------------------------------------------------


def test_the_attestation_holds_relative_paths_hashes_and_the_run_identity(root: Path) -> None:
    """The committed file names no absolute path and keeps only identifying run fields."""
    doc = json.loads(attest_now(root).read_text())
    assert doc["verdict"] == "pass" and doc["fingerprint"] == current(root).digest
    assert doc["files"] == dict(current(root).files) and doc["report"]["name"] == "eval-x.json"
    assert doc["inputs"] == {"entry_modules": list(ENTRY), "extra_files": list(EXTRA)}
    assert doc["run"] == {"judge_model": "judge:7b", "judge_digest": "d1", "git_sha": "abc1234",
                          "timestamp_utc": "2026-11-09T07:00:00+00:00",
                          "small_model": "small:4b", "large_model": "large:4b"}
    assert doc["created_utc"] == "2026-11-09T08:00:00+00:00"
    assert str(root) not in json.dumps(doc)


@pytest.mark.parametrize(
    ("report", "problems", "reason"),
    [
        (passing(), ["faithfulness: regression"], "the gate did not pass"),
        (passing(negative_control=True), [], "negative control"),
        ({**passing(), "cases": [{"id": "g1"}]}, [], "not a full run: 1 of 2"),
        (passing(git_dirty=True), [], "uncommitted changes"),
        (passing(git_dirty=None), [], "git could not tell"),
        (passing(judge_digest="unknown"), [], "judge_digest is unknown"),
        (passing(git_sha="unknown"), [], "git_sha is unknown"),
    ],
)
def test_write_refuses_every_run_that_cannot_stand_for_the_files(
    root: Path, report: dict[str, Any], problems: list[str], reason: str
) -> None:
    """Nothing is written for a failed, control, subset, dirty or unidentifiable run."""
    with pytest.raises(AttestationRefused, match=reason):
        attest_now(root, report, problems)
    assert not (root / "evals" / "attestation.json").exists()


def test_files_edited_during_the_run_are_refused_and_an_older_attestation_is_kept(root: Path) -> None:
    """The fingerprint at the start must equal the one at the end."""
    path = attest_now(root)
    before, kept = current(root), path.read_bytes()
    (root / "app/core.py").write_text("PROMPT = 'edited mid-run'\n")
    with pytest.raises(AttestationRefused, match="changed during the run: changed: app/core.py"):
        attest_now(root, started=before)
    assert path.read_bytes() == kept


# ---- Verifying --------------------------------------------------------------------


def test_verify_reports_a_missing_malformed_or_failed_attestation(root: Path) -> None:
    """Each is a problem with a message, never a traceback."""
    path = root / "evals" / "attestation.json"
    assert verify(path, current(root))[0].startswith("no attestation at ")
    path.parent.mkdir(exist_ok=True)
    path.write_text("{not json")
    assert "is not valid JSON" in verify(path, current(root))[0]
    path.write_text("[]")
    assert "must hold a JSON object" in verify(path, current(root))[0]
    path = attest_now(root)
    rewrite(path, verdict="fail")
    assert verify(path, current(root)) == ["the attested verdict is 'fail', not 'pass'"]
    rewrite(path, verdict="pass", fingerprint="0" * 64)
    assert verify(path, current(root)) == ["the recorded fingerprint does not match its own file hashes"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True), ("version", "1"), ("verdict", ["pass"]), ("fingerprint", 7),
        ("created_utc", None), ("files", ["app/core.py"]), ("files", {"app/core.py": 1}),
        ("inputs", {"entry_modules": "app.entry", "extra_files": []}), ("report", {"name": "x"}),
        ("run", ["abc"]), ("run", {"git_sha": 1234}),
    ],
)
def test_every_field_read_is_type_checked(root: Path, field: str, value: Any) -> None:
    """A wrong type is a message naming the field, from the library and the command line (exit 2)."""
    path = attest_now(root)
    rewrite(path, **{field: value})
    assert f"'{field}' must be" in verify(path, current(root))[0]
    result = cli_run(root)
    assert result.returncode == 2 and f"'{field}' must be" in result.stderr and "Traceback" not in result.stderr


def test_printable_drops_control_characters_breaks_workflow_commands_and_caps_length() -> None:
    """One line out, whatever comes in. Pure."""
    assert printable("a\n::error::x\r\x1b[31m") == "a: :error: :x[31m"
    assert len(printable("y" * 1000)) == attest.MAX_LINE and printable("y" * 1000).endswith("...")


def test_no_attested_value_can_start_a_workflow_command_in_the_ci_log(root: Path) -> None:
    """Crafted paths and run fields reach the log as plain text on lines that start with `attest:`."""
    path = attest_now(root)
    doc = json.loads(path.read_text())
    rewrite(path, files={**doc["files"], "x\n::stop-commands::tok": "0" * 64})
    failed = cli_run(root)
    rewrite(path, files=doc["files"], run={**doc["run"], "git_sha": "abc\n::error file=a::owned",
                                           "timestamp_utc": "\n::add-mask::x"})
    passed = cli_run(root, "--list")
    assert failed.returncode == 1 and passed.returncode == 0
    for out in (failed.stdout, passed.stdout):
        assert "::" not in out
        assert all(line.startswith("attest:") for line in out.splitlines())


def cli_run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """The command line as CI runs it: this one file, -I, and no site-packages at all (-S)."""
    return subprocess.run(
        [sys.executable, "-I", "-S", str(ATTEST_SCRIPT), "verify", "--root", str(root),
         "--attestation", "evals/attestation.json", *args],
        capture_output=True, text=True, timeout=60,
    )


def test_the_command_line_runs_alone_and_exits_0_1_or_2(root: Path, tmp_path: Path) -> None:
    """1 without an attestation or after a change, 0 when it matches, 2 when inputs are unreadable."""
    missing = cli_run(root)
    assert missing.returncode == 1 and "no attestation at" in missing.stdout and "make eval-gate" in missing.stdout
    attest_now(root)
    ok = cli_run(root, "--list")
    assert ok.returncode == 0 and "attest: pass. 8 answer-affecting files" in ok.stdout
    assert "attest:   app/core.py" in ok.stdout
    (root / "app/core.py").write_text("PROMPT = 'v2'\n")
    changed = cli_run(root)
    assert changed.returncode == 1 and "attest: FAIL: changed: app/core.py" in changed.stdout
    (root / "uv.lock").unlink()
    assert cli_run(root).returncode == 2
    (root / "evals" / "attestation.json").write_text("{broken")
    assert cli_run(root).returncode == 2
    assert cli_run(tmp_path / "nowhere").returncode == 2


# ---- The --gate wiring in cli.py --------------------------------------------------


def gate_cli(root: Path, report: dict[str, Any], argv: list[str], *,
             checks: list[list[str]] | None = None, **kw: Any) -> int:
    """`main` with a fixed run and an attestation spec over the tree.

    `checks` are the precondition results, one per call, in order (default:
    always in sync). The run records that it ran in `report["_ran"]`.
    """
    answers = iter(checks or [])
    spec = AttestationSpec(path=root / "evals" / "attestation.json", root=root, inputs=INPUTS,
                           case_ids=lambda: ["g1", "g2"], preconditions=lambda: next(answers, []))

    def fixed(progress, **_):
        """Mark the run and return a copy of the report."""
        report["_ran"] = True
        return copy.deepcopy({k: v for k, v in report.items() if k != "_ran"})

    return main(argv, prog="python -m some.app", run=fixed, reports_dir=root / "reports",
                baseline_path=root / "evals" / "baseline.json", display_root=root, attestation=spec,
                now=lambda: NOW, **kw)


def fake_report(**provenance: Any) -> dict[str, Any]:
    """A real two-case report from the harness fakes, from a clean tree."""
    prov = {**PROV, "judge_model": "j", "negative_control": False, **provenance}
    return run([case("g1"), case("g2")], provenance=prov)


@pytest.fixture
def baseline(root: Path) -> Path:
    """A baseline the fake report gates clean against."""
    path = root / "evals" / "baseline.json"
    write_baseline(fake_report(), path)
    return path


def test_a_passing_gate_writes_the_attestation_and_says_to_commit_it(root: Path, baseline: Path, capsys) -> None:
    """The printed path is relative, and the attestation verifies against the tree."""
    assert gate_cli(root, fake_report(), ["--gate"]) == 0
    out = capsys.readouterr().out
    assert "gate: pass" in out and "attestation written: evals/attestation.json (8 files)" in out
    assert "commit it" in out
    assert verify(root / "evals" / "attestation.json", current(root)) == []


def test_a_failing_gate_or_a_refused_run_writes_nothing_and_keeps_the_gate_exit_code(
    root: Path, baseline: Path, capsys
) -> None:
    """A fail writes nothing (exit 1); a dirty-tree pass is refused but still exits 0."""
    worse = fake_report()
    worse["aggregates"]["overall"]["n_run_errors"] = 1
    worse["cases"][0]["run_error"] = "boom"
    assert gate_cli(root, worse, ["--gate"]) == 1
    assert "attestation not written: the command failed (exit 1)" in capsys.readouterr().out
    assert gate_cli(root, fake_report(git_dirty=True), ["--gate"]) == 0
    assert "attestation not written: the working tree had uncommitted changes" in capsys.readouterr().out
    assert not (root / "evals" / "attestation.json").exists()


def test_a_refused_baseline_in_the_same_command_writes_no_attestation(root: Path, baseline: Path, capsys) -> None:
    """`--gate --write-baseline`: the gate passes, the baseline is refused, so the command fails and nothing is attested."""
    # The app's legacy control key marks this run a control for the baseline only.
    assert gate_cli(root, fake_report(), ["--gate", "--write-baseline"], legacy_control={"git_sha": "abc1234"}) == 1
    captured = capsys.readouterr()
    assert "gate: pass" in captured.out and "baseline refused" in captured.err
    assert "attestation not written: the command failed (exit 1)" in captured.out
    assert not (root / "evals" / "attestation.json").exists()


def test_a_stale_database_stops_the_gate_before_the_run(root: Path, baseline: Path, capsys) -> None:
    """The precondition is checked first: exit 2, the reason, no case run, no report, no attestation."""
    report = fake_report()
    assert gate_cli(root, report, ["--gate"], checks=[[STALE]]) == 2
    err = capsys.readouterr().err
    assert f"error: {STALE}" in err and "not running" in err
    assert "_ran" not in report and not (root / "reports").exists()
    assert not (root / "evals" / "attestation.json").exists()


def test_a_database_that_went_stale_during_the_run_is_not_attested(root: Path, baseline: Path, capsys) -> None:
    """Checked again before writing: in sync at the start, stale at the end, nothing written."""
    assert gate_cli(root, fake_report(), ["--gate"], checks=[[], [STALE]]) == 0
    assert f"attestation not written: {STALE}" in capsys.readouterr().out
    assert not (root / "evals" / "attestation.json").exists()


def test_a_run_without_gate_writes_no_attestation_and_a_broken_tree_fails_before_the_run(
    root: Path, baseline: Path, capsys
) -> None:
    """No --gate, no attestation and no precondition. An unparseable module exits 2 before any case runs."""
    assert gate_cli(root, fake_report(), [], checks=[[STALE]]) == 0
    assert not (root / "evals" / "attestation.json").exists()
    report = fake_report()
    (root / "app/core.py").write_text("def (:\n")
    assert gate_cli(root, report, ["--gate"]) == 2
    assert "cannot fingerprint" in capsys.readouterr().err
    assert "_ran" not in report  # the run never started
