"""The command line every app's eval entry point hands its wiring to.

What it does
    run (default)       run every case, write `reports/eval-<UTC>.json`, print the table
    --gate              also compare to the baseline; exit 1 on any problem
    --write-baseline    freeze this run as the baseline (refused on a failed gate)
    --workers N         run N cases at once (1 to MAX_WORKERS)
    a control flag      the negative control: a run the app builds to fail. Its
                        name is the app's (FindyFixy: `--no-recall`).
    --show-case ID      print one case of a saved report (`--report` picks the file)
    --baseline-from F   freeze a saved report as the baseline, without running

The attestation (harness/eval/attest.py)
    When the app passes an `AttestationSpec`, a `--gate` command that ends
    with exit 0 on a full, real run from a clean tree also writes the
    attestation CI checks. Anything that fails the command (the gate, a
    refused baseline) writes none. The fingerprint is taken before the run
    and again after it, so a file edited during an hour-long run cannot be
    attested as if it had run. The app's preconditions (FindyFixy: the
    database holds the current corpus) are checked before the run, which
    then does not start, and again before writing. A refused attestation is
    printed with every reason and does not change the exit code: CI fails on
    the missing attestation instead.

Why the app passes everything in
    The harness cannot know where an app keeps its golden set, its reports or
    its baseline, or how to build its system. `main` takes those as
    arguments, and `live_runner` takes a `SessionFactory`: a context manager the
    app writes, which opens its resources (database, models), yields a
    `Session`, and closes them when the run ends. So the app owns its
    lifecycle and the harness owns the policy.

Exit codes: 0 pass, 1 gate problems or baseline refused, 2 baseline missing,
a bad report file, a missing setting (`ConfigError`), answer-affecting files
that cannot be fingerprinted, or an unmet attestation precondition.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Collection, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Protocol

from harness.eval.attest import (
    NO_LEGACY_CONTROL,
    AttestationRefused,
    Fingerprint,
    FingerprintError,
    Inputs,
    LegacyControl,
    file_sha256,
    fingerprint,
    utc_now,
    write_attestation,
)
from harness.eval.case import Case
from harness.eval.events import EvalEmit, no_eval_emit
from harness.eval.judge import Judge, make_ollama_judge
from harness.eval.metrics import Embed
from harness.eval.provenance import build_provenance, git_info, ollama_digest
from harness.eval.report import (
    BaselineMissing,
    BaselineRefused,
    format_table,
    freeze_baseline,
    gate,
    load_baseline,
    show_case,
    write_baseline,
    write_report,
)
from harness.eval.runner import MAX_WORKERS, run_eval
from harness.eval.stats import Report
from harness.eval.sut import SystemUnderTest


class ConfigError(Exception):
    """A setting the app needs is missing. The message says what to set.

    The app raises it (translating its own config error), the CLI prints it and
    exits 2. A KeyError from a bug elsewhere is not caught: it keeps its traceback.
    """


@dataclass(frozen=True)
class Session:
    """Everything a live run needs from the app, valid while its context is open.

    `judge_model` is a name, not a judge: the harness builds the judge and
    fingerprints the same model, so the stamp and the scorer cannot disagree.
    """

    sut: SystemUnderTest
    cases: Sequence[Case]
    judge_model: str
    embed: Embed


@dataclass(frozen=True)
class AttestationSpec:
    """What a passing gate attests and where it writes it. The app declares it once.

    `root` is the folder that holds the app's packages; every attested path
    is relative to it. `case_ids` returns the ids of the full golden set, so
    a run of fewer cases is refused. `preconditions` returns every reason the
    app's live state does not match its files (empty means it does), for
    state no file hash can show, such as a database loaded from an older
    corpus. Both are functions: reading the golden set and the database is
    the app's job, and the harness calls them only around a gate run.
    """

    path: Path
    root: Path
    inputs: Inputs
    case_ids: Callable[[], Collection[str]]
    preconditions: Callable[[], Sequence[str]] = lambda: ()


# negative_control -> a context manager that yields the open session.
SessionFactory = Callable[[bool], AbstractContextManager[Session]]


class RunFn(Protocol):
    """One evaluation run: progress sink in, report out. `main` calls it once.

    `on_event` receives the live events (harness/eval/events.py) and
    `should_stop` is asked before each case (runner.EvalStopped). The command
    line passes neither; a viewer such as an app's console passes both.
    """

    def __call__(
        self,
        progress: Callable[[str], None],
        *,
        negative_control: bool,
        workers: int,
        on_event: EvalEmit = no_eval_emit,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> Report:
        """Run every case and return the report. No file is written here."""
        ...


def live_runner(
    open_session: SessionFactory,
    *,
    golden_path: Path,
    repo_dir: Path,
    make_judge: Callable[[str], Judge] = make_ollama_judge,
    digest: Callable[[str], str] = ollama_digest,
    git: Callable[[Path], tuple[str, bool | None]] = git_info,
    now: Callable[[], datetime] = utc_now,
) -> RunFn:
    """Compose a live run from the app's session factory. Builds nothing until called.

    The provenance is stamped after the session opens, so any warm-up the app
    does there lands before the timestamp, as it did before the harness split.
    """

    def run(
        progress: Callable[[str], None],
        *,
        negative_control: bool,
        workers: int,
        on_event: EvalEmit = no_eval_emit,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> Report:
        """Open the session, stamp provenance, run every case, close the session."""
        with open_session(negative_control) as session:
            provenance = build_provenance(
                judge_model=session.judge_model,
                digest=digest,
                golden_sha256=file_sha256(golden_path),
                describe=session.sut.describe(),
                git=lambda: git(repo_dir),
                now=now,
                negative_control=negative_control,
            )
            return run_eval(
                session.cases,
                sut=session.sut,
                judge=make_judge(session.judge_model),
                embed=session.embed,
                provenance=provenance,
                progress=progress,
                workers=workers,
                on_event=on_event,
                should_stop=should_stop,
            )

    return run


def _parser(prog: str, control_flag: str, control_help: str) -> argparse.ArgumentParser:
    """The argument parser. The control flag's name and help are the app's words."""
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--gate", action="store_true", help="compare to the baseline, exit 1 on problems")
    parser.add_argument("--write-baseline", action="store_true", help="freeze this run as the baseline")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=range(1, MAX_WORKERS + 1),
        metavar=f"1..{MAX_WORKERS}",
        help="cases to run in parallel. Set OLLAMA_NUM_PARALLEL on the server to match.",
    )
    parser.add_argument(control_flag, dest="negative_control", action="store_true", help=control_help)
    parser.add_argument("--show-case", metavar="ID", help="print one case from a saved report, with every judge decision")
    parser.add_argument("--report", type=Path, help="the saved report for --show-case (default: the newest)")
    parser.add_argument(
        "--baseline-from",
        metavar="REPORT",
        type=Path,
        help="freeze a saved report as the baseline, without running anything",
    )
    return parser


def main(
    argv: Sequence[str] | None,
    *,
    prog: str,
    run: RunFn,
    reports_dir: Path,
    baseline_path: Path,
    display_root: Path | None = None,
    control_flag: str = "--negative-control",
    control_help: str = "negative control: a run built to fail the gate. Never saved as a baseline.",
    legacy_control: LegacyControl = NO_LEGACY_CONTROL,
    attestation: AttestationSpec | None = None,
    now: Callable[[], datetime] = utc_now,
) -> int:
    """Run the eval, print the table and the report path. Returns the exit code.

    `run`, `now` and the paths are injected so tests drive this without a
    model. Progress goes to stderr. `display_root` shortens the printed report
    path (relative to it when the report lies under it). `legacy_control` is
    the app's older control flag in saved reports (see attest.LegacyControl).
    `attestation`, when given, is written after a passing `--gate` run (see
    the module docstring); without it no attestation is ever written.
    """
    args = _parser(prog, control_flag, control_help).parse_args(argv)

    if args.show_case:
        return show_case(args.show_case, args.report, reports_dir)
    if args.baseline_from:
        return freeze_baseline(args.baseline_from, baseline_path, legacy_control)

    baseline = None
    if args.gate:  # fail on a missing baseline before spending an hour on the run
        try:
            baseline = load_baseline(baseline_path, command=prog)
        except BaselineMissing as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
    armed: tuple[AttestationSpec, Fingerprint] | None = None
    if baseline is not None and attestation is not None:
        # Fingerprint and preconditions first: a broken import, a missing
        # declared file or a stale database fails in seconds, not after the run.
        try:
            armed = (attestation, _fingerprint(attestation))
        except FingerprintError as err:
            print(f"error: cannot fingerprint the answer-affecting files: {err}", file=sys.stderr)
            return 2
        blockers = attestation.preconditions()
        if blockers:
            for blocker in blockers:
                print(f"error: {blocker}", file=sys.stderr)
            print("error: not running: this gate run could not be attested", file=sys.stderr)
            return 2

    try:
        report = run(
            lambda msg: print(msg, file=sys.stderr, flush=True),
            negative_control=args.negative_control,
            workers=args.workers,
        )
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    path = write_report(report, now(), reports_dir)
    print(format_table(report))
    print(f"report: {_shown(path, display_root)}")
    timing = report.get("timing")
    if timing:
        print(
            f"timing: {timing['wall_s']} s wall, {timing['workers']} worker(s), "
            f"{timing['cases_per_min']} cases/min"
        )

    code = 0
    if baseline is not None:
        problems = gate(report, baseline)
        for problem in problems:
            print(f"GATE FAIL: {problem}")
        print("gate: FAIL" if problems else "gate: pass")
        code = 1 if problems else 0
    if args.write_baseline and code:
        # The gate just failed. Writing this run as the baseline would turn the
        # regression into the new standard, and the next gate would pass.
        print("baseline not written: the gate failed in this same run", file=sys.stderr)
    elif args.write_baseline:
        try:
            write_baseline(report, baseline_path, legacy_control=legacy_control)
            print(f"baseline written: {baseline_path.name}")
        except BaselineRefused as err:
            print(f"baseline refused: {err}", file=sys.stderr)
            code = 1
    if armed is not None:
        # Last on purpose: only a command that succeeded as a whole attests,
        # and a run that also rewrote the baseline changed an attested file.
        _attest(*armed, report=report, report_file=path, code=code, display_root=display_root)
    return code


def _shown(path: Path, display_root: Path | None) -> Path:
    """`path` relative to `display_root` when it lies under it, else unchanged."""
    return path.relative_to(display_root) if display_root and path.is_relative_to(display_root) else path


def _fingerprint(spec: AttestationSpec) -> Fingerprint:
    """The fingerprint of the spec's inputs. Raises `FingerprintError`."""
    return fingerprint(spec.root, spec.inputs.entry_modules, spec.inputs.extra_files)


def _attest(
    spec: AttestationSpec,
    started: Fingerprint,
    *,
    report: Report,
    report_file: Path,
    code: int,
    display_root: Path | None,
) -> None:
    """Write the attestation after a command that succeeded, or say why not. Never raises a refusal.

    A failed command writes nothing: the previous attestation, if any, still
    describes the files it was made over, and CI compares against those.
    Reaching here with `code == 0` means the gate passed, so the gate's
    problems are empty by construction.
    """
    if code:
        print(f"attestation not written: the command failed (exit {code})")
        return
    blockers = spec.preconditions()  # again: the state may have moved during the run
    if blockers:
        print(f"attestation not written: {'; '.join(blockers)}")
        return
    try:
        finished = _fingerprint(spec)
        write_attestation(
            spec.path,
            report,
            (),
            finished,
            inputs=spec.inputs,
            report_file=report_file,
            expected_case_ids=spec.case_ids(),
            started=started,
        )
    except (AttestationRefused, FingerprintError) as err:
        print(f"attestation not written: {err}")
        return
    print(f"attestation written: {_shown(spec.path, display_root)} ({len(finished.files)} files)")
    print("commit it with this change: CI fails a change to these files without a matching attestation")
