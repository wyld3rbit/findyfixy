"""Tests for the harness command line (wyld3rness/eval/cli.py). Fake runs and sessions only.

What these tests cover
    Exit codes, the gate and baseline flags, the app-named control flag,
    `--workers`, `ConfigError`, the case viewer and freezing a saved report
    through `main`, and `live_runner`: that it opens the app's session, stamps
    provenance from the app's `describe()`, and closes the session.

What a staff+ reader should notice
    A failed gate in the same run never rewrites the baseline. That rule is
    what stops a regression from quietly becoming the new standard.
"""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import pytest

from wyld3rness.eval.cli import ConfigError, Session, live_runner, main
from wyld3rness.eval.report import load_baseline, write_baseline
from wyld3rness.testing import NOW, PROV, FakeJudge, FakeSUT, case, embed, run


def report_with_errors(errors: int) -> dict[str, Any]:
    """A real two-case report from the fakes; with `errors`, the second case raises."""
    return run([case("g1"), case("g2")], provenance={**PROV, "git_dirty": False},
               sut=FakeSUT(raise_on=["g2"] if errors else []))


def fixed(report: dict[str, Any]):
    """A run function that returns a copy of `report` and records its keyword arguments."""
    seen: list[dict[str, Any]] = []

    def go(progress, **kwargs):
        """Record the keyword arguments and return a copy of the report."""
        seen.append(kwargs)
        return copy.deepcopy(report)

    go.seen = seen  # type: ignore[attr-defined]
    return go


def cli(argv, tmp_path: Path, run_fn=None, **kw) -> int:
    """`main` with the fixed test paths, clock and program name."""
    return main(argv, prog="python -m some.app", run=run_fn or fixed(report_with_errors(0)),
                reports_dir=tmp_path / "r", baseline_path=tmp_path / "evals" / "baseline.json",
                now=lambda: NOW, **kw)


def test_write_baseline_then_gate_passes(tmp_path: Path, capsys) -> None:
    """A frozen run gates clean against itself."""
    assert cli(["--write-baseline"], tmp_path) == 0
    assert (tmp_path / "evals" / "baseline.json").exists()
    assert cli(["--gate"], tmp_path) == 0
    assert "gate: pass" in capsys.readouterr().out


def test_write_baseline_refuses_when_the_run_had_errors(tmp_path: Path, capsys) -> None:
    """Run errors make the numbers untrustworthy, so no baseline is written."""
    assert cli(["--write-baseline"], tmp_path, fixed(report_with_errors(1))) == 1
    assert not (tmp_path / "evals" / "baseline.json").exists()
    assert "baseline refused" in capsys.readouterr().err


def test_gate_exits_2_without_a_baseline_and_1_on_problems(tmp_path: Path, capsys) -> None:
    """A missing baseline is caught before the run; gate problems exit 1."""
    bad = fixed(report_with_errors(1))
    assert cli(["--gate"], tmp_path, bad) == 2
    assert "no baseline at baseline.json. Run `python -m some.app --write-baseline`" in capsys.readouterr().err
    assert bad.seen == []  # nothing ran: the missing baseline is caught first
    write_baseline(report_with_errors(0), tmp_path / "evals" / "baseline.json")
    assert cli(["--gate"], tmp_path, bad) == 1
    assert "GATE FAIL" in capsys.readouterr().out


def test_a_failed_gate_never_rewrites_the_baseline(tmp_path: Path, capsys) -> None:
    """A regression must never become the new standard in the same run."""
    baseline = tmp_path / "evals" / "baseline.json"
    write_baseline(report_with_errors(0), baseline)
    before = baseline.read_bytes()
    worse = run([case("g1", kind="y")], provenance={**PROV, "git_dirty": False})  # the fake answers x
    assert cli(["--gate", "--write-baseline"], tmp_path, fixed(worse)) == 1
    assert baseline.read_bytes() == before
    assert "gate failed in this same run" in capsys.readouterr().err


def test_the_control_flag_and_workers_reach_the_run(tmp_path: Path) -> None:
    """Flags arrive as keyword arguments; the app can rename the control flag."""
    go = fixed(report_with_errors(0))
    cli([], tmp_path, go)
    cli(["--negative-control"], tmp_path, go)
    cli(["--no-recall", "--workers", "3"], tmp_path, go, control_flag="--no-recall")
    assert go.seen == [{"negative_control": False, "workers": 1}, {"negative_control": True, "workers": 1},
                       {"negative_control": True, "workers": 3}]
    with pytest.raises(SystemExit):
        cli(["--workers", "9"], tmp_path, go)
    with pytest.raises(SystemExit):  # the app renamed the flag, so the default name is gone
        cli(["--negative-control"], tmp_path, go, control_flag="--no-recall")


def test_a_missing_setting_exits_2_and_a_bug_keeps_its_traceback(tmp_path: Path, capsys) -> None:
    """Only `ConfigError` is turned into a message; anything else propagates."""
    def missing(progress, **kwargs):
        """A run whose app setting is missing."""
        raise ConfigError("set the environment variable SMALL_MODEL")

    assert cli([], tmp_path, missing) == 2
    assert "error: set the environment variable SMALL_MODEL" in capsys.readouterr().err

    def bug(progress, **kwargs):
        """A run with a bug in it."""
        raise KeyError("some_record_key")

    with pytest.raises(KeyError):
        cli([], tmp_path, bug)


def test_the_report_path_is_printed_relative_to_the_display_root(tmp_path: Path, capsys) -> None:
    """The printed path is short, and the timing line follows it."""
    cli([], tmp_path, display_root=tmp_path)
    out = capsys.readouterr().out
    assert "report: r/eval-20261026T123005Z.json" in out
    assert "timing: " in out and "worker(s)" in out


def test_show_case_and_baseline_from_run_nothing(tmp_path: Path, capsys) -> None:
    """Viewing and freezing a saved report never start a run."""
    def never(progress, **kwargs):
        """A run that must not be called."""
        raise AssertionError("viewing or freezing a saved report must not run the evaluation")

    saved = tmp_path / "r" / "eval-20261026T120000Z.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps(report_with_errors(0)))
    assert cli(["--show-case", "g1"], tmp_path, never) == 0
    assert "ALERT      question g1" in capsys.readouterr().out
    assert cli(["--baseline-from", str(saved)], tmp_path, never) == 0
    assert load_baseline(tmp_path / "evals" / "baseline.json")["scores"]["kind_match"]["mean"] == 1.0
    newer = tmp_path / "r" / "eval-20261026T130000Z.json"
    newer.write_text(saved.read_text())
    capsys.readouterr()
    assert cli(["--compare", str(saved)], tmp_path, never) == 0  # against the newest
    assert "compare: identical" in capsys.readouterr().out
    assert cli(["--compare", str(newer)], tmp_path, never) == 2  # the newest against itself proves nothing


def test_a_run_prints_the_time_of_each_pass(tmp_path: Path, capsys) -> None:
    """The timing line names wall time and both passes, so a slow pass is visible at a glance."""
    assert cli([], tmp_path) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("timing:"))
    assert "s answering" in line and "s judging" in line


# ---- live_runner ------------------------------------------------------------


def session_factory(log: list[str]):
    """A session factory that records opening and closing, like an app's database."""

    @contextmanager
    def open_session(negative_control: bool):
        """Log opening and closing around one fake session."""
        log.append(f"open negative_control={negative_control}")
        try:
            yield Session(sut=FakeSUT(), cases=[case("g1"), case("g2", "b")], judge_model="judge-m", embed=embed)
        finally:
            log.append("close")

    return open_session


def live(log: list[str], tmp_path: Path, judges: list[str] | None = None, **kw):
    """A `live_runner` over a recording session with a fake judge, digest, git and clock."""
    golden = tmp_path / "golden.jsonl"
    golden.write_text("fixed bytes\n")

    def make_judge(model: str) -> FakeJudge:
        """A fake judge, recording which model it was built for."""
        if judges is not None:
            judges.append(model)
        return FakeJudge()

    return live_runner(session_factory(log), golden_path=golden, repo_dir=tmp_path, make_judge=make_judge,
                       digest=lambda m: "dig-" + m, git=lambda repo: (f"sha-of-{repo.name}", False),
                       now=lambda: NOW, **kw)


def test_live_runner_stamps_the_apps_settings_and_closes_the_session(tmp_path: Path) -> None:
    """Provenance carries the app's settings, and the session is closed after the run."""
    log: list[str] = []
    judges: list[str] = []
    report = live(log, tmp_path, judges)(lambda _m: None, negative_control=True, workers=2)
    prov = report["provenance"]
    assert log == ["open negative_control=True", "close"]
    assert judges == ["judge-m"]  # one judge, for the model the stamp names
    assert prov["judge_model"] == "judge-m" and prov["judge_digest"] == "dig-judge-m"
    assert prov["fake_model"] == "m1" and prov["fake_threshold"] == 0.5
    assert prov["negative_control"] is True
    assert prov["git_sha"] == f"sha-of-{tmp_path.name}"
    assert len(prov["golden_sha256"]) == 64
    assert report["timing"]["workers"] == 2 and len(report["cases"]) == 2


def test_live_runner_closes_the_session_when_the_run_fails(tmp_path: Path) -> None:
    """A failing run still closes the app's resources."""
    log: list[str] = []
    with pytest.raises(ValueError, match="workers"):
        live(log, tmp_path)(lambda _m: None, negative_control=False, workers=9)
    assert log == ["open negative_control=False", "close"]


def test_a_control_run_through_main_can_never_become_a_baseline(tmp_path: Path, capsys) -> None:
    """A control run built by the live path is refused as a baseline."""
    code = cli(["--negative-control", "--write-baseline"], tmp_path, live([], tmp_path))
    assert code == 1 and not (tmp_path / "evals" / "baseline.json").exists()
    assert "negative control" in capsys.readouterr().err
