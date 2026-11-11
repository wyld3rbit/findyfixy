"""Eval runs in the console: the harness runner, bridged into the live event stream.

What it does
    The console's "Eval run" mode runs the first N golden cases through the
    harness (wyld3rness/eval/runner.py) with FindyFixy plugged in, and streams two
    kinds of events side by side: the app's own run events (findyfixy/events.py)
    from the graph inside each case, and the harness's events
    (wyld3rness/eval/events.py) from the runner. `HarnessEvent` carries the second
    kind, so the stream can tag each event with its source.

    `live_eval` builds the real run from the same session factory as the
    command line (`findyfixy.evals.answers.open_live_session`), so the viewer and
    `make eval` can never run different wiring. An `EvalSuite` pairs a run with
    how many cases it has, which the composition root reads from the case set
    itself, so no case count is written down twice. `run_and_judge` drives any
    run, applies the gate when the full set ran, and always ends with exactly
    one terminal event.

The console is a viewer, not a record
    It never writes a report or a baseline. A run here leaves no file behind;
    `make eval` and `make eval-gate` are the runs that count. The gate is
    applied read-only against `evals/baseline.json`, and only when every
    golden case ran: the baseline's means and intervals are defined on the
    full set, so comparing a subset to them would pass or fail on sampling,
    not on quality. A subset run reports its scores and says "subset, gate
    not applied".

Stopping
    POST /stop sets the hub's stop flag, and the runner asks for it before
    each case. The case in flight finishes, then the run ends with
    `eval_failed`, reason "stopped", and no verdict: a run cut short at a
    moment someone chose is not a subset anyone designed.

Failure modes
    - Ollama, the database or the judge down: the run raises inside the
      harness or while opening the session. `run_and_judge` turns any
      exception into `eval_failed`, so a viewer never waits forever.
    - A live run borrows the console's one `SharedRetrieval` (findyfixy/wiring.py),
      the same database connection, embedder and reranker its triage runs use.
      A second copy (about 1.3 GB) would not fit beside two answering models
      and a judge on a 16 GB machine.
    - Cases run one at a time (`workers` 1), so events arrive in order and the
      page can follow one case at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from findyfixy.events import Emit
from wyld3rness.eval import cli
from wyld3rness.eval.cli import Session
from wyld3rness.eval.events import EvalEmit, EvalEventKind, score_summary
from wyld3rness.eval.report import gate, load_baseline
from wyld3rness.eval.runner import EvalStopped
from wyld3rness.eval.stats import Report

# Two cases: enough to watch the loop turn over, short enough for a live demo.
EVAL_DEFAULT = 2


@dataclass(frozen=True)
class HarnessEvent:
    """One harness event on its way to the stream. The app's `Event` has the same shape.

    A separate type, and not a widened `Event.kind`, so the app's event
    vocabulary stays exactly what the graph can emit. The console tags each
    event with its source when it serialises it, and stamps it with the same
    emitter code as the app's events (`findyfixy.events.make_stamped_emit`).
    """

    kind: EvalEventKind
    t: float
    data: dict[str, Any]


@dataclass(frozen=True)
class EvalResult:
    """What one eval run produced: the report, and whether only part of the set ran."""

    report: Report
    subset: bool


class EvalRun(Protocol):
    """One eval run of the first `n` cases. App events go to `emit`, harness events to `on_event`."""

    def __call__(self, n: int, *, emit: Emit, on_event: EvalEmit, should_stop: Callable[[], bool]) -> EvalResult:
        """Run the cases and return the report. Writes no file. Raises `EvalStopped` on a stop."""
        ...


@dataclass(frozen=True)
class EvalSuite:
    """A run and the number of cases it can take: a request may ask for 1 to `size`."""

    run: EvalRun
    size: int


class LiveSessionFactory(Protocol):
    """`open_live_session`'s shape: the control flag, and where the graph's events go."""

    def __call__(self, negative_control: bool, *, emit: Emit) -> AbstractContextManager[Session]:
        """Open the app's resources and yield the session; close them on exit."""
        ...


def live_eval(
    open_session: LiveSessionFactory,
    *,
    golden_path: Path,
    repo_dir: Path,
    progress: Callable[[str], None],
    size: int,
) -> EvalSuite:
    """The real eval run, through the harness's own `live_runner`. Builds nothing until called.

    `live_runner` stamps provenance exactly as the command line does (judge
    digest, prompt versions, golden hash, git), which is what makes a full run
    comparable with the baseline. The only change is a wrapper that keeps the
    first `n` cases of the session. `size` is the golden set's length, read by
    the caller when it wires the console.
    """

    def run(n: int, *, emit: Emit, on_event: EvalEmit, should_stop: Callable[[], bool]) -> EvalResult:
        """Open the live session with this run's emitter, keep `n` cases, run them."""
        total = 0

        @contextmanager
        def first_n(negative_control: bool) -> Iterator[Session]:
            """The live session, cut to its first `n` cases. Closing is the factory's job."""
            nonlocal total
            with open_session(negative_control, emit=emit) as session:
                total = len(session.cases)
                yield replace(session, cases=list(session.cases)[:n])

        runner = cli.live_runner(first_n, golden_path=golden_path, repo_dir=repo_dir)
        report = runner(progress, negative_control=False, workers=1, on_event=on_event, should_stop=should_stop)
        return EvalResult(report=report, subset=len(report["cases"]) < total)

    return EvalSuite(run=run, size=size)


def verdict(result: EvalResult, baseline_path: Path) -> list[str] | None:
    """The gate's problems for a full run (empty means pass), or None for a subset.

    Read-only: the baseline is loaded, never written. A missing baseline raises
    `BaselineMissing`, which `run_and_judge` reports as a failed run.
    """
    if result.subset:
        return None
    return gate(result.report, load_baseline(baseline_path))


def run_and_judge(
    run: EvalRun,
    n: int,
    *,
    emit: Emit,
    on_event: EvalEmit,
    should_stop: Callable[[], bool],
    baseline_path: Path,
) -> None:
    """Drive one eval run to exactly one terminal event: `eval_finished` or `eval_failed`.

    A stop ends with `eval_failed` and reason "stopped". The catch is broad on
    purpose. This runs on the console's run thread, and whatever goes wrong (a
    model server down, a missing baseline, a bug) the viewer must hear about it
    instead of waiting on a stream that never ends.
    """
    try:
        result = run(n, emit=emit, on_event=on_event, should_stop=should_stop)
        problems = verdict(result, baseline_path)
    except EvalStopped as err:
        on_event("eval_failed", error_type="EvalStopped", error=str(err), reason="stopped")
        return
    except Exception as err:
        on_event("eval_failed", error_type=type(err).__name__, error=str(err), reason="error")
        return
    on_event("eval_finished", summary=score_summary(result.report), gate=problems, subset=result.subset)
