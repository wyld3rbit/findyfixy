"""Eval events: what the harness reports while a run is in flight, for any viewer.

What it does
    Defines the harness's own small event vocabulary (`EvalEventKind`), the
    callback type the runner takes (`EvalEmit`), its no-op default, and
    `score_summary`, the compact overall result that `eval_finished` carries.

Why a vocabulary of its own, and not the app's run events
    The harness never imports an application (tests/test_architecture.py), so
    it cannot reuse an app's event types. It also should not: an app's events
    describe its own steps (a model call, a route), and the harness must work
    for an app that has none of them. A viewer that shows both, such as the
    FindyFixy console, tags each event with where it came from.

Who emits what
    `run_eval` (runner.py) emits `eval_started`, then in the answer pass
    `case_started` and `case_answered` for every case, then in the judge pass
    one `metric_scored` per metric as each finishes and `case_scored`, case
    by case. A case with a run error has nothing to judge: its `case_scored`
    follows its `case_answered` at once, in the answer pass. It never emits
    `eval_finished` or `eval_failed`: those belong to whoever drives the run and
    decides the gate. The runner never sees a baseline, so it cannot say
    whether the run passed.

Payloads
    Every value is JSON-safe (str, int, float, bool, None, list, dict), so a
    viewer can forward an event as it is. Labels and run errors carry model
    output and exception text: untrusted, so a viewer must render them as text.

        eval_started   n, cases (ids in run order), system (the app's describe())
        case_started   id, group, index (1-based position), n
        case_answered  id, labels (answered), expected, run_error (str or None)
        metric_scored  id, name, score (float or None), failed, error
        case_scored    id
        eval_finished  summary (score_summary), gate (problems, or None when
                       not applied), subset (True when fewer cases ran than
                       the baseline was defined on)
        eval_failed    error_type, error, reason ("stopped" when the driver
                       asked the runner to stop, else "error")

Ordering and threads
    With `workers` 1, events arrive in run order: every case is answered
    before any is judged. With more, each case's own events keep their order
    and the passes still do not overlap, but cases interleave within a pass, and the callback is called
    from several threads: it must be thread-safe. A slow callback slows the
    run, because it is a plain call. A callback that raises stops the run like
    any other bug, so keep it to an append.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol

EvalEventKind = Literal[
    "eval_started",
    "case_started",
    "case_answered",
    "metric_scored",
    "case_scored",
    "eval_finished",
    "eval_failed",
]


class EvalEmit(Protocol):
    """How the harness reports an event: `on_event("case_scored", id="g01")`.

    A Protocol, not a bare `Callable`, so the call shape is typed: a kind
    first, then keyword details.
    """

    def __call__(self, kind: EvalEventKind, /, **data: Any) -> None:
        """Report one event. Must not raise and must return quickly (see the module notes)."""
        ...


def no_eval_emit(kind: EvalEventKind, /, **data: Any) -> None:
    """The default `EvalEmit`: report nothing. A command-line run needs no viewer."""


def score_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The overall aggregate of a report, cut down to what a viewer shows. Pure.

    Mean, 95% interval and failed count per score, plus the case and run-error
    counts. Groups and per-case detail stay in the report, which a viewer can
    fetch another way if it needs them.
    """
    overall = report["aggregates"]["overall"]
    return {
        "n": overall["n"],
        "n_run_errors": overall["n_run_errors"],
        "scores": {
            name: {"mean": stat["mean"], "ci": stat["ci"], "n_failed": stat["n_failed"]}
            for name, stat in overall["scores"].items()
        },
    }
