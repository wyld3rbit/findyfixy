"""Run events: the one vocabulary the graph uses to report what it is doing.

What it does
    Defines `Event` (one thing that happened), `Emit` (how code reports it),
    and `make_emit` (builds an `Emit` that stamps each event with a time and
    hands it to a sink). The live console (Day 2) turns these into the
    animation and the event log.

Why code reports through a callback, and not through logging
    The graph and `ask_validated` should not know who is listening. They get
    an `Emit` argument, the same dependency injection as `ask`. A test passes
    a list's `append` and asserts on the events. The console passes a queue.
    Day 6 can pass an exporter that forwards to Langfuse. No change to the
    graph.

Why the clock is injected
    `make_emit` takes a `clock`. Production uses `time.perf_counter`, a
    monotonic clock, so a system clock change cannot reorder or skew events.
    Tests pass a fake counter, so timestamps are exact and nothing is flaky.
    `t` is seconds since the emitter was built, i.e. since the run started.

Failure modes and what changes at scale
    - A slow sink slows the run, because `emit` is a plain call. The console
      sink only does `queue.put_nowait`, so it never blocks. A network
      exporter must buffer and drop, not block, or tracing can take prod down.
    - Events carry prompts and raw replies. In this lab the data is synthetic.
      With real data, redact before emitting (Day 7).
    - Nothing here is a stable wire format yet. Treat `kind` names as
      internal until a consumer outside this repo depends on them.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

EventKind = Literal[
    "run_started",
    "node_entered",
    "model_request",
    "model_reply",
    "fault_injected",
    "validation",
    "retry",
    "node_exited",
    "route",
    "run_finished",
    "run_failed",
    "hunt_started",
    "hunt_run",
    "hunt_finished",
]


@dataclass(frozen=True)
class Event:
    """One thing that happened during a run.

    `t` is seconds since the run started. `data` holds the details for this
    `kind` (model name, attempt number, error text, and so on). It is a plain
    dict of JSON-safe values, so the console can send it as is.
    """

    kind: EventKind
    t: float
    data: dict[str, Any] = field(default_factory=dict)


class Emit(Protocol):
    """How code reports an event: `emit("route", target="large")`.

    A Protocol, not a bare `Callable`, so the call shape is typed: a kind
    first, then keyword details.
    """

    def __call__(self, kind: EventKind, /, **data: Any) -> None: ...


def no_emit(kind: EventKind, /, **data: Any) -> None:
    """The default `Emit`: report nothing. Lets callers skip the argument."""


def make_emit(sink: Callable[[Event], None], clock: Callable[[], float]) -> Emit:
    """Build an `Emit` that stamps events with elapsed time and sends them on.

    The start time is read once, here. Call this at the start of a run, not
    at import time, so `t` begins at 0 for each run.
    """
    start = clock()

    def emit(kind: EventKind, /, **data: Any) -> None:
        sink(Event(kind=kind, t=clock() - start, data=data))

    return emit
