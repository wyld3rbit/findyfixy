"""Tracing for commands: one place that turns settings into a running tracer for a command.

Why a module of its own, next to `app.wiring`
    Every entry point that touches a model, the database, a tool or the
    network runs inside `command_trace` (Day 6, step 5). Some of those entry
    points are also imported by the composition root (`app.graph` holds the
    triage command; `app.wiring` builds its retriever). Putting this in
    `app.wiring` made an import cycle. This module imports only settings and
    the harness, so any entry point can use it. It is composition, like
    wiring: no domain module imports it (tests/test_architecture.py).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.trace.command import CommandTrace
    from harness.trace.tracing import Tracing


@contextmanager
def command_trace(name: str, *, input: object = None, attributes: "Mapping[str, object] | None" = None,
                  tracing: "Tracing | None" = None) -> "Iterator[CommandTrace]":
    """One command as one trace, with the provider built from settings and closed at the end.

    Every entry point that touches a model, the database, a tool or the
    network runs inside one (Day 6, step 5). Without tracing configured it
    yields a no-op `CommandTrace`. A caller that already runs a provider (the
    console) passes it as `tracing`, and this does not close it. It loads
    `code/.env` first, so every command honours the same tracing settings.
    """
    from app.settings import load_env, read_tracing
    from harness.trace.command import CommandTrace
    from harness.trace.tracing import Tracing as _Tracing

    owned = tracing is None
    if tracing is None:
        load_env()  # TRACING, TRACE_USER and the keys may be set in code/.env
        config = read_tracing()
        tracing = _Tracing.start(config) if config else None
    try:
        with CommandTrace(tracing, name, input=input, attributes=attributes) as command:
            yield command
    finally:
        if owned and tracing is not None:
            tracing.close()
