"""Tracing for entry points: one place that turns settings into a running tracer provider.

What it holds
    `open_tracing` starts the process's provider from settings and closes it
    at the end; every entry point gets its provider here (the console, the
    MCP server, the eval session). `command_trace` wraps a command in one
    trace on top of it.

Why a module of its own, next to `app.wiring`
    Every entry point that touches a model, the database, a tool or the
    network is traced. Some of those entry
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

from harness.trace.tracing import EXPORT_TIMEOUT_S

if TYPE_CHECKING:
    from harness.trace.command import CommandTrace
    from harness.trace.tracing import Tracing


@contextmanager
def open_tracing(*, export_timeout_s: float = EXPORT_TIMEOUT_S) -> "Iterator[Tracing | None]":
    """The process's tracer provider from settings, or None when tracing is off; closed at the end.

    It loads `code/.env` first, so every entry point honours the same tracing
    settings. Closing flushes what is still queued, bounded by the export
    timeout: a process that starts and stops often (the MCP server in
    per-call mode) passes a short one.
    """
    from app.settings import load_env, read_tracing
    from harness.trace.tracing import Tracing as _Tracing

    load_env()  # TRACING, TRACE_USER and the keys may be set in code/.env
    config = read_tracing()
    if config is None:
        yield None
        return
    tracing = _Tracing.start(config, export_timeout_s=export_timeout_s)
    try:
        yield tracing
    finally:
        tracing.close()


@contextmanager
def command_trace(name: str, *, input: object = None,
                  attributes: "Mapping[str, object] | None" = None) -> "Iterator[CommandTrace]":
    """One command as one trace, on a provider of its own. A no-op `CommandTrace` with tracing off."""
    from harness.trace.command import CommandTrace

    with open_tracing() as tracing, CommandTrace(tracing, name, input=input, attributes=attributes) as command:
        yield command
