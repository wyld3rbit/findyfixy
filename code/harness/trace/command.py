"""One command as one trace: a top span for the whole command and a span per step inside it.

What it does
    `CommandTrace(tracing, name)` is a context manager. Inside it the
    command's top span is current, so every span made in the block (a step,
    a retriever stage, a judge call, an app run) lands in the same trace.
    `step(name)` times one piece of work; `output(value)` records what the
    command produced. With `tracing=None` every method is a no-op and nothing
    from OpenTelemetry is imported, so a command reads the same either way.

Why one helper for every command
    "Everything the project does is observable" (Day 6, step 5) means every
    entry point that touches a model, the database, a tool or the network
    makes a trace. Without one shape for it, each command would start a
    provider, name its spans and close the exporter its own way, and one of
    them would forget to flush. The app's composition root builds the
    provider (app/tracing.py `command_trace`); this class is the shape.

Failure modes
    An exception in the block ends the top span as an error and is re-raised:
    a failed ingest is a red trace, not a missing one. A step that raises is
    marked the same way.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

from harness.trace.spans import Attributes, SpanTree, attribute_value, mark_failed

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

    from harness.trace.tracing import Tracing


class Step:
    """A step in progress. `set` adds what was learned (counts, sizes, tokens)."""

    def __init__(self, span: Span | None) -> None:
        """Hold the span, or None when tracing is off."""
        self._span = span

    def set(self, **attributes: Any) -> None:
        """Add attributes; None values are left out (unknown is not zero)."""
        if self._span is not None:
            self._span.set_attributes({k: attribute_value(v) for k, v in attributes.items() if v is not None})


class CommandTrace:
    """A command's trace. Use it as a context manager around the whole command."""

    def __init__(
        self,
        tracing: Tracing | None,
        name: str,
        *,
        input: Any = None,  # noqa: A002 - the backend's word for it
        attributes: Attributes | None = None,
    ) -> None:
        """Hold the settings. The top span starts on `__enter__`."""
        self._tracing = tracing
        self._name = name
        self._input = input
        self._attributes = attributes
        self._tree: SpanTree | None = None
        self._stack = ExitStack()
        self._output: Any = None

    @property
    def tracer(self) -> Tracer | None:
        """The tracer, for code that builds its own spans (an app's adapter), or None."""
        return self._tracing.tracer() if self._tracing is not None else None

    @property
    def tracing(self) -> Tracing | None:
        """The provider, or None."""
        return self._tracing

    def __enter__(self) -> CommandTrace:
        """Start the top span and make it current."""
        if self._tracing is not None:
            self._tree = SpanTree(self._tracing.tracer(), self._name, input=self._input,
                                  attributes=self._attributes)
            self._stack.enter_context(self._tree.activate())
        return self

    def __exit__(self, kind: Any, error: BaseException | None, tb: Any) -> None:
        """End the top span: an error when the block raised, else with the output."""
        self._stack.close()
        if self._tree is not None:
            failure = f"{type(error).__name__}: {error}" if error is not None else None
            self._tree.finish(output=self._output if failure is None else None, error=failure)

    def set(self, **attributes: Any) -> None:
        """Add attributes to the top span (a status code, a count). A no-op without tracing."""
        if self._tree is not None:
            self._tree.set(attributes)

    def output(self, value: Any) -> None:
        """Record what the command produced: the trace's output."""
        self._output = value

    def step(self, name: str, attributes: Mapping[str, Any] | None = None) -> AbstractContextManager[Step]:
        """A span for one piece of work, current while it runs. A no-op without tracing."""
        if self._tracing is None:
            return _no_step()
        return _step(self._tracing.tracer(), name, attributes)


@contextmanager
def _no_step() -> Iterator[Step]:
    """The step when tracing is off."""
    yield Step(None)


@contextmanager
def _step(tracer: Tracer, name: str, attributes: Mapping[str, Any] | None) -> Iterator[Step]:
    """A current span for one step; an exception marks it failed and is re-raised."""
    with tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        step = Step(span)
        step.set(**dict(attributes or {}))
        try:
            yield step
        except BaseException as err:
            mark_failed(span, f"{type(err).__name__}: {err}")
            raise
