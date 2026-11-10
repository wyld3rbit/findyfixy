"""The spans of one run, opened and closed as the run's events arrive.

What it does
    `SpanTree` starts a root span for a run. `open(key, ...)` starts a child of
    the innermost span still open, `close(key, ...)` ends it, `event(...)` marks
    a moment on the innermost open span, and `finish(...)` ends whatever is
    still open, then the root. An app's adapter calls these from its event
    stream; it never touches OpenTelemetry itself.

Why parents come from a stack, with one exception
    An app reports its steps in order: a node starts, a model call inside it
    starts and ends, a tool call starts and ends, the node ends. The innermost
    open span is therefore the parent of the next one, so the adapter usually
    passes no parent. The exception is siblings that are open at the same
    time, such as two tool calls asked for in one model reply: both start
    before either ends. `open(..., under="node")` makes each a child of the
    named span instead of the one before it. This holds for one run on one
    thread. Two runs at once need two trees, which is why each run builds its own.

Failure modes
    - A step that raises leaves its spans open. `finish(error=...)` ends them
      innermost first and marks the root as failed, so a failed run still
      shows where it stopped.
    - `close` for a key that is not open is ignored: an app event without its
      opening event (a viewer that joined late) must not break the run. A
      span closed while its own children are still open ends those first;
      an open sibling stays open.
    - Values are cut to `MAX_VALUE_CHARS`. Prompts and replies can be long,
      and a backend may refuse an oversized span; the cut is marked.

Attribute names
    Model calls use the OpenTelemetry GenAI semantic conventions
    (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), so any OTLP
    backend reads them. Those conventions are still marked experimental and
    names have moved between releases (`gen_ai.system` became
    `gen_ai.provider.name`), so they are defined once, here. The `langfuse.*`
    names are the backend's own: the observation type that makes Langfuse
    render a span as a generation or a tool, and the input and output it
    shows. Another backend ignores them.

What changes at scale
    Prompts and replies are user data. Here the data is synthetic; with real
    data, redact before a value reaches `open` or `close`, and sample (keep
    every error, a share of the rest) before the exporter, not after.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

# OpenTelemetry GenAI semantic conventions.
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_PROVIDER = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# Langfuse's own attributes, read from OTLP spans.
LF_TYPE = "langfuse.observation.type"
LF_INPUT = "langfuse.observation.input"
LF_OUTPUT = "langfuse.observation.output"
LF_LEVEL = "langfuse.observation.level"
LF_STATUS = "langfuse.observation.status_message"
LF_TRACE_NAME = "langfuse.trace.name"
LF_TRACE_INPUT = "langfuse.trace.input"
LF_TRACE_OUTPUT = "langfuse.trace.output"

# What a span stands for, in Langfuse's vocabulary. "generation" is a model
# call (shown with its model, tokens and latency), "tool" a tool call,
# "retriever" a lookup, "span" anything else.
SpanKind = Literal["span", "generation", "tool", "retriever"]

MAX_VALUE_CHARS = 20_000
Attributes = Mapping[str, str | int | float | bool | None]


def as_text(value: Any) -> str:
    """A value as span text: strings as they are, anything else as JSON. Cut and marked when long."""
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > MAX_VALUE_CHARS:
        return text[:MAX_VALUE_CHARS] + f"... [cut, {len(text)} chars]"
    return text


def _clean(attributes: Attributes | None) -> dict[str, str | int | float | bool]:
    """Drop None values: OpenTelemetry attributes cannot be null, and unknown is not zero."""
    return {k: v for k, v in (attributes or {}).items() if v is not None}


@dataclass(frozen=True)
class _Open:
    """One open span: the key `close` finds it by, the span, and its parent (None: the root)."""

    key: str
    span: Span
    parent: Span | None


class SpanTree:
    """The spans of one run. Build one per run; see the module notes."""

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        *,
        attributes: Attributes | None = None,
        input: Any = None,  # noqa: A002 - the backend's word for it
    ) -> None:
        """Start the root span now. `name` also names the trace in the backend."""
        # Resolved once per run, not per span: the run's thread makes every span.
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        self._in_context = trace.set_span_in_context
        self._error_status = lambda text: Status(StatusCode.ERROR, text)
        root_attrs = {**_clean(attributes), LF_TRACE_NAME: name}
        if input is not None:
            # Both: the trace's input (the list view) and the root span's (the tree).
            root_attrs[LF_TRACE_INPUT] = root_attrs[LF_INPUT] = as_text(input)
        self._tracer = tracer
        self._root = tracer.start_span(name, attributes=root_attrs)
        # Innermost last. A list, not a dict: order is the parent rule.
        self._open: list[_Open] = []

    @property
    def trace_id(self) -> str:
        """The trace's id as 32 hex characters: what a backend's URL and score API use."""
        return format(self._root.get_span_context().trace_id, "032x")

    def _find(self, key: str) -> int | None:
        """The index of the innermost open span named `key`, or None."""
        return next((i for i in range(len(self._open) - 1, -1, -1) if self._open[i].key == key), None)

    def _parent(self, under: str | None = None) -> Span:
        """The span named `under` (or, without it, the innermost open span), else the root."""
        if under is not None:
            index = self._find(under)
            return self._open[index].span if index is not None else self._root
        return self._open[-1].span if self._open else self._root

    def _inside(self, entry: _Open, ancestor: Span) -> bool:
        """True when `entry` was opened inside `ancestor`, at any depth."""
        parents = {e.span: e.parent for e in self._open}
        parent = entry.parent
        while parent is not None:
            if parent is ancestor:
                return True
            parent = parents.get(parent)
        return False

    def open(
        self,
        key: str,
        name: str,
        *,
        kind: SpanKind = "span",
        attributes: Attributes | None = None,
        input: Any = None,  # noqa: A002
        under: str | None = None,
    ) -> None:
        """Start a child of the innermost open span, or of the open span named `under`.

        `key` is what `close` names it by. Several open spans may share a key;
        `close` ends the innermost one.
        """
        attrs = {**_clean(attributes), LF_TYPE: kind}
        if input is not None:
            attrs[LF_INPUT] = as_text(input)
        parent = self._parent(under)
        span = self._tracer.start_span(name, context=self._in_context(parent), attributes=attrs)
        self._open.append(_Open(key, span, None if parent is self._root else parent))

    def close(
        self,
        key: str,
        *,
        attributes: Attributes | None = None,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        """End the innermost open span named `key`, ending the spans opened inside it first."""
        index = self._find(key)
        if index is None:
            return
        target = self._open[index]
        for entry in reversed([e for e in self._open if self._inside(e, target.span)]):
            self._end(entry.span)
            self._open.remove(entry)
        self._end(target.span, attributes=attributes, output=output, error=error)
        self._open.remove(target)

    def open_keys(self) -> list[str]:
        """The keys of the spans still open, outermost first. For an adapter's bookkeeping."""
        return [e.key for e in self._open]

    def event(self, name: str, attributes: Attributes | None = None) -> None:
        """Mark a moment (a validation result, a retry, a route) on the innermost open span."""
        self._parent().add_event(name, attributes=_clean(attributes))

    def set(self, attributes: Attributes) -> None:
        """Add attributes to the root span, such as the route the run took."""
        self._root.set_attributes(_clean(attributes))

    def finish(self, *, output: Any = None, error: str | None = None) -> None:
        """End every open span, innermost first, then the root. Call once, at the run's end."""
        while self._open:
            self._end(self._open.pop().span, error=error)
        extra = {LF_TRACE_OUTPUT: as_text(output)} if output is not None else None
        self._end(self._root, attributes=extra, output=output, error=error)

    def _end(
        self,
        span: Span,
        *,
        attributes: Attributes | None = None,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        """Set the closing attributes and status, then end the span."""
        span.set_attributes(_clean(attributes))
        if output is not None:
            span.set_attribute(LF_OUTPUT, as_text(output))
        if error is not None:
            span.set_attributes({LF_LEVEL: "ERROR", LF_STATUS: as_text(error)})
            span.set_status(self._error_status(as_text(error)))
        span.end()
