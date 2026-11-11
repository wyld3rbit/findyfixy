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

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
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


def _restore(token: Any, detach: Any) -> None:
    """Undo an `attach`, unless the context it was made in is already gone.

    LangGraph runs each node in a copy of the context. A node that raises is
    closed from outside that copy (by the run's failure event), where its
    token cannot be reset: the copy, and the span it made current, are
    already discarded, so there is nothing to restore. OpenTelemetry's
    `detach` would log a traceback for that; resetting the context variable
    directly lets the mismatch pass quietly. Tokens of another runtime
    context (selected with `OTEL_PYTHON_CONTEXT`; no `var`) go through
    `detach` as usual.
    """
    var = getattr(token, "var", None)
    if var is None:
        detach(token)
        return
    with suppress(ValueError):  # made in a context copy that no longer exists: nothing to restore
        var.reset(token)


def as_text(value: Any) -> str:
    """A value as span text: strings as they are, anything else as JSON. Cut and marked when long."""
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > MAX_VALUE_CHARS:
        return text[:MAX_VALUE_CHARS] + f"... [cut, {len(text)} chars]"
    return text


def attribute_value(value: Any) -> str | int | float | bool | None:
    """A value as an attribute value: None, numbers and bools as they are, anything else as cut text."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return as_text(value)


def mark_failed(span: Span, error: str) -> None:
    """Mark a span as failed, for OpenTelemetry (its status) and for Langfuse (its level and message)."""
    from opentelemetry.trace import Status, StatusCode

    text = as_text(error)
    span.set_attributes({LF_LEVEL: "ERROR", LF_STATUS: text})
    span.set_status(Status(StatusCode.ERROR, text))


def _clean(attributes: Attributes | None) -> dict[str, str | int | float | bool]:
    """Drop None values: OpenTelemetry attributes cannot be null, and unknown is not zero."""
    return {k: v for k, v in (attributes or {}).items() if v is not None}


@dataclass(frozen=True)
class _Open:
    """One open span: the key `close` finds it by, the span, and its parent (None: the root)."""

    key: str
    span: Span
    parent: Span | None
    token: object | None = None  # set when the span was made current; detached when it ends


class SpanTree:
    """The spans of one run. Build one per run; see the module notes."""

    _root: Span  # declared here: `__init__` reads a parent tree's root before setting its own

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        *,
        attributes: Attributes | None = None,
        input: Any = None,
        parent: SpanTree | None = None,
    ) -> None:
        """Start this tree's top span now.

        Its parent is `parent`'s top span when given (a later step of the same
        trace, such as the judge pass after the answer), else whatever span is
        current on this thread (`activate`), else none. Only a span with no
        parent names the trace: a nested tree must not rename the trace it
        joins.
        """
        # Resolved once per run, not per span: the run's thread makes every span.
        from opentelemetry import context as otel_context
        from opentelemetry import trace

        self._attach = otel_context.attach
        self._detach = otel_context.detach
        self._in_context = trace.set_span_in_context
        context = trace.set_span_in_context(parent._root) if parent is not None else None
        self._is_trace_root = parent is None and not trace.get_current_span().get_span_context().is_valid
        root_attrs = dict(_clean(attributes))
        if input is not None:
            root_attrs[LF_INPUT] = as_text(input)
        if self._is_trace_root:
            root_attrs[LF_TRACE_NAME] = name
            if input is not None:  # the trace's input (the list view), next to the span's (the tree)
                root_attrs[LF_TRACE_INPUT] = root_attrs[LF_INPUT]
        self._tracer = tracer
        self._use_span = trace.use_span
        self._root = tracer.start_span(name, context=context, attributes=root_attrs)
        # Innermost last. A list, not a dict: order is the parent rule.
        self._open: list[_Open] = []

    @property
    def trace_id(self) -> str:
        """The trace's id as 32 hex characters: what a backend's URL and score API use."""
        return format(self._root.get_span_context().trace_id, "032x")

    def follow(self, name: str) -> SpanTree:
        """A new tree in the same trace, as a child of this one's top span, even after it ended.

        For a later step of the same unit of work: the judge pass of an eval
        case runs after the case's answer span has ended.
        """
        return SpanTree(self._tracer, name, parent=self)

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Make this tree's top span the current one on this thread, for code that makes its own spans.

        An app called inside `activate` starts its spans as children of this
        one, through OpenTelemetry's context, without being handed a span.
        The span does not end when the block ends; `finish` ends it.
        """
        with self._use_span(self._root, end_on_exit=False):
            yield

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
        input: Any = None,
        under: str | None = None,
        current: bool = False,
    ) -> None:
        """Start a child of the innermost open span, or of the open span named `under`.

        `key` is what `close` names it by. Several open spans may share a key;
        `close` ends the innermost one.

        `current=True` also makes the span the current one on this thread
        until it closes, so code that makes its own spans (a retriever's
        stages, a judge's calls) nests them under it with no span handed over.
        Use it only for spans that open and close strictly nested, such as a
        graph node: the current context is a stack, and closing out of order
        would restore the wrong one. Siblings open at the same time (tool
        calls) stay not current.
        """
        attrs = {**_clean(attributes), LF_TYPE: kind}
        if input is not None:
            attrs[LF_INPUT] = as_text(input)
        parent = self._parent(under)
        span = self._tracer.start_span(name, context=self._in_context(parent), attributes=attrs)
        token = self._attach(self._in_context(span)) if current else None
        self._open.append(_Open(key, span, None if parent is self._root else parent, token))

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
            self._close_entry(entry)
        self._close_entry(target, attributes=attributes, output=output, error=error)

    def _close_entry(self, entry: _Open, **closing: Any) -> None:
        """End one open span, restore the context it replaced if it was current, forget it."""
        self._end(entry.span, **closing)
        if entry.token is not None:
            _restore(entry.token, self._detach)
        self._open.remove(entry)

    def event(self, name: str, attributes: Attributes | None = None) -> None:
        """Mark a moment (a validation result, a retry, a route) on the innermost open span."""
        self._parent().add_event(name, attributes=_clean(attributes))

    def set(self, attributes: Attributes) -> None:
        """Add attributes to the root span, such as the route the run took."""
        self._root.set_attributes(_clean(attributes))

    def finish(self, *, output: Any = None, error: str | None = None) -> None:
        """End every open span, innermost first, then the root. Call once, at the run's end."""
        while self._open:
            self._close_entry(self._open[-1], error=error)
        extra = {LF_TRACE_OUTPUT: as_text(output)} if output is not None and self._is_trace_root else None
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
            mark_failed(span, error)
        span.end()


class Generation:
    """One model call in progress, made by `generation`. Call `done` with what came back."""

    def __init__(self, span: Span) -> None:
        """Hold the span."""
        self._span = span

    def done(
        self,
        output: Any,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        response_model: str | None = None,
    ) -> None:
        """Record the reply and what the server reported about it."""
        self._span.set_attributes(
            _clean(
                {
                    LF_OUTPUT: as_text(output),
                    GEN_AI_INPUT_TOKENS: input_tokens,
                    GEN_AI_OUTPUT_TOKENS: output_tokens,
                    GEN_AI_RESPONSE_MODEL: response_model,
                }
            )
        )


@contextmanager
def generation(tracer: Tracer, model: str, prompt: Any, *, name: str | None = None) -> Iterator[Generation]:
    """A generation span for one model call, a child of the current span. Ends when the block ends.

    For code that calls a model outside an app's event stream, such as the
    harness's judge. An exception in the block marks the span as failed and
    is re-raised.
    """
    span = tracer.start_span(
        name or f"model {model}",
        attributes={
            LF_TYPE: "generation",
            GEN_AI_OPERATION: "chat",
            GEN_AI_PROVIDER: "ollama",
            GEN_AI_REQUEST_MODEL: model,
            LF_INPUT: as_text(prompt),
        },
    )
    try:
        yield Generation(span)
    except BaseException as err:
        mark_failed(span, f"{type(err).__name__}: {err}")
        raise
    finally:
        span.end()
