"""FindyFixy's run events as OpenTelemetry spans: the adapter between findyfixy/events.py and harness/trace.

What it does
    `EventSpans` is an event sink. Give it every `Event` of a run, in order,
    and it builds that run's trace through `wyld3rness.trace.spans.SpanTree`:

        run_started              root span "findyfixy run" (the question is the trace input)
        node_entered/exited      one span per graph node: recall (a retriever),
                                 investigate, small, large
        model_request/reply      a generation span: model, served-by model,
                                 input and output tokens, prompt and reply
        tool_call/result         a tool span inside investigate. Several calls from one
                                 reply are siblings, and results close them in call
                                 order. A call never run (the turn cap) ends as an error
        validation, retry,       events on the span in progress
        fault_injected
        route                    an event, and the route target on the root span
        run_finished/failed      the triage (or the error) as the trace output; ends the trace

    A hunt (many runs on one emitter) sends `run_started` once per run, so each
    run gets its own trace, and all of them share one `session.id` (from
    `hunt_started` to `hunt_finished`), which Langfuse shows as one session.
    The `hunt_*` events have no span of their own.

    An eval case has no `run_started` event: the graph alone runs, inside the
    harness's case span. `RunTracer` covers that: the code that runs a case
    opens the run directly (`with run_tracer.run(question) as spans`), the
    session's shared emitter forwards events to the run open on the calling
    thread, and the run's spans nest under the case span through
    OpenTelemetry's current context.

Why an adapter and not spans in the graph
    The graph already reports every step through `emit` (findyfixy/events.py), and
    the console, the tests and the event log read the same stream. Tracing is
    one more reader. The graph does not change, and a run with tracing off is
    the same code path as before.

Failure modes
    - The sink runs on the run's own thread, so it must never block: it only
      creates and ends spans in memory. The export to the backend happens on
      the exporter's thread (wyld3rness/trace/tracing.py).
    - A bug in this mapping must not fail a triage. `__call__` catches any
      exception, counts it in `errors`, and ends that run's trace as an
      error, so the backend never gets spans whose root never arrives. A test
      asserts the count stays 0 for every event kind the graph emits.
    - Events are trusted in shape (the graph writes them), but their values
      carry model output, so they are sent as text, never interpreted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
import uuid

from findyfixy.events import Emit, Event
from wyld3rness.trace.spans import (
    GEN_AI_INPUT_TOKENS,
    GEN_AI_OPERATION,
    GEN_AI_OUTPUT_TOKENS,
    GEN_AI_PROVIDER,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_TOOL_NAME,
    SpanTree,
    attribute_value,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

TRACE_NAME = "findyfixy run"
# The recall node is a lookup; Langfuse shows a retriever span differently.
_NODE_KIND = {"recall": "retriever"}


class EventSpans:
    """An event sink that turns one emitter's events into traces. One per emitter."""

    def __init__(self, tracer: Tracer) -> None:
        """Hold the tracer. No span exists until `run_started`."""
        self._tracer = tracer
        self._tree: SpanTree | None = None
        # Tool calls opened and not yet answered, oldest first. Tool results
        # come back in call order (LangGraph's ToolNode), so the oldest
        # pending call is the one a result answers.
        self._pending_tools: list[str] = []
        self._tool_seq = 0
        self._session: str | None = None  # set during a hunt
        self.errors = 0  # mapping bugs swallowed; tests assert it stays 0

    @property
    def trace_id(self) -> str | None:
        """The trace id of the run in progress, or None between runs."""
        return self._tree.trace_id if self._tree is not None else None

    def __call__(self, event: Event) -> None:
        """Map one event. Never raises: see "Failure modes" above."""
        try:
            self._handle(event)
        except Exception as err:  # noqa: BLE001 - tracing must never fail a run
            self.errors += 1
            tree, self._tree = self._tree, None
            if tree is not None:
                try:
                    tree.finish(error=f"tracing adapter error: {type(err).__name__}")
                except Exception:  # noqa: BLE001 - the trace is lost; the run is not
                    pass

    def _handle(self, event: Event) -> None:
        """The mapping table in the module docstring, one branch per event kind."""
        d: dict[str, Any] = event.data
        if event.kind == "hunt_started":
            self._session = f"hunt {uuid.uuid4().hex[:12]}"
            return
        if event.kind == "hunt_finished":
            self._session = None
            return
        if event.kind == "run_started":
            self.start_run(d.get("question"), {
                "findyfixy.run": d.get("run"),
                "findyfixy.mode": d.get("mode"),
                "findyfixy.small_model": d.get("small_model"),
                "findyfixy.large_model": d.get("large_model"),
                "findyfixy.threshold": d.get("threshold"),
                "findyfixy.tools": d.get("tools"),
                "findyfixy.recall": d.get("recall"),
            })
            return
        tree = self._tree
        if tree is None:  # an event outside a run (a hunt's own events)
            return
        kind = event.kind
        if kind == "node_entered":
            node = d["node"]
            # Current while the node runs, so spans made inside it (recall's
            # stages) nest under it. Nodes open and close strictly nested.
            tree.open("node", f"node {node}", kind=_NODE_KIND.get(node, "span"), current=True,
                      attributes={"findyfixy.node": node, GEN_AI_REQUEST_MODEL: d.get("model")})
        elif kind == "node_exited":
            # Tool calls the model asked for but the loop never ran (it stopped
            # at its turn cap) must not look like lookups that happened.
            for key in self._pending_tools:
                tree.close(key, error="not run: the investigate loop stopped first")
            self._pending_tools = []
            out = {k: v for k, v in d.items() if k not in ("node", "model")}
            tree.close("node", output=out or None)
        elif kind == "model_request":
            # Scripted (fake) investigate calls have no model name.
            name = f"model {d['model']}" if d.get("model") else "model (scripted)"
            tree.open("model", name, kind="generation", input=d.get("prompt"),
                      attributes={GEN_AI_OPERATION: "chat", GEN_AI_PROVIDER: "ollama",
                                  GEN_AI_REQUEST_MODEL: d.get("model"), "findyfixy.attempt": d.get("attempt")})
        elif kind == "model_reply":
            tree.close("model", output=d.get("reply"), attributes={
                GEN_AI_RESPONSE_MODEL: d.get("served_by"),
                GEN_AI_INPUT_TOKENS: d.get("input_tokens"),
                GEN_AI_OUTPUT_TOKENS: d.get("output_tokens"),
                "findyfixy.model_mismatch": d.get("mismatch"),
            })
        elif kind == "tool_call":
            self._tool_seq += 1
            key = f"tool {self._tool_seq}"
            tree.open(key, f"tool {d.get('name')}", kind="tool", input=d.get("args"), under="node",
                      attributes={GEN_AI_TOOL_NAME: d.get("name")})
            self._pending_tools.append(key)
        elif kind == "tool_result":
            if self._pending_tools:
                tree.close(self._pending_tools.pop(0), output=d.get("result"),
                           error="tool returned an error" if d.get("error") else None)
        elif kind in ("validation", "retry", "fault_injected", "route"):
            tree.event(kind, {k: attribute_value(v) for k, v in d.items()})
            if kind == "route":
                tree.set({"findyfixy.route": d.get("target")})
        elif kind == "run_finished":
            self.finish_run(output=d.get("triage"), model_used=d.get("model_used"))
        elif kind == "run_failed":
            self.finish_run(error=f"{d.get('error_type')}: {d.get('error')}")

    def start_run(self, question: Any, attributes: Mapping[str, Any] | None = None) -> None:
        """Open a run's top span: what `run_started` does, for callers that emit no such event."""
        if self._session is not None:
            attributes = {**(attributes or {}), "session.id": self._session}
        self._tree = SpanTree(self._tracer, TRACE_NAME, input=question, attributes=attributes)
        self._pending_tools = []

    def finish_run(self, *, output: Any = None, error: str | None = None, model_used: str | None = None) -> None:
        """End the open run, if any: what `run_finished` and `run_failed` do."""
        tree, self._tree = self._tree, None
        if tree is None:
            return
        if model_used is not None:
            tree.set({"findyfixy.model_used": model_used})
        tree.finish(output=output, error=error)


class RunTracer:
    """Run traces for code that shares one emitter across runs and threads (an eval session).

    `run(question)` opens a run for the calling thread; `wrap(emit)` returns an
    emitter that also feeds each event to the run open on the thread that
    emits it. Parallel eval workers never share a trace.

    The open run lives in a `ContextVar`, the mechanism OpenTelemetry uses for
    the current span. Code that copies the context to another thread (as
    LangGraph does for a node it runs on an executor) carries both the span
    and the run along; a thread-local would lose the run there.
    """

    def __init__(self, tracer: Tracer) -> None:
        """Hold the tracer. No run is open in any context yet."""
        self._tracer = tracer
        self._current: ContextVar[EventSpans | None] = ContextVar("findyfixy_run", default=None)

    @contextmanager
    def run(self, question: str, attributes: Mapping[str, Any] | None = None) -> Iterator[EventSpans]:
        """Open a run on this thread for the block. End it with `finish_run` on the yielded spans.

        A run still open when the block ends (an exception, or a caller that
        did not finish it) is ended here, as an error when the block raised.
        """
        spans = EventSpans(self._tracer)
        spans.start_run(question, attributes)
        token = self._current.set(spans)
        error: str | None = None
        try:
            yield spans
        except BaseException as err:
            error = f"{type(err).__name__}: {err}"
            raise
        finally:
            spans.finish_run(error=error)  # a no-op when the caller already finished it
            self._current.reset(token)

    def wrap(self, emit: Emit) -> Emit:
        """`emit`, plus the event fed to this thread's open run (if any)."""

        def traced(kind: Any, /, **data: Any) -> None:
            """Emit as before, then trace."""
            emit(kind, **data)
            spans = self._current.get()
            if spans is not None:
                spans(Event(kind, 0.0, data))  # spans use real time; `t` is not read

        return traced


def traced_sink(spans: EventSpans, sink: Callable[[Event], None],
                link: Callable[[str], str | None]) -> Callable[[Event], None]:
    """One sink for a traced console run: tracing first, then the viewer, plus each run's trace link.

    Tracing sees each event first, so a trace is complete before any viewer
    hears its run end. After `run_started` opens a trace, the viewer also gets
    `run_traced` with the trace id and `link(trace_id)`, the backend page that
    shows it (None when the backend project is not known), so the event log
    can link each run to its trace. A run whose trace failed to open gets no
    `run_traced`.
    """

    def both(event: Event) -> None:
        """Pass `event` to tracing and the viewer; announce a newly opened trace."""
        spans(event)
        sink(event)
        if event.kind == "run_started" and (trace_id := spans.trace_id) is not None:
            sink(Event("run_traced", event.t, {"trace_id": trace_id, "url": link(trace_id)}))

    return both
