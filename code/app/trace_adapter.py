"""FindyFixy's run events as OpenTelemetry spans: the adapter between app/events.py and harness/trace.

What it does
    `EventSpans` is an event sink. Give it every `Event` of a run, in order,
    and it builds that run's trace through `harness.trace.spans.SpanTree`:

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
    run gets its own trace. The `hunt_*` events have no span.

Why an adapter and not spans in the graph
    The graph already reports every step through `emit` (app/events.py), and
    the console, the tests and the event log read the same stream. Tracing is
    one more reader. The graph does not change, and a run with tracing off is
    the same code path as before.

Failure modes
    - The sink runs on the run's own thread, so it must never block: it only
      creates and ends spans in memory. The export to the backend happens on
      the exporter's thread (harness/trace/tracing.py).
    - A bug in this mapping must not fail a triage. `__call__` catches any
      exception, counts it in `errors`, and ends that run's trace as an
      error, so the backend never gets spans whose root never arrives. A test
      asserts the count stays 0 for every event kind the graph emits.
    - Events are trusted in shape (the graph writes them), but their values
      carry model output, so they are sent as text, never interpreted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from app.events import Event
from harness.trace.spans import (
    GEN_AI_INPUT_TOKENS,
    GEN_AI_OPERATION,
    GEN_AI_OUTPUT_TOKENS,
    GEN_AI_PROVIDER,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_TOOL_NAME,
    SpanTree,
    as_text,
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
        self.errors = 0  # mapping bugs swallowed; tests assert it stays 0
        self.trace_ids: list[str] = []  # one per run, in order

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
        if event.kind == "run_started":
            self._tree = SpanTree(
                self._tracer,
                TRACE_NAME,
                input=d.get("question"),
                attributes={
                    "findyfixy.run": d.get("run"),
                    "findyfixy.mode": d.get("mode"),
                    "findyfixy.small_model": d.get("small_model"),
                    "findyfixy.large_model": d.get("large_model"),
                    "findyfixy.threshold": d.get("threshold"),
                    "findyfixy.tools": d.get("tools"),
                    "findyfixy.recall": d.get("recall"),
                },
            )
            self.trace_ids.append(self._tree.trace_id)
            self._pending_tools = []
            return
        tree = self._tree
        if tree is None:  # an event outside a run (a hunt's own events)
            return
        kind = event.kind
        if kind == "node_entered":
            node = d["node"]
            tree.open("node", f"node {node}", kind=_NODE_KIND.get(node, "span"),
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
            tree.event(kind, {k: _scalar(v) for k, v in d.items()})
            if kind == "route":
                tree.set({"findyfixy.route": d.get("target")})
        elif kind == "run_finished":
            tree.set({"findyfixy.model_used": d.get("model_used")})
            tree.finish(output=d.get("triage"))
            self._tree = None
        elif kind == "run_failed":
            tree.finish(error=f"{d.get('error_type')}: {d.get('error')}")
            self._tree = None


def _scalar(value: Any) -> str | int | float | bool | None:
    """An event value as an attribute value: numbers as they are, anything else as cut text."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return as_text(value)


def tee(*sinks: Callable[[Event], None]) -> Callable[[Event], None]:
    """One sink that hands each event to every sink in order. For console sink plus tracing."""

    def each(event: Event) -> None:
        """Pass `event` to every sink."""
        for sink in sinks:
            sink(event)

    return each
