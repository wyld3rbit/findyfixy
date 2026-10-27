"""Day 1 router, Day 2 contract: small-model-first, with validated triage output.

What it does
    1. The small model (1.5B) returns a `Triage` record (app/triage.py). An
       invalid reply is re-asked once with the error text.
    2. If the record is valid and confidence is at or above the threshold,
       it is final.
    3. If the record is still invalid, or confidence is low, the large model
       (7B) answers instead. That answer is final.

Why this shape
    Most triage questions are routine. A small model answers them fast and
    cheap. The large model is the expensive tail. The routing rule is the
    cost lever: raise the threshold and more traffic goes to the large model
    (more cost, usually better answers). Lower it and the reverse.

Why a graph, and not an if statement
    Two nodes would fit in an if statement. The graph pays off as the system
    grows. Each step is a named node with its own inputs and outputs. That
    makes it easy to trace, test, retry, and pause for a human later (Day 9).
    Day 4 adds a recall node before `small`. Day 7 adds a policy node.
    With a graph, those are new nodes. With an if statement, they are a rewrite.

Why the graph is built by a function that takes `ask`
    `build_graph` does not know how a model is called. The caller passes in
    `ask`, a function (model name, prompt) -> raw reply text. Production passes
    `ollama_ask`. Tests pass a fake that returns a chosen confidence.
    This is dependency injection. It buys three things:
      - Routing can be tested in milliseconds with no model and no network.
      - Day 8 can swap in a gateway client without touching the graph.
      - Importing this module does nothing. No client is built, no graph is
        compiled, no environment is read until `main()` or a test asks.
    The alternative, patching a module-level function in tests, couples tests
    to private names and breaks silently when the code is renamed.

Known weaknesses (read these before trusting the numbers)
    - Confidence is still self-reported. Validation now rejects values outside
      0 to 1, so the Day 1 "75" case retries and then escalates. It does not
      make the number meaningful: a small model can write 0.9 and be wrong.
    - The large model gets the same retry. If it is still invalid, the run
      raises `TriageError`. There is no third tier and no human yet.
    - No timeout. If Ollama is down, the run raises. Retries cover bad
      output, not infrastructure faults.

Run it:   uv run python -m app.graph
Test it:  uv run pytest
"""

import time
from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.events import Emit, no_emit
from app.investigate import analyse, content_text, format_evidence, investigation_prompt
from app.settings import MissingEnvironment, load_env, read_models
from app.triage import (
    Ask,
    Prior,
    Reply,
    Triage,
    TriageError,
    answer_options,
    ask_validated,
    format_prior,
)

# The recall step's seam: a question in, similar past incidents out. Production
# passes a function built on `app.retrieve.Retriever`. Tests pass a fake.
Recall = Callable[[str], Sequence[Prior]]

# The routing policy in one number. Treat it as a product decision, not a
# constant of nature: it trades cost against quality. Day 8 tunes it
# (0.6, 0.7, 0.8) and measures the effect on cost, latency and faithfulness.
THRESHOLD = 0.7

# The five questions from the lesson, word for word. They are fixed so that
# runs are comparable over time. Changing them would break that comparison.
QUESTIONS = [
    "A pod is in CrashLoopBackOff after a config change. First three checks?",
    "p95 latency on auth doubled at 02:00 UTC with no deploy. Likely causes?",
    "What does HTTP 429 from our vendor API mean and what do we do?",
    "Two regions disagree on a feature flag value. How do we find which is right?",
    "Customer reports intermittent voice-verification failures only on one carrier. Plan?",
]


class S(TypedDict, total=False):
    """The graph state: the shared notebook every node reads and writes.

    A node returns only the keys it changes. LangGraph merges them in.
    With no custom reducer, a new value for a key replaces the old one.
    That is why `large` overwrites `small`'s result and `model_used`.

    `triage` is None when the small model never produced a valid record. That
    is a real state, not an error: `route` sends it to `large`.

    `total=False` makes every key optional. That lets us start a run with
    just {"question": ...}. The cost: the type checker no longer guarantees
    that `triage` exists when `route` reads it. It exists at runtime only
    because `small` always runs first and always sets it.
    """

    question: str
    triage: Triage | None
    model_used: str
    evidence: str
    prior: str


def ollama_ask(model: str, prompt: str) -> str:
    """The real `ask`: call one Ollama model and return its raw reply text.

    `format=<JSON Schema>` turns on constrained decoding: Ollama masks tokens
    that would break the schema, so the reply is JSON of the right shape.
    Values are checked later, in `ask_validated`. We return text, not a parsed
    object, so a bad value fails in our code with a readable message.

    We build the client on every call. That is cheap because it is only a
    thin HTTP client. The expensive part is Ollama loading model weights into
    memory. On a 16 GB Mac, switching from the 1.5B to the 7B model can add a
    load delay, so an escalated run pays for a model swap on top of two calls.

    The call is greedy (temperature 0) and has a read timeout; see
    `app.triage.answer_options`.

    Fails when: Ollama is not running (connection error), the model is not
    pulled (404), or the server stalls past the timeout. Nothing here retries.
    """
    llm = ChatOllama(model=model, format=Triage.model_json_schema(), **answer_options())
    message = llm.invoke(prompt)
    # Ollama names the model that served the request in its response. Keep it
    # next to the text, so a log can show what ACTUALLY answered, not only what
    # we asked for. Absent metadata gives None (unknown), never a guess.
    return Reply(message.content, served_by=message.response_metadata.get("model"))


def build_graph(
    ask: Ask,
    *,
    small_model: str,
    large_model: str,
    threshold: float = THRESHOLD,
    emit: Emit = no_emit,
    investigator: CompiledStateGraph | None = None,
    investigate_model: str | None = None,
    recall: Recall | None = None,
) -> CompiledStateGraph:
    """Wire the router and return a compiled graph, ready for `.invoke()`.

    Nothing runs here. Compiling validates the wiring (every edge points at a
    real node) and returns a runnable. A model is called only on `.invoke()`.

    Model names are plain arguments, not environment lookups. Reading the
    environment is the caller's job (`main` does it). That keeps this function
    pure and keeps a missing variable from surfacing deep inside a graph run.

    `emit` reports node entry and exit, model calls and the route decision
    (app/events.py). The default reports nothing.

    `investigator` (app/investigate.py) is an optional tool-calling step that
    runs once, before `small`. Its tool results become `evidence` in state and
    in both models' prompts. If `large` takes over, it reuses that evidence and
    the tools are not called again. With no investigator the graph is exactly
    the two-node router it was before: same nodes, same prompts.

    `investigate_model` is the name to show for the investigator's model in
    events. The investigator is injected with its model already bound, so the
    graph cannot know the name on its own. `None` means unnamed.

    `recall` (app/retrieve.py) is an optional step that runs first. It looks up
    similar past incidents for the question and puts them in both models' prompts
    as `prior`. Retrieval before generation: if the right incident never reaches
    the prompt, no prompt recovers it. With no `recall` the graph is unchanged.
    """

    def recall_node(state: S) -> S:
        """Look up similar past incidents and store them as prompt text.

        An infrastructure fault (database down) propagates, like Ollama being
        down: a retry here would hide it. The exit event lists the incident IDs
        so the page and the log show what the model was given.
        """
        emit("node_entered", node="recall", model=None)
        items = list(recall(state["question"]))
        emit(
            "node_exited",
            node="recall",
            model=None,
            incidents=[item.incident_id for item in items],
        )
        return {"prior": format_prior(items)}

    def investigate(state: S) -> S:
        """Run the tool loop once, reporting each step as it finishes.

        The loop is a nested graph. Running it with `stream_mode="updates"`
        hands this node one update per step, so events are emitted while the
        loop runs, not after it. That matters for a live viewer: each model
        call here is a real request to Ollama, and the viewer should show it
        when it happens. `tool_call` is emitted BEFORE the tool runs (the loop
        is paused until this body returns to the generator) and `tool_result`
        after it. Failures from infrastructure (Ollama down) propagate like any
        other.

        The node's exit event carries the whole step's cost, because this step
        adds model calls to every run, even one that ends with no lookup.
        """
        emit("node_entered", node="investigate", model=investigate_model)
        messages: list[BaseMessage] = [
            HumanMessage(investigation_prompt(state["question"]))
        ]
        turn = 1
        emit("model_request", model=investigate_model, attempt=turn)
        updates = investigator.stream({"messages": list(messages)}, stream_mode="updates")
        for update in updates:
            for step, delta in update.items():
                new = delta.get("messages", [])
                messages.extend(new)
                if step == "agent":
                    for reply in new:
                        _emit_agent_reply(reply, turn)
                elif step == "tools":
                    for result in new:
                        if isinstance(result, ToolMessage):
                            emit(
                                "tool_result",
                                name=result.name,
                                result=content_text(result.content)[:600],
                                error=result.status == "error",
                            )
                    turn += 1
                    emit("model_request", model=investigate_model, attempt=turn)
        inv = analyse(messages)
        emit(
            "node_exited",
            node="investigate",
            model=investigate_model,
            calls=len(inv.calls),
            turns=inv.turns,
            input_tokens=inv.input_tokens,
            output_tokens=inv.output_tokens,
            hit_turn_cap=inv.hit_turn_cap,
            served_by=list(inv.served_by),
        )
        return {"evidence": format_evidence(inv.calls)}

    def _emit_agent_reply(reply: BaseMessage, turn: int) -> None:
        """Report one investigator model reply, then any tool calls in it."""
        if not isinstance(reply, AIMessage):
            return
        served_by = (reply.response_metadata or {}).get("model")
        usage = reply.usage_metadata or {}
        emit(
            "model_reply",
            model=investigate_model,
            attempt=turn,
            reply=content_text(reply.content),
            served_by=served_by,
            mismatch=bool(served_by and investigate_model and served_by != investigate_model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        for call in reply.tool_calls:
            emit("tool_call", name=call["name"], args=call["args"])

    def make_node(name: str, model: str, *, tolerate_invalid: bool) -> Callable[[S], S]:
        """Build a node that asks `model` and writes the result into state.

        `small` and `large` differ only in which model they call and in what
        happens when the model stays invalid after its retry. `small` records
        `triage=None` so the router can escalate. `large` has nowhere to
        escalate to, so it lets `TriageError` propagate. One factory means the
        two cannot drift apart. The node returns a partial state update;
        LangGraph merges it. See `S` for the merge rule.
        """

        def node(state: S) -> S:
            emit("node_entered", node=name, model=model)
            try:
                triage = ask_validated(
                    ask,
                    model,
                    state["question"],
                    emit=emit,
                    evidence=state.get("evidence", ""),
                    prior=state.get("prior", ""),
                )
            except TriageError:
                if not tolerate_invalid:
                    raise
                triage = None
            emit(
                "node_exited",
                node=name,
                model=model,
                triage=triage.model_dump() if triage else None,
            )
            return {"triage": triage, "model_used": model}

        return node

    def route(state: S) -> Literal["large", "__end__"]:
        """The routing policy. Returns the name of the next node, or END.

        This is a pure function of the state: no model call, no network, no
        clock. That makes it deterministic and testable on its own.

        A missing record always escalates. Otherwise the comparison is `>=`,
        so exactly 0.7 stays on the small model.

        One model, no second tier: when `small_model` and `large_model` are the
        same, escalating would ask the same model the same question again and
        double the latency for no new information. The run ends instead. This
        is how a single answering model is configured (set both names alike).

        The return annotation is load-bearing. LangGraph reads it to learn
        where this edge can go. With a bare `str` it cannot, and its diagram
        drops the `small -> large` edge. Runtime routing would still work,
        because routing uses the returned value. The annotation exists for
        tooling and for the reader. `tests/test_routing.py` pins it.
        """
        if small_model == large_model:
            return END
        triage = state["triage"]
        if triage is None or triage.confidence < threshold:
            return "large"
        return END

    def decide(state: S) -> Literal["large", "__end__"]:
        """Run `route` and report the decision. Kept apart so `route` stays pure.

        The `Literal` annotation is repeated on purpose: LangGraph reads the
        one on the function it is given, which is this one.
        """
        target = route(state)
        triage = state["triage"]
        emit(
            "route",
            confidence=triage.confidence if triage else None,
            threshold=threshold,
            target=target,
        )
        return target

    g = StateGraph(S)
    g.add_node("small", make_node("small", small_model, tolerate_invalid=True))
    g.add_node("large", make_node("large", large_model, tolerate_invalid=False))
    # Steps before the cheap model, in order: recall, then investigate. Each is
    # optional, and the chain ends at `small`.
    before = []
    if recall is not None:
        g.add_node("recall", recall_node)
        before.append("recall")
    if investigator is not None:
        g.add_node("investigate", investigate)
        before.append("investigate")
    chain = [*before, "small"]
    g.set_entry_point(chain[0])  # without optional steps, the cheap model
    for here, there in zip(chain, chain[1:]):
        g.add_edge(here, there)
    g.add_conditional_edges("small", decide)  # the decision point
    g.add_edge("large", END)  # a plain edge: after `large`, always stop
    return g.compile()


def main() -> None:
    """Run the five lesson questions and print triage, model, confidence, latency.

    Latency is wall time for the whole graph run, measured with perf_counter
    (a monotonic clock, so system clock changes cannot skew it). It includes
    model load time on a cold start, so the first question is often slower.
    Five samples is too few for a p95. It is enough to see the shape.
    Escalation rate (runs that reach `large`, divided by total runs) is the
    number to watch. Count it from the `model:` lines.

    Configuration comes from `app.settings`. A missing model name stops the run
    loudly with the variable's name, not with a silent default.
    """
    load_env()
    try:
        models = read_models()
    except MissingEnvironment as err:
        raise SystemExit(f"error: set the environment variable {err}") from None
    graph = build_graph(
        ollama_ask,
        small_model=models.small,
        large_model=models.large,
    )
    for q in QUESTIONS:
        start = time.perf_counter()
        result = graph.invoke({"question": q})
        latency = time.perf_counter() - start
        print(f"Q: {q}")
        t = result["triage"]
        print(f"A: {t.severity} | {t.component} | {t.next_action}")
        print(
            f"model: {result['model_used']} | confidence: {t.confidence} "
            f"| latency: {latency:.1f}s\n"
        )


if __name__ == "__main__":
    main()
