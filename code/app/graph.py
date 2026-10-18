"""Day 1: a small-model-first router, built as a LangGraph state machine.

What it does
    1. The small model (1.5B) answers the question and rates its own confidence.
    2. If confidence is at or above the threshold, that answer is final.
    3. If not, the large model (7B) answers instead. That answer is final.

Why this shape
    Most triage questions are routine. A small model answers them fast and
    cheap. The large model is the expensive tail. The routing rule is the
    cost lever: raise the threshold and more traffic goes to the large model
    (more cost, usually better answers). Lower it and the reverse.

Why a graph, and not an if statement
    Two nodes would fit in an if statement. The graph pays off as the system
    grows. Each step is a named node with its own inputs and outputs. That
    makes it easy to trace, test, retry, and pause for a human later (Day 9).
    Day 4 adds a retrieval node before `small`. Day 7 adds a policy node.
    With a graph, those are new nodes. With an if statement, they are a rewrite.

Why the graph is built by a function that takes `ask`
    `build_graph` does not know how a model is called. The caller passes in
    `ask`, a function (model name, question) -> Answer. Production passes
    `ollama_ask`. Tests pass a fake that returns a chosen confidence.
    This is dependency injection. It buys three things:
      - Routing can be tested in milliseconds with no model and no network.
      - Day 8 can swap in a gateway client without touching the graph.
      - Importing this module does nothing. No client is built, no graph is
        compiled, no environment is read until `main()` or a test asks.
    The alternative, patching a module-level function in tests, couples tests
    to private names and breaks silently when the code is renamed.

Known weaknesses (read these before trusting the numbers)
    - Confidence is self-reported. The model grades its own work. Small
      models do this badly. In the Day 1 run, all 5 questions stayed on the
      small model, including answers that were incomplete.
    - The large answer is never checked. If the large model is also unsure,
      its answer is still returned. There is no third tier and no human yet.
    - No retry and no timeout. If Ollama is down or the model returns bad
      JSON, the run raises. Day 2 adds validate-retry-escalate.

Run it:   uv run python -m app.graph
Test it:  uv run pytest
"""

import os
import time
from collections.abc import Callable
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

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
    That is why `large` overwrites `small`'s answer and `model_used`.

    `total=False` makes every key optional. That lets us start a run with
    just {"question": ...}. The cost: the type checker no longer guarantees
    that `confidence` exists when `route` reads it. It exists at runtime only
    because `small` always runs first and always sets it.
    """

    question: str
    answer: str
    confidence: float
    model_used: str


class Answer(BaseModel):
    """The contract for what a model must return: an answer plus a confidence.

    The schema is sent to Ollama so the reply comes back as JSON of this shape.
    Shape is constrained. Values are not guaranteed. In the Day 1 run the
    small model wrote confidence = 75 (meaning 75%) though the description
    says 0 to 1. A range check (ge=0, le=1) crashed the run, so it was removed
    and the raw number is used as written.

    Consequence: any value above 1 always passes the `>= 0.7` check. A model
    that writes on a 0-to-100 scale will almost never escalate. Day 2 handles
    this properly with validation and one retry.
    """

    answer: str
    confidence: float = Field(
        description="How sure you are the answer is correct, from 0 to 1."
    )


# The one seam between the graph and the outside world: (model name, question)
# in, a parsed Answer out. Anything with this shape can drive the graph.
Ask = Callable[[str, str], Answer]


def ollama_ask(model: str, question: str) -> Answer:
    """The real `ask`: call one Ollama model and return a parsed Answer.

    We build the client on every call. That is cheap because it is only a
    thin HTTP client. The expensive part is Ollama loading model weights into
    memory. On a 16 GB Mac, switching from the 1.5B to the 7B model can add a
    load delay, so an escalated run pays for a model swap on top of two calls.

    Fails when: Ollama is not running (connection error), the model is not
    pulled (404), or the reply does not parse into `Answer` (an
    OutputParserException). Nothing here retries. See the module docstring.
    """
    llm = ChatOllama(model=model).with_structured_output(Answer)
    return llm.invoke(question)


def build_graph(
    ask: Ask,
    *,
    small_model: str,
    large_model: str,
    threshold: float = THRESHOLD,
) -> CompiledStateGraph:
    """Wire the router and return a compiled graph, ready for `.invoke()`.

    Nothing runs here. Compiling validates the wiring (every edge points at a
    real node) and returns a runnable. A model is called only on `.invoke()`.

    Model names are plain arguments, not environment lookups. Reading the
    environment is the caller's job (`main` does it). That keeps this function
    pure and keeps a missing variable from surfacing deep inside a graph run.
    """

    def make_node(model: str) -> Callable[[S], S]:
        """Build a node that asks `model` and writes the result into state.

        `small` and `large` differ only in which model they call. One factory
        means they cannot drift apart. The node returns a partial state
        update; LangGraph merges it. See `S` for the merge rule.
        """

        def node(state: S) -> S:
            out = ask(model, state["question"])
            return {
                "answer": out.answer,
                "confidence": out.confidence,
                "model_used": model,
            }

        return node

    def route(state: S) -> Literal["large", "__end__"]:
        """The routing policy. Returns the name of the next node, or END.

        This is a pure function of the state: no model call, no network, no
        clock. That makes it deterministic and testable on its own.

        The return annotation is load-bearing. LangGraph reads it to learn
        where this edge can go. With a bare `str` it cannot, and its diagram
        drops the `small -> large` edge. Runtime routing would still work,
        because routing uses the returned value. The annotation exists for
        tooling and for the reader. `tests/test_routing.py` pins it.

        The comparison is `>=`, so exactly 0.7 stays on the small model.
        """
        return END if state["confidence"] >= threshold else "large"

    g = StateGraph(S)
    g.add_node("small", make_node(small_model))
    g.add_node("large", make_node(large_model))
    g.set_entry_point("small")  # every run starts at the cheap model
    g.add_conditional_edges("small", route)  # the decision point
    g.add_edge("large", END)  # a plain edge: after `large`, always stop
    return g.compile()


def main() -> None:
    """Run the five lesson questions and print model, confidence and latency.

    Latency is wall time for the whole graph run, measured with perf_counter
    (a monotonic clock, so system clock changes cannot skew it). It includes
    model load time on a cold start, so the first question is often slower.
    Five samples is too few for a p95. It is enough to see the shape.
    Escalation rate (runs that reach `large`, divided by total runs) is the
    number to watch. Count it from the `model:` lines.

    This is the only place that reads `.env` and the environment.
    `os.environ[...]` raises KeyError if a name is missing. That is on
    purpose: a missing model name should stop the run loudly, not fall back
    to a silent default. `load_dotenv()` does not override variables that are
    already set, so CI or a container can set them without editing `.env`.
    """
    load_dotenv()
    graph = build_graph(
        ollama_ask,
        small_model=os.environ["SMALL_MODEL"],
        large_model=os.environ["LARGE_MODEL"],
    )
    for q in QUESTIONS:
        start = time.perf_counter()
        result = graph.invoke({"question": q})
        latency = time.perf_counter() - start
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(
            f"model: {result['model_used']} | confidence: {result['confidence']} "
            f"| latency: {latency:.1f}s\n"
        )


if __name__ == "__main__":
    main()
