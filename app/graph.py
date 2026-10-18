"""Day 1: a small-model-first router, built as a LangGraph state machine.

What it does
    1. The small model (1.5B) answers the question and rates its own confidence.
    2. If confidence is at or above THRESHOLD, that answer is final.
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

Known weaknesses (read these before trusting the numbers)
    - Confidence is self-reported. The model grades its own work. Small
      models do this badly. In the Day 1 run, all 5 questions stayed on the
      small model, including answers that were incomplete.
    - The large answer is never checked. If the large model is also unsure,
      its answer is still returned. There is no third tier and no human yet.
    - No retry and no timeout. If Ollama is down or the model returns bad
      JSON, the run raises. Day 2 adds validate-retry-escalate.

Run it:  uv run python -m app.graph
"""

import os
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Reads SMALL_MODEL and LARGE_MODEL from .env into the process environment.
# By default it does not override variables that are already set. That lets
# CI or a container set them without editing .env.
load_dotenv()

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


def _ask(model_env: str, question: str) -> tuple[str, Answer]:
    """Call one model and return (model name, parsed Answer).

    `model_env` is the NAME of an environment variable, not a model name.
    Model names live in .env so they can be swapped without a code change
    (Day 8 points these at a gateway). `os.environ[...]` raises KeyError if
    the variable is missing. That is on purpose: a missing model name should
    stop the run loudly, not fall back to a silent default.

    We build the client on every call. That is cheap because it is only a
    thin HTTP client. The expensive part is Ollama loading model weights into
    memory. On a 16 GB Mac, switching from the 1.5B to the 7B model can add a
    load delay, so an escalated run pays for a model swap on top of two calls.
    """
    name = os.environ[model_env]
    llm = ChatOllama(model=name).with_structured_output(Answer)
    return name, llm.invoke(question)


def small_node(state: S) -> S:
    """Cheap first pass. Always runs first. Sets the answer and the confidence."""
    name, out = _ask("SMALL_MODEL", state["question"])
    return {"answer": out.answer, "confidence": out.confidence, "model_used": name}


def large_node(state: S) -> S:
    """Expensive fallback. Runs only when `route` sends the run here.

    It overwrites `answer`, `confidence` and `model_used`. The small answer is
    discarded, not kept for comparison. If you wanted to audit how often the
    large model disagrees with the small one, you would keep both. Day 6
    traces capture the small answer anyway.
    """
    name, out = _ask("LARGE_MODEL", state["question"])
    return {"answer": out.answer, "confidence": out.confidence, "model_used": name}


def route(state: S) -> str:
    """The routing policy. Returns the name of the next node, or END.

    This is a pure function of the state: no model call, no network, no
    clock. That makes it deterministic and testable without a model. We
    tested the wiring by swapping `_ask` for a fake that returns a chosen
    confidence, then checking which nodes ran.

    Drawing caveat: the return type is `str`, so LangGraph cannot tell which
    nodes this can reach. Its own diagram omits the `small -> large` edge.
    Runtime routing is unaffected. A `Literal["large", "__end__"]` return
    type would fix the diagram. See docs/graph.md.
    """
    return END if state["confidence"] >= THRESHOLD else "large"


# Build the graph. Nodes are the steps. Edges decide what runs next.
g = StateGraph(S)
g.add_node("small", small_node)
g.add_node("large", large_node)
g.set_entry_point("small")  # every run starts at the cheap model
g.add_conditional_edges("small", route)  # the decision point
g.add_edge("large", END)  # a plain edge: after `large`, always stop
app = g.compile()  # validates the wiring and returns a runnable


def main() -> None:
    """Run the five lesson questions and print model, confidence and latency.

    Latency is wall time for the whole graph run, measured with perf_counter
    (a monotonic clock, so system clock changes cannot skew it). It includes
    model load time on a cold start, so the first question is often slower.
    Five samples is too few for a p95. It is enough to see the shape.
    Escalation rate (runs that reach `large`, divided by total runs) is the
    number to watch. Count it from the `model:` lines.
    """
    for q in QUESTIONS:
        start = time.perf_counter()
        result = app.invoke({"question": q})
        latency = time.perf_counter() - start
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(
            f"model: {result['model_used']} | confidence: {result['confidence']} "
            f"| latency: {latency:.1f}s\n"
        )


if __name__ == "__main__":
    main()
