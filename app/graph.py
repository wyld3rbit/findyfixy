import os
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

THRESHOLD = 0.7

QUESTIONS = [
    "A pod is in CrashLoopBackOff after a config change. First three checks?",
    "p95 latency on auth doubled at 02:00 UTC with no deploy. Likely causes?",
    "What does HTTP 429 from our vendor API mean and what do we do?",
    "Two regions disagree on a feature flag value. How do we find which is right?",
    "Customer reports intermittent voice-verification failures only on one carrier. Plan?",
]


class S(TypedDict, total=False):
    question: str
    answer: str
    confidence: float
    model_used: str


class Answer(BaseModel):
    answer: str
    confidence: float = Field(
        description="How sure you are the answer is correct, from 0 to 1."
    )


def _ask(model_env: str, question: str) -> tuple[str, Answer]:
    name = os.environ[model_env]
    llm = ChatOllama(model=name).with_structured_output(Answer)
    return name, llm.invoke(question)


def small_node(state: S) -> S:
    name, out = _ask("SMALL_MODEL", state["question"])
    return {"answer": out.answer, "confidence": out.confidence, "model_used": name}


def large_node(state: S) -> S:
    name, out = _ask("LARGE_MODEL", state["question"])
    return {"answer": out.answer, "confidence": out.confidence, "model_used": name}


def route(state: S) -> str:
    return END if state["confidence"] >= THRESHOLD else "large"


g = StateGraph(S)
g.add_node("small", small_node)
g.add_node("large", large_node)
g.set_entry_point("small")
g.add_conditional_edges("small", route)
g.add_edge("large", END)
app = g.compile()


def main() -> None:
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
