"""Day 2, step 4: tool calling on the direct path, measured.

What it does
    Gives a model one tool (`get_recent_deploys`), asks a fixed question that
    needs it, and runs the call loop: the model asks for the tool, our code
    runs it in-process, the result goes back, the model answers. Each run
    records input tokens, output tokens, wall time, how many tool calls were
    made, and whether the answer was right. `main` repeats this and prints a
    table. Step 5 repeats it through an MCP server so the two can be compared.

Why "direct" is a plain function call here
    The lesson binds the tool in a LangGraph `ToolNode`. This loop calls the
    tool itself instead. The point of the measurement is the cost of the
    transport (in-process call vs an MCP hop). A `ToolNode` would add
    framework time to one side only. Same tool, same schema, same loop on
    both paths, so the only difference left is the transport.

The loop is bounded
    `max_turns` caps model calls per run. A model that keeps asking for tools,
    or keeps sending bad arguments, would otherwise loop forever and burn
    tokens. Hitting the cap is recorded, not hidden.

The model never executes anything
    It emits a tool name and arguments. Those arguments come from a model that
    has read untrusted text, so `execute_tool_call` treats them as input from
    the internet: unknown tool names and invalid arguments become an error
    message sent back to the model. They never crash the run and never reach
    the data file unchecked (`DeploysArgs` does the validating, see tools.py).

What the numbers can and cannot say
    - Input tokens include the tool's name, description and JSON Schema. They
      are sent on every turn. With one small tool that is tens of tokens.
      The MCP overhead the lesson warns about grows with the NUMBER of tools,
      so one tool will understate it. Say so when you quote the number.
    - The first run after a model load is slow. `main` runs one warm-up call
      that is printed but excluded from the summary.
    - Sampling is random, so runs differ. Ten runs show a shape, not a rate.
    - The question says "use the tool". That tests whether the model can call
      it correctly, not whether it would choose to.

Run it:   uv run python -m app.direct_run [runs]
Test it:  uv run pytest
"""

import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool, ToolException
from pydantic import ValidationError

from app.tools import make_deploys_tool

# Fixed so runs are comparable across the direct and MCP paths. The data
# (data/deploys.json) holds two auth deploys near the boundary: 2.14.1 at 3h
# and 2.14.0 at 30h. A correct answer names the first and not the second.
TASK = (
    "Which deploys to the auth service happened in the last 24 hours? "
    "Use the get_recent_deploys tool, then answer in one sentence."
)
MUST_INCLUDE = "2.14.1"
MUST_EXCLUDE = "2.14.0"

DEFAULT_RUNS = 10
MAX_TURNS = 4


class ToolModel(Protocol):
    """A chat model with its tools already bound: messages in, one message out.

    `ChatOllama(...).bind_tools([tool])` fits. So does a scripted fake in
    tests. Injected, so the loop never builds a client.
    """

    def invoke(self, messages: list[BaseMessage]) -> AIMessage: ...


@dataclass(frozen=True)
class RunStats:
    """What one run cost and did.

    Token fields are None when the model did not report usage on some turn.
    A partial sum would look like a real number, so it is dropped instead.
    """

    input_tokens: int | None
    output_tokens: int | None
    seconds: float
    turns: int
    tool_calls: int
    tool_errors: int
    hit_turn_cap: bool
    correct: bool
    answer: str


def execute_tool_call(tools: dict[str, StructuredTool], call: dict) -> ToolMessage:
    """Run one model-requested tool call. Always returns a `ToolMessage`.

    Unknown tool name or invalid arguments become an error message the model
    can read and correct, instead of an exception that ends the run.
    Infrastructure faults (a missing data file) are not caught: the model
    cannot fix them, and hiding them would look like a model failure.
    """
    tool = tools.get(call["name"])
    if tool is None:
        return ToolMessage(
            content=f"error: unknown tool {call['name']!r}",
            tool_call_id=call["id"],
            status="error",
        )
    try:
        return tool.invoke(call)
    except (ValidationError, ToolException) as err:
        return ToolMessage(
            content=f"error: {err}", tool_call_id=call["id"], status="error"
        )


def _sum_tokens(messages: list[AIMessage], key: str) -> int | None:
    """Total one usage field across turns, or None if any turn lacked it."""
    total = 0
    for m in messages:
        usage = m.usage_metadata
        if not usage or key not in usage:
            return None
        total += usage[key]
    return total


def is_correct(answer: str) -> bool:
    """True when the answer names the in-window deploy and not the stale one."""
    return MUST_INCLUDE in answer and MUST_EXCLUDE not in answer


def run_once(
    model: ToolModel,
    tools: dict[str, StructuredTool],
    question: str = TASK,
    *,
    max_turns: int = MAX_TURNS,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """One full run: ask, execute requested tools, repeat until the model answers.

    Wall time covers every model call and every tool call. It is measured
    with a monotonic clock, so a system clock change cannot skew it.
    """
    messages: list[BaseMessage] = [HumanMessage(question)]
    replies: list[AIMessage] = []
    tool_calls = tool_errors = 0
    answer, hit_cap = "", False
    start = clock()
    for turn in range(max_turns):
        ai = model.invoke(messages)
        replies.append(ai)
        messages.append(ai)
        if not ai.tool_calls:
            answer = str(ai.content)
            break
        for call in ai.tool_calls:
            result = execute_tool_call(tools, call)
            tool_calls += 1
            tool_errors += result.status == "error"
            messages.append(result)
        if turn == max_turns - 1:
            hit_cap = True
    seconds = clock() - start
    return RunStats(
        input_tokens=_sum_tokens(replies, "input_tokens"),
        output_tokens=_sum_tokens(replies, "output_tokens"),
        seconds=seconds,
        turns=len(replies),
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        hit_turn_cap=hit_cap,
        correct=is_correct(answer),
        answer=answer,
    )


def summarise(runs: list[RunStats]) -> dict[str, float | int | None]:
    """Reduce runs to the numbers worth quoting. Pure.

    Median, not mean, for time: one slow run (a model reload) should not
    drag the headline number. Token means skip runs with unknown usage and
    return None if none are known.
    """

    def mean(values: list[int | None]) -> float | None:
        known = [v for v in values if v is not None]
        return statistics.mean(known) if known else None

    return {
        "runs": len(runs),
        "called_tool": sum(r.tool_calls > 0 for r in runs),
        "correct": sum(r.correct for r in runs),
        "hit_turn_cap": sum(r.hit_turn_cap for r in runs),
        "median_seconds": statistics.median(r.seconds for r in runs),
        "mean_input_tokens": mean([r.input_tokens for r in runs]),
        "mean_output_tokens": mean([r.output_tokens for r in runs]),
    }


def format_row(label: str, r: RunStats) -> str:
    """One table row. Unknown tokens print as '?'."""
    i = "?" if r.input_tokens is None else r.input_tokens
    o = "?" if r.output_tokens is None else r.output_tokens
    return (
        f"{label:>7} {i:>7} {o:>7} {r.seconds:>7.2f} {r.turns:>5} "
        f"{r.tool_calls:>5} {r.tool_errors:>4} {'yes' if r.correct else 'no':>8}"
    )


HEADER = f"{'run':>7} {'in_tok':>7} {'out_tok':>7} {'secs':>7} {'turns':>5} {'calls':>5} {'errs':>4} {'correct':>8}"


def main(argv: list[str] | None = None) -> None:
    """Run one warm-up and N measured runs against the small model, then print.

    The only place that reads `.env` and builds a client. `SMALL_MODEL` must
    be set. A missing name stops the run loudly rather than falling back.
    """
    from dotenv import load_dotenv
    from langchain_ollama import ChatOllama

    load_dotenv()
    args = sys.argv[1:] if argv is None else argv
    n = int(args[0]) if args else DEFAULT_RUNS
    tool = make_deploys_tool()
    model_name = os.environ["SMALL_MODEL"]
    model = ChatOllama(model=model_name).bind_tools([tool])
    tools = {tool.name: tool}

    print(f"model: {model_name}   task: {TASK}")
    print(HEADER)
    print(format_row("warm-up", run_once(model, tools)), " (excluded)")
    runs = []
    for i in range(1, n + 1):
        stats = run_once(model, tools)
        runs.append(stats)
        print(format_row(str(i), stats))
    s = summarise(runs)
    print(
        f"\ncalled the tool: {s['called_tool']}/{s['runs']}   "
        f"correct: {s['correct']}/{s['runs']}   "
        f"hit turn cap: {s['hit_turn_cap']}   "
        f"median {s['median_seconds']:.2f}s   "
        f"mean tokens in/out: {s['mean_input_tokens']}/{s['mean_output_tokens']}"
    )


if __name__ == "__main__":
    main()
