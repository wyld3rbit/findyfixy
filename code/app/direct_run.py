"""Day 2, step 4: the direct tool path, measured.

What it does
    Runs a fixed question that needs the deploy tool through the LangGraph
    tool loop (app/investigate.py), 10 times, and prints one row per run:
    input tokens, output tokens, wall time, tool calls, the arguments the
    model chose, and whether the answer was right. Step 5 repeats it with the
    tool served over MCP so the two can be compared.

Why "direct" means an in-process tool inside a ToolNode
    The lesson binds the tool with `bind_tools` in a LangGraph `ToolNode`.
    That is what `build_investigator` does. The tool is a plain Python
    function called in the same process. Step 5 changes only where the tool
    comes from (an MCP server). The loop, the model, the prompt and this
    measurement stay the same, so the difference left is the transport.

The loop is bounded
    The graph carries a turn counter and stops at `MAX_TURNS`. A hit is
    recorded in the table, not hidden.

What the numbers can and cannot say
    - Input tokens include the tool's name, description and JSON Schema. They
      are sent on every turn. With one small tool that is tens of tokens.
      The MCP overhead the lesson warns about grows with the NUMBER of tools,
      so one tool will understate it. Say so when you quote the number.
    - The first run after a model load is slow. `main` runs one warm-up call
      that is printed but excluded from the summary.
    - Sampling is random, but this model repeated itself almost exactly on
      this task. Ten identical rows show one behaviour, not a spread.
    - The question says "use the tool". That tests whether the model can call
      it correctly, not whether it would choose to.
    - The `args` column shows what the model asked for. Legal but wrong
      arguments (a 1 hour window for a 24 hour question) pass validation, so
      this column is the only place the mistake is visible.

Run it:   uv run python -m app.direct_run [runs]
Test it:  uv run pytest
"""

import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.investigate import analyse, build_investigator, format_args
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


@dataclass(frozen=True)
class RunStats:
    """What one run cost and did. Built from an `Investigation` plus the clock.

    Token fields are None when the model did not report usage on some turn.
    `args` lists each tool call's arguments, joined with " | ".
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
    args: str


def is_correct(answer: str) -> bool:
    """True when the answer names the in-window deploy and not the stale one."""
    return MUST_INCLUDE in answer and MUST_EXCLUDE not in answer


def run_once(
    investigator: CompiledStateGraph,
    question: str = TASK,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """One full run through the tool loop, timed.

    Wall time covers every model call and every tool call, measured with a
    monotonic clock so a system clock change cannot skew it.
    """
    start = clock()
    final = investigator.invoke({"messages": [HumanMessage(question)]})
    return stats_from(final["messages"], clock() - start)


def stats_from(messages: list, seconds: float) -> RunStats:
    """Turn a finished conversation and its wall time into a table row. Pure.

    Shared by the direct path and the MCP paths (app/mcp_run.py), so every
    transport reports exactly the same columns, computed the same way.
    """
    inv = analyse(messages)
    return RunStats(
        input_tokens=inv.input_tokens,
        output_tokens=inv.output_tokens,
        seconds=seconds,
        turns=inv.turns,
        tool_calls=len(inv.calls),
        tool_errors=sum(c.error for c in inv.calls),
        hit_turn_cap=inv.hit_turn_cap,
        correct=is_correct(inv.answer),
        answer=inv.answer,
        args=" | ".join(format_args(c.args) for c in inv.calls),
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
        f"{r.tool_calls:>5} {r.tool_errors:>4} {'yes' if r.correct else 'no':>8}  {r.args}"
    )


HEADER = (
    f"{'run':>7} {'in_tok':>7} {'out_tok':>7} {'secs':>7} {'turns':>5} "
    f"{'calls':>5} {'errs':>4} {'correct':>8}  args"
)


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
    investigator = build_investigator(
        ChatOllama(model=model_name).bind_tools([tool]), tool
    )

    print(f"model: {model_name}   task: {TASK}")
    print(HEADER)
    print(format_row("warm-up", run_once(investigator)), " (excluded)")
    runs = []
    for i in range(1, n + 1):
        stats = run_once(investigator)
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
