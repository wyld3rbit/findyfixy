"""Day 2, step 5: the MCP path, measured. Same task, same loop, other transport.

What it does
    Runs the fixed deploy question through the same tool loop as
    app/direct_run.py, but with the tool served by an MCP server over stdio and
    reached through `langchain-mcp-adapters`. It prints the same table, so the
    two paths compare column for column.

Modes
    session   One server process and one MCP session for the whole run. Startup
              is timed once and printed separately. This is the realistic best
              case: a client that connects once and keeps the connection.

What the numbers can and cannot say
    - The model, the prompt, the loop and the tool logic are identical to the
      direct path. What differs is the transport and the sync/async plumbing:
      the graph runs with `ainvoke`, and the model node is still synchronous,
      so LangGraph runs it in a worker thread. That small constant cost is part
      of what is measured here and is not paid by the direct path.
    - Input tokens include the tool's name, description and schema, sent every
      turn. The server's schema is built from the same fields as the direct
      tool's (tests/test_mcp.py compares them), so a token difference would
      come from the transport, not the wording. With one tool it will be small.
      MCP's token cost grows with the number of tools exposed.
    - Startup (spawn, initialize, list tools) is not in the per-run time in
      `session` mode. It is a one-off, printed on its own line.
    - Ten runs of this model repeated almost exactly on the direct path, so
      expect little spread here too.

Run it:   uv run python -m app.mcp_run session [runs]
Test it:  uv run pytest
"""

import asyncio
import os
import sys
import time
from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.direct_run import (
    DEFAULT_RUNS,
    HEADER,
    TASK,
    RunStats,
    format_row,
    stats_from,
    summarise,
)
from app.investigate import build_investigator
from app.mcp_client import persistent_tools

MODES = ("session",)


async def arun_once(
    investigator: CompiledStateGraph,
    question: str = TASK,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> RunStats:
    """One full run through the tool loop with `ainvoke`, timed.

    Same measurement as `direct_run.run_once`: wall time over every model call
    and tool call, from a monotonic clock.
    """
    start = clock()
    final = await investigator.ainvoke({"messages": [HumanMessage(question)]})
    return stats_from(final["messages"], clock() - start)


async def run_session(runs: int, model_name: str) -> None:
    """Warm up, then run `runs` measured runs on one persistent MCP session."""
    from langchain_ollama import ChatOllama

    t0 = time.perf_counter()
    async with persistent_tools() as tools:
        startup = time.perf_counter() - t0
        investigator = build_investigator(
            ChatOllama(model=model_name).bind_tools(tools), tools
        )
        print(f"model: {model_name}   transport: MCP over stdio, one persistent session")
        print(f"task: {TASK}")
        print(f"server startup (spawn, initialize, list tools): {startup:.2f}s, once")
        print(HEADER)
        print(format_row("warm-up", await arun_once(investigator)), " (excluded)")
        stats = []
        for i in range(1, runs + 1):
            row = await arun_once(investigator)
            stats.append(row)
            print(format_row(str(i), row))
    s = summarise(stats)
    print(
        f"\ncalled the tool: {s['called_tool']}/{s['runs']}   "
        f"correct: {s['correct']}/{s['runs']}   "
        f"hit turn cap: {s['hit_turn_cap']}   "
        f"median {s['median_seconds']:.2f}s   "
        f"mean tokens in/out: {s['mean_input_tokens']}/{s['mean_output_tokens']}"
    )


def main(argv: list[str] | None = None) -> None:
    """Parse `mode [runs]`, read `.env`, run. The only place that reads `.env`."""
    from dotenv import load_dotenv

    load_dotenv()
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in MODES:
        sys.exit(f"usage: python -m app.mcp_run {{{'|'.join(MODES)}}} [runs]")
    runs = int(args[1]) if len(args) > 1 else DEFAULT_RUNS
    asyncio.run(run_session(runs, os.environ["SMALL_MODEL"]))


if __name__ == "__main__":
    main()
