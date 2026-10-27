"""Day 2, step 5: connect to the MCP deploy server and get LangChain tools.

What it does
    `persistent_tools()` starts the server as a child process, opens ONE MCP
    session, and yields the server's tools as LangChain tools. The session
    stays open for the whole `async with` block, so every tool call reuses it.
    The process is stopped when the block ends, even on an error.

Two ways to hold the connection
    `persistent_tools()` opens ONE session and keeps it for the whole block:
    the realistic best case, a client that connects once.
    `per_call_tools()` is the adapter's DEFAULT: no session is held, and each
    tool call opens a new one. Over stdio that starts the server process again
    on every call. It is a large, constant cost per call, and it is what a
    team gets by following the adapter's basic example. Measuring both shows
    the cost of the default and what a persistent session buys back.

Why the tools are async-only
    `langchain-mcp-adapters` builds each tool with a coroutine and no sync
    function, because talking to the server is I/O. A graph that uses them must
    run with `ainvoke`. The sync `invoke` would raise.

Environment
    The child gets a small default environment (no API keys, no home-dir
    secrets), which is what we want for a tool server. It is started with the
    same Python as this process, from the project root, so `app` is importable
    and nothing depends on the caller's working directory.
"""

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from app.paths import ROOT

# code/ : the folder that holds `app/` (see app/paths.py).
PROJECT_ROOT = ROOT
SERVER_NAME = "deploys"


def server_connection() -> dict:
    """The stdio connection config for the deploy server."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "app.mcp_server"],
        "cwd": str(PROJECT_ROOT),
    }


@asynccontextmanager
async def per_call_tools() -> AsyncIterator[list[BaseTool]]:
    """The adapter's default: each tool call starts and stops its own session.

    Listing the tools opens one session (one server process) once. After that
    every call spawns a fresh server, runs, and shuts it down. Nothing is held
    between calls, so there is nothing to clean up on exit, and a long-running
    client pays the spawn cost on every call.
    """
    client = MultiServerMCPClient({SERVER_NAME: server_connection()})
    yield await client.get_tools()


@asynccontextmanager
async def persistent_tools() -> AsyncIterator[list[BaseTool]]:
    """Start the server, hold one session open, and yield its tools.

    Startup (spawn, initialize, list tools) happens on entry. Callers that
    measure time should time the entry separately from the calls.
    """
    client = MultiServerMCPClient({SERVER_NAME: server_connection()})
    async with client.session(SERVER_NAME) as session:
        yield await load_mcp_tools(session)
