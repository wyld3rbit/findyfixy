"""Day 2, step 5: the deploy tool, served over MCP.

What it does
    A FastMCP server that exposes `get_recent_deploys` over stdio. It calls the
    SAME function as the direct path (`app.tools.get_recent_deploys`), so the
    behaviour and the validation are identical. Only the transport differs:
    JSON-RPC over a child process's stdin and stdout, instead of a function
    call in the same process.

Why the schema is built from the direct tool's own fields
    The model sees the tool's name, description and argument schema in its
    prompt, on every turn. If the MCP schema were worded or bounded
    differently, the token comparison would measure that difference and not
    the cost of MCP. So the argument descriptions and bounds are read from
    `DeploysArgs`, and `tests/test_mcp.py` compares the two schemas.

How it runs
    The client starts it as `python -m app.mcp_server` and talks over stdio
    (see app/mcp_client.py). Run by hand it waits for JSON-RPC on stdin, which
    is not interactive. stdout belongs to the protocol: nothing else may print
    there, or the client sees garbage. Log to stderr.

What changes at scale
    - One process per server per client. Many clients means many processes, or
      the HTTP transport (and the starlette and uvicorn packages this project
      deliberately does not install, see pyproject.toml).
    - A server is a separate thing to deploy, patch and authenticate. The data
      path here reads a local file, so there is no credential to protect. A real
      tool would hold credentials, and that is the case for MCP: the model-facing
      client never sees them (Day 7).
"""

from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.tools import (
    DEFAULT_DEPLOYS_PATH,
    MAX_HOURS,
    DeploysArgs,
    Service,
    get_recent_deploys,
)

TOOL_NAME = "get_recent_deploys"
# Same text the direct tool shows the model (see make_deploys_tool in tools.py).
TOOL_DESCRIPTION = "List recent deploys of one service, newest first."


def build_server(path: Path = DEFAULT_DEPLOYS_PATH) -> FastMCP:
    """Build the server. Nothing runs until `.run()`. `path` is injected for tests.

    FastMCP derives the tool's argument schema from the function signature, so
    the annotations below must carry the same constraints as `DeploysArgs`.
    """
    fields = DeploysArgs.model_fields
    # WARNING, not the default INFO: the server logs every request to stderr,
    # which would add noise to every measured run.
    server = FastMCP("deploys", log_level="WARNING")

    @server.tool(name=TOOL_NAME, description=TOOL_DESCRIPTION)
    def get_recent_deploys_tool(
        service: Annotated[Service, Field(description=fields["service"].description)],
        hours: Annotated[
            int, Field(ge=1, le=MAX_HOURS, description=fields["hours"].description)
        ],
    ) -> str:
        """Run the shared function. Returns JSON text, like the direct tool."""
        return get_recent_deploys(service, hours, path=path)

    return server


def main() -> None:
    """Serve over stdio until the client closes the connection."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
