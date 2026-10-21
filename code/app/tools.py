"""Day 2: a fake deploy-history tool, bound directly into a LangGraph ToolNode.

What it does
    `get_recent_deploys(service, hours)` returns deploys of one service in the
    last N hours, from a local synthetic file. `make_deploys_tool` wraps it as
    a LangChain tool so a model can request it and a `ToolNode` can run it.

Why the arguments are validated
    The model never executes anything. It emits a tool name and arguments, and
    those arguments come from a model that has read untrusted text (an alert,
    a customer message). Treat them as input from the internet. `DeploysArgs`
    rejects an unknown service, a non-positive window and an absurd window.
    `service` is a `Literal`, so it cannot be steered to another file or key.

Why records hold `hours_ago`, not timestamps
    No clock in the code means no flaky tests and no data that goes stale. The
    cost: the file cannot say "when" in absolute terms. Fine for a fake. A real
    tool would query a deploy API with timestamps and its own credentials.
    Credentials belong to this code, never to the model (Day 7).

Why the file is read on each call
    No work at import time. For a file this small the cost is microseconds.
    At scale this is a network call, so add a timeout and cache with care.

Failure modes
    - Bad arguments raise Pydantic `ValidationError`. `make_tools_node`
      turns that into a ToolMessage with the error text, so the model can
      correct itself. Without that, one bad call crashes the whole run.
    - Missing or malformed data file raises. That is an operator fault, not
      something the model can fix, so it is not caught.
"""

import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Resolved from this file, so it works from any working directory.
DEFAULT_DEPLOYS_PATH = Path(__file__).resolve().parents[1] / "data" / "deploys.json"

# One week. A longer window is almost certainly a mistake, and an unbounded
# window on a real API is an expensive query.
MAX_HOURS = 168

Service = Literal["auth", "billing", "gateway", "voice-verification", "notifications"]


class DeploysArgs(BaseModel):
    """The tool's argument contract. Also the schema the model sees."""

    service: Service = Field(description="service whose deploys to list")
    hours: int = Field(
        ge=1, le=MAX_HOURS, description="look-back window in hours (1 to 168)"
    )


def get_recent_deploys(service: str, hours: int, *, path: Path) -> str:
    """Return JSON text: this service's deploys within `hours`, newest first.

    Arguments are validated here too, not only by the tool wrapper, so a
    direct call is just as safe as one from a model. Returns a JSON string
    because that is what goes back into the model's context.
    """
    args = DeploysArgs(service=service, hours=hours)
    records = json.loads(path.read_text())
    hits = [
        r
        for r in records
        if r["service"] == args.service and r["hours_ago"] <= args.hours
    ]
    hits.sort(key=lambda r: r["hours_ago"])
    return json.dumps(hits)


def make_deploys_tool(path: Path = DEFAULT_DEPLOYS_PATH) -> StructuredTool:
    """Wrap `get_recent_deploys` as a tool. `path` is injected for tests."""

    def run(service: str, hours: int) -> str:
        return get_recent_deploys(service, hours, path=path)

    return StructuredTool.from_function(
        func=run,
        name="get_recent_deploys",
        description="List recent deploys of one service, newest first.",
        args_schema=DeploysArgs,
    )


def make_tools_node(tool: StructuredTool) -> ToolNode:
    """A ToolNode that reports bad arguments to the model instead of crashing.

    `handle_tool_errors=True` catches the tool's exception and returns it as a
    ToolMessage. The model sees the validation text and can retry. Bound this
    with a max-turns limit in the loop, or a stubborn model loops forever.
    """
    return ToolNode([tool], handle_tool_errors=True)
