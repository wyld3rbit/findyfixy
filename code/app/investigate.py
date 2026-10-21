"""Day 2: the tool-calling loop, as a LangGraph graph with a `ToolNode`.

What it does
    `build_investigator(model, tool)` returns a compiled graph with two
    nodes. `agent` asks the model. `tools` is a LangGraph `ToolNode` that runs
    whatever tools the model requested. The edge after `agent` goes to `tools`
    if the model asked for a tool and the turn budget allows it, and ends
    otherwise.

        question -> agent --tool requested--> tools --+
                      ^                               |
                      +-------------------------------+
                      |
                      +--answer, or out of turns--> END

    `analyse` reads the finished conversation and reports tokens, tool calls
    and evidence. `format_evidence` turns the tool calls into text that a later
    step (the Triage prompt) can quote.

Why a graph for a loop this small
    The loop alone is about 25 lines of plain Python, and an earlier version
    of this module was exactly that. The graph pays off because the tool step
    becomes a named node: it can be traced (Day 6), a policy check can sit
    between the model's request and the execution (Day 7), and a human can
    approve before a risky tool runs (Day 9). Inside a function, each of those
    is a rewrite. As nodes, each is one more node and one more edge.

Why the turn budget is in the graph state
    A model that keeps asking for tools, or keeps sending bad arguments, would
    loop forever and burn tokens. `turns` counts model calls. When it reaches
    `max_turns` the edge ends the run even if a tool is still requested, and
    `analyse` reports `hit_turn_cap`. The cap is a cost bound, so it is
    recorded, not hidden. Note: that last requested tool is NOT executed.

Why the model is injected
    `build_investigator` takes a model with its tools already bound (anything
    with `invoke(messages) -> AIMessage`). Tests pass a scripted fake.
    Production passes `ChatOllama(...).bind_tools([tool])`. No client is built
    here and nothing runs at import.

The model never executes anything
    It emits a tool name and arguments. They come from a model that has read
    untrusted text, so `ToolNode(handle_tool_errors=True)` (see tools.py) turns
    invalid arguments into an error message the model can read and correct.
    Legal but wrong arguments (the 1.5B model sent `hours: 1` for 24) pass
    validation. Nothing in this loop can tell. The evidence text therefore
    always includes the arguments, so a reader sees what window was asked for.

What changes at scale
    - MCP tools are async-only. A graph that uses them must run with
      `ainvoke`. The sync `invoke` used here would raise on them.
    - Parallel tool calls: `ToolNode` runs several calls from one message in
      parallel. Each is one entry in `calls` here, but the cost is the max of
      them, not the sum.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, TypedDict, get_args

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from app.tools import Service, make_tools_node

# Model calls per run. Four is enough for ask, fix a bad argument, retry,
# answer. More only helps a model that is already lost.
MAX_TURNS = 4

# Tool output goes back into the model's context. Clip it in the evidence text
# so one huge reply cannot flood a later prompt.
_MAX_RESULT_CHARS = 600


# What the investigate step tells the model. It may look up deploys before
# triage, or skip. The service list comes from the tool's own argument type,
# so the prompt and the validation cannot drift apart.
INVESTIGATE_PROMPT = (
    "You are an incident triage assistant. Before triage you may look up recent "
    "deploys with the get_recent_deploys tool. Known services: {services}. If the "
    "report involves one of them, look up its deploys over a window that fits "
    "the timeline in the report. If no lookup helps, reply: no lookup needed.\n\n"
    "Report: {question}"
)


def investigation_prompt(question: str) -> str:
    """Wrap a report in the investigate instructions. Pure."""
    return INVESTIGATE_PROMPT.format(
        services=", ".join(get_args(Service)), question=question
    )


class ToolModel(Protocol):
    """A chat model with its tools already bound: messages in, one message out.

    `ChatOllama(...).bind_tools([tool])` fits. So does a scripted fake in
    tests.
    """

    def invoke(self, messages: list[BaseMessage]) -> AIMessage: ...


class InvestigationState(TypedDict, total=False):
    """Graph state: the conversation, and how many model calls were made.

    `add_messages` appends, so each node returns only the new message(s).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    turns: int


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the model made, with what came back."""

    name: str
    args: dict
    result: str
    error: bool


@dataclass(frozen=True)
class Investigation:
    """What a finished loop did and cost. Built by `analyse`.

    Token fields are None when the model did not report usage on some turn.
    A partial sum would look like a real number, so it is dropped instead.
    """

    input_tokens: int | None
    output_tokens: int | None
    turns: int
    calls: tuple[ToolCallRecord, ...]
    hit_turn_cap: bool
    answer: str


def build_investigator(
    model: ToolModel,
    tool: BaseTool | Sequence[BaseTool],
    *,
    max_turns: int = MAX_TURNS,
) -> CompiledStateGraph:
    """Compile the agent and tools loop. Nothing runs until `.invoke()`.

    Start it with `{"messages": [HumanMessage(question)]}`. `tool` is one tool
    or a list. Tools from an MCP server are async-only, so a graph that uses
    them must be run with `.ainvoke()`.
    """

    def agent(state: InvestigationState) -> InvestigationState:
        """Ask the model once and count the turn."""
        reply = model.invoke(state["messages"])
        return {"messages": [reply], "turns": state.get("turns", 0) + 1}

    def after_agent(state: InvestigationState) -> Literal["tools", "__end__"]:
        """Run tools if requested and budget remains. Pure.

        The `Literal` return type is load-bearing: LangGraph reads it to draw
        the edge and to know where this branch can go.
        """
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        if wants_tools and state.get("turns", 0) < max_turns:
            return "tools"
        return END

    g = StateGraph(InvestigationState)
    g.add_node("agent", agent)
    g.add_node("tools", make_tools_node(tool))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", after_agent)
    g.add_edge("tools", "agent")
    return g.compile()


def content_text(content: str | list) -> str:
    """Plain text of a message's content. Pure.

    A direct tool returns a string. An MCP tool returns a list of content
    blocks, `[{"type": "text", "text": ...}]`. The model is shown only their
    text (ChatOllama drops the block metadata), so the evidence must be the
    text too. Without this, the Triage prompt would carry the Python repr of
    the list: noise, extra tokens, and a prompt that differs between transports.
    Non-text blocks are shown as their repr, so nothing is silently dropped.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif isinstance(block, str):
            parts.append(block)
        else:
            parts.append(repr(block))
    return "".join(parts)


def _sum_tokens(replies: list[AIMessage], key: str) -> int | None:
    """Total one usage field across turns, or None if any turn lacked it."""
    total = 0
    for reply in replies:
        usage = reply.usage_metadata
        if not usage or key not in usage:
            return None
        total += usage[key]
    return total


def analyse(messages: list[BaseMessage]) -> Investigation:
    """Summarise a finished conversation. Pure.

    A tool call is matched to its result by `tool_call_id`. A call with no
    result (the run ended on the turn cap) is reported with an empty result,
    marked as an error, so it is never mistaken for "no data".
    """
    replies = [m for m in messages if isinstance(m, AIMessage)]
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    calls: list[ToolCallRecord] = []
    for reply in replies:
        for call in reply.tool_calls:
            res = results.get(call["id"])
            calls.append(
                ToolCallRecord(
                    name=call["name"],
                    args=dict(call["args"]),
                    result="" if res is None else content_text(res.content),
                    error=res is None or res.status == "error",
                )
            )
    last = replies[-1] if replies else None
    return Investigation(
        input_tokens=_sum_tokens(replies, "input_tokens"),
        output_tokens=_sum_tokens(replies, "output_tokens"),
        turns=len(replies),
        calls=tuple(calls),
        hit_turn_cap=bool(last and last.tool_calls),
        answer="" if last is None or last.tool_calls else str(last.content),
    )


def format_args(args: dict) -> str:
    """Compact, stable text for a call's arguments: `service=auth,hours=1`."""
    return ",".join(f"{k}={args[k]}" for k in sorted(args))


def format_evidence(calls: tuple[ToolCallRecord, ...]) -> str:
    """Tool calls and results as text a prompt can quote. Pure.

    Always names the arguments. A result of `[]` for `hours=1` means "nothing
    in the last hour", not "nothing happened", and only the arguments say which.
    Returns "" when no tool was called.
    """
    lines = []
    for c in calls:
        status = "ERROR" if c.error else "result"
        text = c.result[:_MAX_RESULT_CHARS]
        lines.append(f"{c.name}({format_args(c.args)}) {status}: {text}")
    return "\n".join(lines)
