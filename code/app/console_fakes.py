"""Scripted stand-ins for a fake run of the console: no model, no database.

Why a module of its own: the console serves HTTP and runs real graphs. Its fake mode
(scripted replies per scenario, a fake tool-calling model, invented recalled
incidents) is demo and test data. Mixing it into the server module made that file
long and hid what it does. Nothing here is imported by the real run path.

Invented data is marked as invented: the recalled incidents have `FAKE-` IDs, and the
scripted tool model reports no token usage, because made-up numbers would look real.
"""

import json
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage

from app.events import Emit
from app.triage import Ask, Prior


Scenario = Literal[
    "stay_small",
    "retry_then_small",
    "escalate",
    "large_fails",
    "tool_lookup",
    "tool_wrong_window",
]

# What the page shows for the recall step in a fake run: fixed, invented, and
# marked as such by its IDs. A fake run must never need a database or a model.
FAKE_RECALL: tuple[Prior, ...] = (
    Prior("FAKE-1", "auth", "sev2", "Latency rose after a certificate rotation.", "Peers lacked the new intermediate authority.", "Test trust chains before every rotation."),
    Prior("FAKE-2", "auth", "sev3", "Slow logins after a rotation.", "Trust was not tested before cutover.", "Add a trust check to the rotation runbook."),
    Prior("FAKE-3", "gateway", "sev2", "Saturation at the daily peak.", "The autoscaler floor was too low.", "Raise the autoscaler floor ahead of the daily peak."),
)


def scripted_reply(confidence: float | str, severity: str = "sev2") -> str:
    """A Triage JSON text. `confidence` may be a string to make it invalid."""
    return json.dumps(
        {
            "severity": severity,
            "component": "auth-service",
            "next_action": "check the last deploy and the error rate",
            "confidence": confidence,
        }
    )


# Scripted replies per scenario and tier. They exist so you can watch a retry
# or an escalation on demand, without waiting for a real model to slip.
# `75` is the Day 1 failure: legal JSON, illegal value.
SCENARIOS: dict[str, dict[str, list[str]]] = {
    "stay_small": {"small": [scripted_reply(0.9)], "large": []},
    "retry_then_small": {"small": [scripted_reply(75), scripted_reply(0.85)], "large": []},
    "escalate": {"small": [scripted_reply(0.3)], "large": [scripted_reply(0.92, "sev1")]},
    "large_fails": {
        "small": [scripted_reply(75), scripted_reply(75)],
        "large": ["not json", "still not json"],
    },
    # Scenarios with the tool step. The tool and the ToolNode are the real
    # ones, reading data/deploys.json. Only the tool-calling model is scripted.
    "tool_lookup": {"small": [scripted_reply(0.9)], "large": []},
    "tool_wrong_window": {"small": [scripted_reply(0.9)], "large": []},
}

# The look-back window the scripted tool model asks for, per scenario.
# `tool_wrong_window` mirrors what the real 1.5B model did on a 24 hour
# question: it sent a legal but wrong 1 hour window, so the tool returns [].
TOOL_HOURS: dict[str, int] = {"tool_lookup": 24, "tool_wrong_window": 1}


class ScriptedToolModel:
    """A tool-calling model that asks for one deploy lookup, then finishes.

    It reports no token usage on purpose: invented numbers would look real.
    The page then shows the token fields as unknown.
    """

    def __init__(self, hours: int) -> None:
        self._hours = hours
        self._asked = False

    def invoke(self, messages: list) -> AIMessage:
        if self._asked:
            return AIMessage(content="done")
        self._asked = True
        call = {
            "name": "get_recent_deploys",
            "args": {"service": "auth", "hours": self._hours},
            "id": "fake-1",
            "type": "tool_call",
        }
        return AIMessage(content="", tool_calls=[call])


FAKE_QUESTION = "Fake run: replies are scripted, no model is called."


def fake_ask(scenario: Scenario, sleep: Callable[[float], None], delay: float) -> Ask:
    """Build an `ask` that plays back one scenario's scripted replies.

    The model name tells the tiers apart: the fake models are "small" and
    "large". `delay` is a pause per call so the animation is watchable. Tests
    pass 0. Running past the script means the graph made more calls than the
    scenario allows, which is a bug in the scenario, so it raises.
    """
    queues = {tier: iter(replies) for tier, replies in SCENARIOS[scenario].items()}

    def ask(model: str, prompt: str) -> str:
        sleep(delay)
        return next(queues[model])

    return ask


def inject_bad_first_reply(ask: Ask, small_model: str, emit: Emit) -> Ask:
    """Wrap `ask` so the small model's first reply is an illegal one.

    The reply is well-formed JSON with `confidence: 75`, the Day 1 slip. It
    passes the grammar and fails Pydantic, so the validate-retry path runs for
    real: the real error text goes into the retry prompt, and the real model
    answers it. Only that first reply is replaced, and the real model is not
    called for it. An event marks the swap, so nobody mistakes it for a
    natural failure. Later calls pass through untouched.

    A hunt must not use this: the point of a hunt is to find real slips.
    """
    state = {"injected": False}

    def wrapped(model: str, prompt: str) -> str:
        if model == small_model and not state["injected"]:
            state["injected"] = True
            emit("fault_injected", model=model, field="confidence", value=75)
            return scripted_reply(75)
        return ask(model, prompt)

    return wrapped
