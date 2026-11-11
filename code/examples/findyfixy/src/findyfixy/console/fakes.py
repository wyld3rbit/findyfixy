"""Scripted stand-ins for a fake run of the console: no model, no database.

Why a module of its own: the console serves HTTP and runs real graphs. Its fake mode
(scripted replies per scenario, a fake tool-calling model, invented recalled
incidents) is demo and test data. Mixing it into the server module made that file
long and hid what it does. Nothing here is imported by the real run path.

Invented data is marked as invented: the recalled incidents have `FAKE-` IDs, and the
scripted tool model reports no token usage, because made-up numbers would look real.

The fake eval run (`fake_eval`) is the same idea one level up. The harness runner,
the FindyFixy adapter and the graph are the real ones. Only the parts that would
need a model or a database are scripted: the replies, the recalled incidents, the
judge and the embedder. The cases are invented and say so (`FAKE-` IDs), and the
system under test describes itself as `data: FAKE`, so the page marks every score
from such a run as fake.
"""

from collections.abc import Callable, Sequence
from itertools import cycle
import json
from typing import Any, Literal, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from findyfixy.console.eval_run import EvalResult, EvalSuite
from findyfixy.evals.adapter import FindyFixy, Invokable
from findyfixy.events import Emit
from findyfixy.triage.graph import Recall, build_graph
from findyfixy.triage.record import Ask, Prior
from wyld3rness.eval.case import Case
from wyld3rness.eval.events import EvalEmit
from wyld3rness.eval.metrics import ActionRating, Claims, Questions, Verdict
from wyld3rness.eval.runner import run_eval

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
    Prior(
        "FAKE-1",
        "auth",
        "sev2",
        "Latency rose after a certificate rotation.",
        "Peers lacked the new intermediate authority.",
        "Test trust chains before every rotation.",
    ),
    Prior(
        "FAKE-2",
        "auth",
        "sev3",
        "Slow logins after a rotation.",
        "Trust was not tested before cutover.",
        "Add a trust check to the rotation runbook.",
    ),
    Prior(
        "FAKE-3",
        "gateway",
        "sev2",
        "Saturation at the daily peak.",
        "The autoscaler floor was too low.",
        "Raise the autoscaler floor ahead of the daily peak.",
    ),
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

    def invoke(self, messages: list[BaseMessage], /) -> AIMessage:
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


# ---- Fake eval run ------------------------------------------------------------

# The scripted scenario each fake case plays, in turn. Mostly routine, with an
# escalation, a retry and one case whose models never give a valid record, so a
# full fake run shows every path, including a run error the gate must catch.
FAKE_EVAL_SCENARIOS: tuple[Scenario, ...] = (
    "stay_small",
    "escalate",
    "retry_then_small",
    "stay_small",
    "stay_small",
    "escalate",
    "stay_small",
    "retry_then_small",
    "stay_small",
    "large_fails",
)
# The invented set mirrors the golden set's shape (20 twin, 10 no_twin), so a
# full fake run exercises the gate. Its size is this data's own, not a limit.
FAKE_EVAL_CASES = 30
_FAKE_TWIN_CASES = 20

T = TypeVar("T", bound=BaseModel)


def fake_cases(n: int = FAKE_EVAL_CASES) -> list[Case]:
    """Invented golden cases, shaped like FindyFixy's: 20 twin, then 10 no_twin.

    Twin cases name `FAKE-1` as their twin, which `FAKE_RECALL` returns, so the
    twin recall rate is computed for real on invented data. Expected severities
    alternate, so some scripted answers match and some miss.
    """
    return [
        Case(
            id=f"FAKE-g{i:02d}",
            group="twin" if i <= _FAKE_TWIN_CASES else "no_twin",
            question=f"Fake alert {i}: invented text, no model is called.",
            reference="Root cause: invented for the demo. Remediation: invented for the demo.",
            expected={"severity": "sev2" if i % 3 else "sev1"},
            meta={"source_id": f"FAKE-SRC-{i:02d}", "twin_id": "FAKE-1" if i <= _FAKE_TWIN_CASES else None},
        )
        for i in range(1, n + 1)
    ]


class ScriptedJudge:
    """A judge that answers by schema from a fixed script. No model.

    Verdicts follow a fixed cycle (two supported, one not), so the scores vary
    and the page has something to show, but they measure nothing. One instance
    per run, called from one thread (the console runs one worker).
    """

    def __init__(self) -> None:
        """Start the verdict and rating cycles at the beginning."""
        self._supported = cycle((True, True, False))
        self._rating = cycle(("helpful", "partial", "helpful", "unhelpful"))

    def ask(self, prompt: str, schema: type[T]) -> T:
        """A canned reply of the right schema."""
        # model_validate builds the reply as the asked-for type, so the return
        # type follows `schema` without a cast, and pydantic checks the values.
        if schema is Claims:
            return schema.model_validate(
                {"claims": ["FAKE claim: the component is named.", "FAKE claim: an action is given."]}
            )
        if schema is Questions:
            return schema.model_validate(
                {"questions": ["FAKE question one?", "FAKE question two?", "FAKE question three?"]}
            )
        if schema is ActionRating:
            return schema.model_validate({"rating": next(self._rating), "reason": "FAKE rating"})
        if schema is Verdict:
            return schema.model_validate({"supported": next(self._supported), "reason": "FAKE verdict"})
        raise TypeError(f"the scripted judge has no reply for {schema.__name__}")


def fake_embed(texts: Sequence[str]) -> list[list[float]]:
    """A deterministic stand-in for the embedder: a 2-d vector from each text's length."""
    return [[1.0, (len(t) % 7) / 10] for t in texts]


class FakeFindyFixy(FindyFixy):
    """The real adapter, describing itself as fake so a viewer can say so."""

    def describe(self) -> dict[str, Any]:
        """The adapter's settings plus `data: FAKE`."""
        return {**super().describe(), "data": "FAKE"}


def fake_eval(sleep: Callable[[float], None], delay: float) -> EvalSuite:
    """An eval run on invented cases with scripted models: no model, no database.

    The real runner, adapter and graph run, so the events on the page are the
    ones the code really emits. Each case compiles a graph whose `ask` plays the
    next scenario in `FAKE_EVAL_SCENARIOS`; `delay` per model call keeps it
    watchable (tests pass 0). The report is never written. The suite's size is
    the number of invented cases.
    """
    cases = fake_cases()

    def run(n: int, *, emit: Emit, on_event: EvalEmit, should_stop: Callable[[], bool]) -> EvalResult:
        """Run the first `n` invented cases and return the report."""
        scenarios = cycle(FAKE_EVAL_SCENARIOS)

        def make_graph(recall: Recall) -> Invokable:
            """A graph for one case, playing the next scripted scenario."""
            return build_graph(
                fake_ask(next(scenarios), sleep, delay),
                small_model="small",
                large_model="large",
                emit=emit,
                recall=recall,
            )

        sut = FakeFindyFixy(
            make_graph=make_graph,
            candidates=lambda question, k: list(FAKE_RECALL[:k]),
            small_model="small",
            large_model="large",
        )
        report = run_eval(
            cases[:n],
            sut=sut,
            judge=ScriptedJudge(),
            embed=fake_embed,
            provenance={"data": "FAKE"},
            on_event=on_event,
            should_stop=should_stop,
        )
        return EvalResult(report=report, subset=n < len(cases))

    return EvalSuite(run=run, size=len(cases))
