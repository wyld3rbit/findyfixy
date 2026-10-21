"""The picture must follow reality: a regression check for the live visualizer.

What this covers
    A live run's events arrive when the real work happens (each model call is a
    request to Ollama). The page animates them, and an earlier version added a
    fixed pause and a fixed move to every step. Real calls are often shorter
    than that, so the picture fell further behind with each step: on a 4.6s run
    it showed the small model starting 2s late and the run finishing 4.6s late.
    Ollama's log lines then appeared while the page still showed an earlier
    node.

How it checks
    `tests/js/replay.js` replays a timeline through the page's own animation
    code at real speed, with a stub DOM, and reports when each event was shown.
    The timeline below has the shape of a real run (a tool step, then the small
    model, then an escalation to the large model, with several events at the
    same instant). Every event must be shown within half a second of arriving.

Needs Node. It is not a project dependency, so the test skips without it.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "architecture.html"
REPLAY = ROOT / "tests" / "js" / "replay.js"

# Seconds a shown event may trail the moment it arrived.
MAX_LAG = 0.5

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")


def timeline(mode: str) -> list[dict]:
    """A run shaped like a real live one. `t` is seconds since the run started."""

    def ev(t, kind, **data):
        return {"kind": kind, "t": t, "data": data}

    return [
        ev(0.00, "run_started", mode=mode, tools=True, forced=True, question="q",
           small_model="s", large_model="l", threshold=1.01),
        ev(0.00, "node_entered", node="investigate", model="s"),
        ev(0.00, "model_request", model="s", attempt=1),
        ev(0.90, "model_reply", model="s", attempt=1, reply="", served_by="s", mismatch=False,
           input_tokens=900, output_tokens=30),
        ev(0.90, "tool_call", name="get_recent_deploys", args={"service": "billing", "hours": 24}),
        ev(0.92, "tool_result", name="get_recent_deploys", result="[]", error=False),
        ev(0.92, "model_request", model="s", attempt=2),
        ev(1.60, "model_reply", model="s", attempt=2, reply="no lookup needed", served_by="s",
           mismatch=False, input_tokens=950, output_tokens=8),
        ev(1.60, "node_exited", node="investigate", model="s", calls=1, turns=2,
           input_tokens=1850, output_tokens=38, hit_turn_cap=False, served_by=["s"]),
        ev(1.60, "node_entered", node="small", model="s"),
        ev(1.60, "model_request", model="s", attempt=1),
        ev(2.30, "model_reply", model="s", attempt=1, reply="{}", served_by="s", mismatch=False),
        ev(2.30, "validation", model="s", attempt=1, ok=True, error=None),
        ev(2.30, "node_exited", node="small", model="s", triage={"severity": "sev3"}),
        ev(2.30, "route", confidence=0.9, threshold=1.01, target="large"),
        ev(2.30, "node_entered", node="large", model="l"),
        ev(2.30, "model_request", model="l", attempt=1),
        ev(4.60, "model_reply", model="l", attempt=1, reply="{}", served_by="l", mismatch=False),
        ev(4.60, "validation", model="l", attempt=1, ok=True, error=None),
        ev(4.60, "node_exited", node="large", model="l", triage={"severity": "sev3"}),
        ev(4.60, "run_finished", model_used="l",
           triage={"severity": "sev3", "component": "c", "next_action": "n", "confidence": 0.9}),
    ]


def burst_timeline() -> list[dict]:
    """Eight parallel tool calls reported at one instant, then the small model.

    A burst this size cannot be shown move by move without falling behind, so it
    only passes if the page skips moves and pauses while it catches up.
    """

    def ev(t, kind, **data):
        return {"kind": kind, "t": t, "data": data}

    out = [
        ev(0.0, "run_started", mode="live", tools=True, forced=False, question="q",
           small_model="s", large_model="l", threshold=0.7),
        ev(0.0, "node_entered", node="investigate", model="s"),
    ]
    for i in range(8):
        out.append(ev(1.0, "tool_call", name="get_recent_deploys", args={"service": "auth", "hours": i + 1}))
        out.append(ev(1.0, "tool_result", name="get_recent_deploys", result="[]", error=False))
    out += [
        ev(1.0, "node_exited", node="investigate", model="s", calls=8, turns=2,
           input_tokens=None, output_tokens=None, hit_turn_cap=False, served_by=[]),
        ev(1.0, "node_entered", node="small", model="s"),
        ev(1.0, "model_request", model="s", attempt=1),
    ]
    return out


def replay(tmp_path, events: list[dict], tail: float = 1.0) -> list[dict]:
    """Run the replay and return one row per event: when it arrived and when shown."""
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events))
    env = {k: v for k, v in os.environ.items() if k != "NODE_OPTIONS"}
    out = subprocess.run(["node", str(REPLAY), str(PAGE), str(path), str(tail)],
                         capture_output=True, text=True, timeout=40, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def late(rows: list[dict]) -> list[tuple]:
    return [(r["kind"], r["node"], round(r["started"] - r["real"], 2))
            for r in rows if r["started"] - r["real"] > MAX_LAG]


@pytest.fixture(scope="module")
def live_rows(tmp_path_factory):
    return replay(tmp_path_factory.mktemp("lag"), timeline("live"))


def test_a_live_run_is_shown_as_it_happens_not_seconds_late(live_rows):
    assert len(live_rows) == len(timeline("live"))
    assert late(live_rows) == []


def test_the_run_finishes_on_screen_when_it_finishes_in_reality(live_rows):
    last = live_rows[-1]
    assert last["kind"] == "run_finished" and last["started"] - last["real"] <= MAX_LAG


def test_a_large_burst_is_caught_up_not_replayed_move_by_move(tmp_path):
    rows = replay(tmp_path, burst_timeline())
    assert late(rows) == []
