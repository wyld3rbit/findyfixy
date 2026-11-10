"""The architecture page must say only what the code makes true.

What this covers
    The page (docs/architecture.html) draws the system and animates it from
    events. Two kinds of drift would make it lie without anyone noticing:

    - Status drift. A box marked built whose code is gone, a box marked
      planned whose code already exists, or a graph node with no box at all.
      Each node carries `data-code`: a graph node (`graph:small`), a module
      path relative to code/, or `none`. These tests resolve it against the
      real compiled graph and the file tree.
    - Event drift. The graph or the harness emits a kind the page has no
      handler for, so the event is silently not drawn. The page keeps one
      handler table (`DESCRIBE`) per event source. These tests parse it and
      compare it with the vocabularies the code declares.

    A last test plays a whole fake eval stream, as the console really emits
    it, through the page's own script (tests/js/replay.js) and checks that
    every event is shown and the verdict line is right.

Why parse the page instead of running it in a browser
    The facts checked here are static: attributes and a table of names. A
    regex over a file we own is exact and runs in milliseconds. The replay
    covers behaviour, with a stub DOM, and needs Node (skipped without it).
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import get_args

import pytest

from app.console import EvalRequest, RunHub, wire
from app.console_fakes import ScriptedToolModel, fake_ask
from app.events import EventKind
from app.graph import build_graph
from app.investigate import build_investigator
from app.paths import DOCS, ROOT
from app.tools import make_deploys_tool
from harness.eval.events import EvalEventKind

PAGE = DOCS / "architecture.html"
REPLAY = ROOT / "tests" / "js" / "replay.js"

_NODE = re.compile(r'<g class="node"([^>]*)>(.*?)</g>', re.S)
_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
_NAME = re.compile(r'<text class="t"[^>]*>([^<]+)</text>')


def page_nodes(page: str) -> list[dict[str, str]]:
    """Every node on the page: its name, status and data-code. Pure."""
    nodes = []
    for attrs, body in _NODE.findall(page):
        found = dict(_ATTR.findall(attrs))
        nodes.append({
            "name": _NAME.search(body).group(1),
            "status": found.get("data-status", ""),
            "code": found.get("data-code", ""),
        })
    return nodes


def handler_table(page: str) -> dict[str, set[str]]:
    """The keys of each source block in the page's `DESCRIBE` table. Pure.

    The table's layout is fixed: `const DESCRIBE = {`, then two-space indented
    `source: {` blocks whose keys sit at four spaces. Deeper lines (a handler's
    body) are not keys.
    """
    body = page.split("const DESCRIBE = {\n", 1)[1].split("\n};\n", 1)[0]
    table: dict[str, set[str]] = {}
    for block in re.finditer(r"^  (\w+): \{\n(.*?)^  \},$", body, re.S | re.M):
        table[block.group(1)] = set(re.findall(r"^    (\w+): ", block.group(2), re.M))
    return table


def graph_nodes() -> set[str]:
    """Every node of the real graph, compiled with every optional step and fake parts.

    Recall and the tool step are both switched on, so no optional node is
    missed. Compiling calls no model.
    """
    graph = build_graph(
        fake_ask("stay_small", lambda s: None, 0),
        small_model="small",
        large_model="large",
        investigator=build_investigator(ScriptedToolModel(24), make_deploys_tool()),
        recall=lambda question: [],
    )
    return set(graph.get_graph().nodes)


def code_exists(code: str, graph: set[str]) -> bool:
    """True when `data-code` names a real graph node or an existing module under code/."""
    if code.startswith("graph:"):
        return code.removeprefix("graph:") in graph
    return (ROOT / code).is_file()


def status_problems(page: str, graph: set[str]) -> list[str]:
    """Every way the page's statuses disagree with the code. Empty means honest. Pure."""
    problems = []
    nodes = page_nodes(page)
    for node in nodes:
        name, status, code = node["name"], node["status"], node["code"]
        if status not in ("built", "planned"):
            problems.append(f"{name}: data-status is {status!r}, not built or planned")
        if not code:
            problems.append(f"{name}: no data-code")
        elif status == "built" and (code == "none" or not code_exists(code, graph)):
            problems.append(f"{name}: marked built, but {code!r} does not exist")
        elif status == "planned" and code != "none" and code_exists(code, graph):
            problems.append(f"{name}: marked planned, but {code!r} already exists")
    drawn = {n["code"].removeprefix("graph:") for n in nodes if n["code"].startswith("graph:")}
    problems += [f"the graph has node {g!r} and the page has no box for it" for g in sorted(graph - drawn)]
    return problems


def test_every_node_names_its_code_and_its_status_is_true() -> None:
    """Built means the code exists; planned means it does not; the graph is fully drawn."""
    assert status_problems(PAGE.read_text(), graph_nodes()) == []


def test_the_check_reads_the_real_graph() -> None:
    """Guard the guard: the compiled graph has the nodes the page is checked against."""
    assert {"__start__", "recall", "investigate", "small", "large", "__end__"} <= graph_nodes()


def test_every_event_kind_the_code_can_emit_has_a_handler_and_none_is_stale() -> None:
    """The page's table equals each vocabulary: nothing undrawn, nothing dead."""
    table = handler_table(PAGE.read_text())
    assert table.keys() == {"app", "harness"}
    assert table["app"] == set(get_args(EventKind))
    assert table["harness"] == set(get_args(EvalEventKind))


def test_the_two_vocabularies_share_no_kind() -> None:
    """The console ends a stream on a kind alone, so the two sets must be disjoint."""
    assert not set(get_args(EventKind)) & set(get_args(EvalEventKind))


# ---- The whole stream through the page's own script ---------------------------


def fake_eval_stream(n: int, stop_after_first_model_call: bool = False) -> list[dict]:
    """The events a fake eval run of `n` cases really streams, as the page receives them.

    With `stop_after_first_model_call`, the first scripted model call presses
    stop, as POST /stop would during case 1.
    """
    holder: list[RunHub] = []

    def sleep(seconds: float) -> None:
        """The fake model's pause: here, optionally, the moment someone presses stop."""
        if stop_after_first_model_call:
            holder[0].stop()

    hub = RunHub(live_models=None, sleep=sleep, fake_delay=0)
    holder.append(hub)
    hub.start_eval(EvalRequest(mode="fake", cases=n))
    events, done = [], False
    deadline = time.monotonic() + 10
    while not done and time.monotonic() < deadline:
        new, done = hub.wait_for(len(events), 1.0)
        events += new
    assert done, "the fake eval run did not finish"
    return [wire(e) for e in events]


def replay(tmp_path: Path, events: list[dict], ids: str) -> dict:
    """Play `events` through the page's script with reduced motion; return rows and texts.

    Reduced motion is the page's own setting (prefers-reduced-motion): no
    pauses and no moving dot, whatever the events are. So the stream is played
    exactly as the console sent it, fake marker and timestamps included.
    """
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events))
    env = {k: v for k, v in os.environ.items() if k != "NODE_OPTIONS"}
    out = subprocess.run(["node", str(REPLAY), str(PAGE), str(path), "2", ids, "reduced"],
                         capture_output=True, text=True, timeout=40, env=env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_a_fake_eval_stream_plays_through_the_page_and_ends_on_the_verdict(tmp_path: Path) -> None:
    """Every event of an unmodified fake stream is shown, and the verdict is marked FAKE.

    An exception in any handler would stop the page's queue and leave rows
    missing, so the row count proves every handler ran.
    """
    events = fake_eval_stream(2)
    kinds = {(e["source"], e["kind"]) for e in events}
    assert ("harness", "metric_scored") in kinds and ("app", "route") in kinds
    shown = replay(tmp_path, events, "evalstatus,result,b_eval")
    assert len(shown["rows"]) == len(events)
    assert [r["kind"] for r in shown["rows"]] == [e["kind"] for e in events]
    assert shown["text"]["evalstatus"].startswith("FAKE data. Verdict: subset, gate not applied")
    assert shown["text"]["b_eval"] == "subset, no gate"
    result = shown["text"]["result"]
    assert result[result.index("Gate") + 1] == "subset, gate not applied"
    assert result[result.index("Data") + 1].startswith("FAKE")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_a_stopped_eval_stream_says_stopped_and_gives_no_verdict(tmp_path: Path) -> None:
    """A stop is shown as a stop, never as a verdict."""
    events = fake_eval_stream(30, stop_after_first_model_call=True)
    assert events[-1]["kind"] == "eval_failed" and events[-1]["data"]["reason"] == "stopped"
    shown = replay(tmp_path, events, "evalstatus,b_eval")
    assert len(shown["rows"]) == len(events)
    assert shown["text"]["evalstatus"].startswith("FAKE data. Eval run stopped: stopped after 1 of 30 cases")
    assert shown["text"]["b_eval"] == "stopped"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_scores_in_the_judge_pass_show_under_their_own_case(tmp_path: Path) -> None:
    """Scores arrive after every case is answered; the panel must name the case they belong to.

    The stream is cut right after the first judged case's last score, so the
    panel still shows progress, not the verdict.
    """
    events = fake_eval_stream(2)
    first = next(e for e in events if e["kind"] == "case_scored")
    cut = events[: events.index(first)]
    shown = replay(tmp_path, cut, "result")
    result = shown["text"]["result"]
    assert result[result.index("Case") + 1] == f"1 of 2: {first['data']['id']}"


def test_a_fake_eval_stream_has_the_shape_the_page_expects() -> None:
    """Started; per case started and answered; then per case four metrics and scored; finished."""
    harness = [e["kind"] for e in fake_eval_stream(2) if e["source"] == "harness"]
    answer_pass = ["case_started", "case_answered"]
    judge_pass = [*["metric_scored"] * 4, "case_scored"]
    assert harness == ["eval_started", *answer_pass, *answer_pass, *judge_pass, *judge_pass, "eval_finished"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")
def test_a_trace_link_is_followed_only_for_http_urls() -> None:
    """The run_traced URL travels in an event; a javascript: or data: URL must never become a link."""
    page = PAGE.read_text()
    func = re.search(r"function traceHref\(url\) \{.*?\n\}\n", page, re.S).group(0)
    cases = ["http://127.0.0.1:3000/project/p/traces/abc", "https://x.example/t", "javascript:alert(1)",
             "data:text/html,x", "not a url", None]
    script = func + f"console.log(JSON.stringify({json.dumps(cases)}.map(traceHref)));"
    env = {k: v for k, v in os.environ.items() if k != "NODE_OPTIONS"}
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=20, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [cases[0], cases[1], None, None, None, None]
