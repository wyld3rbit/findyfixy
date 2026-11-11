"""Tests for the live console server (findyfixy/console/server.py).

What these tests cover
    The real HTTP server on a free port, in a thread, with fake mode and no
    delay. That checks routes, the security headers, the one-run-at-a-time
    rule and that a run streams the expected events over SSE.

Why a real socket
    Host checks, Content-Type checks and SSE framing live in the HTTP layer.
    Calling `RunHub` directly would skip all of them.

Why no Ollama
    Fake scenarios are scripted. The live path is covered by running the
    console by hand and by the integration bank.
"""

import http.client
import json
import threading
import time
from collections.abc import Iterator

import pytest

from findyfixy.console.server import BASELINE_PATH as ROOT_BASELINE, RunHub, serve
from findyfixy.console.fakes import FAKE_RECALL, ScriptedToolModel, fake_ask, fake_cases, scripted_reply


@pytest.fixture
def server() -> Iterator[tuple[RunHub, int]]:
    hub = RunHub(live_models=None, sleep=lambda s: None, fake_delay=0)
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield hub, srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def call(port, method, path, body=None, headers=None):
    """One request. Returns (status, body bytes)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def post_run(port, payload, **headers):
    h = {"Content-Type": "application/json", **headers}
    return call(port, "POST", "/run", json.dumps(payload), h)


def stream(port, start=0):
    """Read the SSE stream to its end and return the events."""
    status, data = call(port, "GET", f"/events?from={start}")
    assert status == 200
    out = []
    for line in data.decode().splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


def test_fake_escalation_streams_the_whole_story(server):
    _, port = server
    status, _ = post_run(port, {"mode": "fake", "scenario": "escalate"})
    assert status == 202
    events = stream(port)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_finished"
    assert [e["data"]["node"] for e in events if e["kind"] == "node_entered"] == [
        "small",
        "large",
    ]
    assert events[-1]["data"]["model_used"] == "large"


def test_retry_scenario_shows_a_failed_then_passed_validation(server):
    _, port = server
    post_run(port, {"mode": "fake", "scenario": "retry_then_small"})
    ok = [e["data"]["ok"] for e in stream(port) if e["kind"] == "validation"]
    assert ok == [False, True]


def test_a_failing_run_ends_with_run_failed_not_a_hang(server):
    _, port = server
    post_run(port, {"mode": "fake", "scenario": "large_fails"})
    last = stream(port)[-1]
    assert last["kind"] == "run_failed"
    assert last["data"]["error_type"] == "TriageError"


def test_replay_from_an_index_skips_earlier_events(server):
    _, port = server
    post_run(port, {"mode": "fake", "scenario": "stay_small"})
    full = stream(port)
    tail = stream(port, start=len(full) - 1)
    assert [e["kind"] for e in tail] == ["run_finished"]


def test_replay_past_the_end_of_a_finished_run_closes(server):
    _, port = server
    post_run(port, {"mode": "fake", "scenario": "stay_small"})
    full = stream(port)
    assert stream(port, start=len(full) + 5) == []


def test_events_before_any_run_is_404(server):
    _, port = server
    assert call(port, "GET", "/events")[0] == 404


def test_live_mode_without_models_is_503(server):
    _, port = server
    status, _ = post_run(port, {"mode": "live", "question": "q"})
    assert status == 503


def test_second_run_while_one_is_in_flight_is_409():
    gate = threading.Event()
    hub = RunHub(live_models=None, sleep=lambda s: gate.wait(5), fake_delay=1)
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        assert post_run(port, {"mode": "fake", "scenario": "stay_small"})[0] == 202
        assert post_run(port, {"mode": "fake", "scenario": "stay_small"})[0] == 409
    finally:
        gate.set()
        stream(port)  # let the run finish
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "fake"},  # no scenario
        {"mode": "fake", "scenario": "nope"},
        {"mode": "weird", "scenario": "stay_small"},
        {"mode": "fake", "scenario": "stay_small", "question": "x" * 8001},
    ],
)
def test_bad_requests_are_422(server, payload):
    _, port = server
    assert post_run(port, payload)[0] == 422


def test_foreign_host_header_is_refused(server):
    _, port = server
    status, _ = call(port, "GET", "/config", headers={"Host": "evil.example"})
    assert status == 403


def test_foreign_origin_is_refused(server):
    _, port = server
    status, _ = post_run(
        port,
        {"mode": "fake", "scenario": "stay_small"},
        Origin="http://evil.example",
    )
    assert status == 403


def test_post_needs_json_content_type(server):
    _, port = server
    body = json.dumps({"mode": "fake", "scenario": "stay_small"})
    status, _ = call(
        port, "POST", "/run", body, {"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert status == 415


def test_oversized_body_is_refused(server):
    _, port = server
    status, _ = call(
        port,
        "POST",
        "/run",
        "x" * 40000,
        {"Content-Type": "application/json"},
    )
    assert status == 413


def test_config_lists_the_fixed_questions(server):
    _, port = server
    status, data = call(port, "GET", "/config")
    cfg = json.loads(data)
    assert status == 200
    assert len(cfg["questions"]) == 5
    assert cfg["live_available"] is False


def test_page_is_served_and_never_uses_innerhtml(server):
    # Event data includes model output, which is untrusted. The page must set
    # it with textContent. This guards against someone "simplifying" it later.
    _, port = server
    status, data = call(port, "GET", "/")
    html = data.decode()
    assert status == 200
    assert "FindyFixy" in html
    assert "innerHTML" not in html
    assert "eval(" not in html


def test_force_escalation_sends_a_confident_answer_to_large(monkeypatch):
    # Live path with a fake model. `ollama_ask` is swapped so no Ollama is
    # needed. The small reply is confident (0.95); only the route is forced.
    import findyfixy.console.server as console

    def confident(model: str, prompt: str) -> str:
        return scripted_reply(0.95)

    monkeypatch.setattr(console, "ollama_ask", confident)
    hub = RunHub(live_models=("s", "l"), sleep=lambda s: None, fake_delay=0)
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        body = {"mode": "live", "question": "q"}
        post_run(port, body)
        plain = stream(port)
        assert [e["data"]["node"] for e in plain if e["kind"] == "node_entered"] == ["small"]

        # The hub clears its running flag just after the terminal event, so a
        # run started at once can see 409. Wait for it to clear.
        for _ in range(50):
            status, _ = post_run(port, {**body, "force_escalation": True})
            if status != 409:
                break
            time.sleep(0.02)
        assert status == 202
        forced = stream(port)
        assert [e["data"]["node"] for e in forced if e["kind"] == "node_entered"] == [
            "small",
            "large",
        ]
        assert forced[0]["data"]["forced"] is True
        assert forced[-1]["kind"] == "run_finished"
    finally:
        srv.shutdown()
        srv.server_close()


# ---- hunt: repeat real runs until a reply fails validation -----------------


def hunt_server(monkeypatch, ask):
    """A live-mode hub whose model is `ask`. Returns (hub, port, server)."""
    import findyfixy.console.server as console

    monkeypatch.setattr(console, "ollama_ask", ask)
    hub = RunHub(live_models=("s", "l"), sleep=lambda s: None, fake_delay=0)
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return hub, srv.server_address[1], srv


def hunt(port, **extra):
    body = {"mode": "live", "question": "q", "hunt_max": 10, **extra}
    status, data = post_run(port, body)
    return status, stream(port) if status == 202 else json.loads(data)


def finished(events):
    return next(e["data"] for e in events if e["kind"] == "hunt_finished")


def test_hunt_stops_at_the_run_that_returns_a_bad_value(monkeypatch):
    import findyfixy.console.server as console

    calls = {"n": 0}

    def ask(model, prompt):
        calls["n"] += 1
        # Call 3 is run 3's first attempt: confidence 75 is illegal. The retry
        # (call 4) gets a good reply, so run 3 still finishes.
        return scripted_reply(75 if calls["n"] == 3 else 0.9)

    _, port, srv = hunt_server(monkeypatch, ask)
    try:
        status, events = hunt(port)
        assert status == 202
        assert finished(events) == {"reason": "found", "runs": 3, "max": 10}
        assert [e["kind"] for e in events].count("run_started") == 3
        assert "retry" in [e["kind"] for e in events]
        assert events[-1]["kind"] == "hunt_finished"
    finally:
        srv.shutdown()
        srv.server_close()


def test_hunt_ends_at_the_cap_when_nothing_goes_wrong(monkeypatch):
    import findyfixy.console.server as console

    _, port, srv = hunt_server(monkeypatch, lambda m, p: scripted_reply(0.9))
    try:
        _, events = hunt(port, hunt_max=4)
        assert finished(events) == {"reason": "exhausted", "runs": 4, "max": 4}
    finally:
        srv.shutdown()
        srv.server_close()


def test_hunt_rotates_the_fixed_questions_from_the_chosen_one(monkeypatch):
    import findyfixy.console.server as console
    from findyfixy.triage.graph import QUESTIONS

    _, port, srv = hunt_server(monkeypatch, lambda m, p: scripted_reply(0.9))
    try:
        _, events = hunt(port, question=QUESTIONS[2], hunt_max=3)
        asked = [e["data"]["question"] for e in events if e["kind"] == "hunt_run"]
        assert asked == QUESTIONS[2:5]
    finally:
        srv.shutdown()
        srv.server_close()


def test_hunt_on_infrastructure_error_stops_and_says_so(monkeypatch):
    def ask(model, prompt):
        raise ConnectionError("ollama is down")

    _, port, srv = hunt_server(monkeypatch, ask)
    try:
        _, events = hunt(port)
        assert finished(events) == {"reason": "error", "runs": 1, "max": 10}
        assert any(e["kind"] == "run_failed" for e in events)
    finally:
        srv.shutdown()
        srv.server_close()


def test_stop_ends_a_hunt_after_the_current_run(monkeypatch):
    import findyfixy.console.server as console

    holder = {}

    def ask(model, prompt):
        holder["hub"].stop()  # as if POST /stop arrived during run 1
        return scripted_reply(0.9)

    hub, port, srv = hunt_server(monkeypatch, ask)
    holder["hub"] = hub
    try:
        _, events = hunt(port)
        assert finished(events) == {"reason": "stopped", "runs": 1, "max": 10}
    finally:
        srv.shutdown()
        srv.server_close()


def test_stop_endpoint_is_accepted(server):
    _, port = server
    status, _ = call(port, "POST", "/stop", headers={"Content-Length": "0"})
    assert status == 200


@pytest.mark.parametrize(
    "extra",
    [{"hunt_max": 51}, {"hunt_max": 0}, {"hunt_max": 5, "force_escalation": True}],
)
def test_bad_hunt_requests_are_422(monkeypatch, extra):
    import findyfixy.console.server as console

    _, port, srv = hunt_server(monkeypatch, lambda m, p: scripted_reply(0.9))
    try:
        body = {"mode": "live", "question": "q", **extra}
        assert post_run(port, body)[0] == 422
    finally:
        srv.shutdown()
        srv.server_close()


def test_hunt_in_fake_mode_is_422(server):
    _, port = server
    body = {"mode": "fake", "scenario": "stay_small", "hunt_max": 3}
    assert post_run(port, body)[0] == 422


def test_config_lists_the_incident_bank():
    from findyfixy.retrieval.incidents import load_incidents

    hub = RunHub(
        live_models=None, incidents=load_incidents(), sleep=lambda s: None, fake_delay=0
    )
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = json.loads(call(srv.server_address[1], "GET", "/config")[1])
        assert len(cfg["incidents"]) == 8
        assert all(i["text"] and i["title"] for i in cfg["incidents"])
    finally:
        srv.shutdown()
        srv.server_close()


def test_hunt_rotates_the_incident_bank_from_the_chosen_one(monkeypatch):
    import findyfixy.console.server as console
    from findyfixy.retrieval.incidents import load_incidents

    bank = load_incidents()
    monkeypatch.setattr(console, "ollama_ask", lambda m, p: scripted_reply(0.9))
    hub = RunHub(live_models=("s", "l"), incidents=bank, sleep=lambda s: None, fake_delay=0)
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        _, events = hunt(port, question=bank[3].text, hunt_max=3)
        asked = [e["data"]["question"] for e in events if e["kind"] == "hunt_run"]
        assert asked == [b.text for b in bank[3:6]]
    finally:
        srv.shutdown()
        srv.server_close()


def test_page_keeps_the_hidden_attribute_working(server):
    # `.field { display: grid }` beats the browser's own `[hidden]` rule, so
    # Live and Fake controls both showed at once. This guards the override.
    _, port = server
    html = call(port, "GET", "/")[1].decode()
    assert "[hidden] { display: none !important; }" in html


# ---- force retry: the first small reply is illegal, the retry is real -------


def test_force_retry_runs_the_real_retry_on_the_real_model(monkeypatch):
    import findyfixy.console.server as console

    prompts: list[str] = []

    def ask(model, prompt):
        prompts.append(prompt)
        return scripted_reply(0.9)

    hub, port, srv = hunt_server(monkeypatch, ask)
    try:
        post_run(port, {"mode": "live", "question": "real incident text", "force_retry": True})
        events = stream(port)
        kinds = [e["kind"] for e in events]
        assert kinds.count("fault_injected") == 1
        assert [e["data"]["ok"] for e in events if e["kind"] == "validation"] == [False, True]
        assert "retry" in kinds
        # The real model was called exactly once, for the retry only.
        assert len(prompts) == 1
        # And that retry prompt carries the real question and the real error.
        assert "real incident text" in prompts[0]
        assert "confidence" in prompts[0]
        assert events[-1]["kind"] == "run_finished"
    finally:
        srv.shutdown()
        srv.server_close()


def test_force_retry_alone_does_not_escalate(monkeypatch):
    import findyfixy.console.server as console

    _, port, srv = hunt_server(monkeypatch, lambda m, p: scripted_reply(0.9))
    try:
        post_run(port, {"mode": "live", "question": "q", "force_retry": True})
        nodes = [e["data"]["node"] for e in stream(port) if e["kind"] == "node_entered"]
        assert nodes == ["small"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_force_retry_and_force_escalation_combine(monkeypatch):
    import findyfixy.console.server as console

    _, port, srv = hunt_server(monkeypatch, lambda m, p: scripted_reply(0.9))
    try:
        post_run(
            port,
            {"mode": "live", "question": "q", "force_retry": True, "force_escalation": True},
        )
        events = stream(port)
        assert [e["data"]["node"] for e in events if e["kind"] == "node_entered"] == [
            "small",
            "large",
        ]
        assert [e["kind"] for e in events].count("fault_injected") == 1
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "live", "question": "q", "force_retry": True, "hunt_max": 3},
        {"mode": "fake", "scenario": "stay_small", "force_retry": True},
    ],
)
def test_force_retry_conflicts_are_422(monkeypatch, body):
    import findyfixy.console.server as console

    _, port, srv = hunt_server(monkeypatch, lambda m, p: scripted_reply(0.9))
    try:
        assert post_run(port, body)[0] == 422
    finally:
        srv.shutdown()
        srv.server_close()


# ---- tool step in the console ------------------------------------------------


def run_fake(port, scenario):
    post_run(port, {"mode": "fake", "scenario": scenario})
    return stream(port)


def test_fake_tool_lookup_streams_the_tool_step_before_small(server):
    _, port = server
    events = run_fake(port, "tool_lookup")
    assert events[0]["data"]["tools"] is True
    entered = [e["data"]["node"] for e in events if e["kind"] == "node_entered"]
    assert entered == ["investigate", "small"]
    call = next(e for e in events if e["kind"] == "tool_call")
    assert call["data"]["args"] == {"service": "auth", "hours": 24}
    result = next(e for e in events if e["kind"] == "tool_result")
    assert "2.14.1" in result["data"]["result"] and not result["data"]["error"]
    assert events[-1]["kind"] == "run_finished"


def test_fake_wrong_window_shows_a_legal_but_empty_lookup(server):
    _, port = server
    events = run_fake(port, "tool_wrong_window")
    call = next(e for e in events if e["kind"] == "tool_call")
    result = next(e for e in events if e["kind"] == "tool_result")
    assert call["data"]["args"]["hours"] == 1
    assert result["data"]["result"] == "[]" and not result["data"]["error"]


def test_scripted_tool_model_reports_no_token_numbers(server):
    # Invented token counts would look real. The step must report them unknown.
    _, port = server
    exited = next(
        e for e in run_fake(port, "tool_lookup")
        if e["kind"] == "node_exited" and e["data"]["node"] == "investigate"
    )
    assert exited["data"]["input_tokens"] is None


def test_other_fake_scenarios_have_no_tool_step(server):
    _, port = server
    events = run_fake(port, "stay_small")
    assert events[0]["data"]["tools"] is False
    assert "tool_call" not in [e["kind"] for e in events]


def live_hub_with_tools(monkeypatch, with_investigator=True):
    import findyfixy.console.server as console
    from findyfixy.tools.investigate import build_investigator
    from findyfixy.tools.deploys import make_deploys_tool

    monkeypatch.setattr(console, "ollama_ask", lambda m, p: scripted_reply(0.9))
    investigator = None
    if with_investigator:
        investigator = build_investigator(
            ScriptedToolModel(24), make_deploys_tool()
        )
    hub = RunHub(
        live_models=("s", "l"), investigator=investigator,
        sleep=lambda s: None, fake_delay=0,
    )
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_live_use_tools_runs_the_tool_step(monkeypatch):
    srv = live_hub_with_tools(monkeypatch)
    port = srv.server_address[1]
    try:
        post_run(port, {"mode": "live", "question": "q", "use_tools": True})
        entered = [e["data"]["node"] for e in stream(port) if e["kind"] == "node_entered"]
        assert entered == ["investigate", "small"]
        post_run_when_free(port, {"mode": "live", "question": "q"})
        entered = [e["data"]["node"] for e in stream(port) if e["kind"] == "node_entered"]
        assert entered == ["small"]  # without the box ticked, no tool step
    finally:
        srv.shutdown()
        srv.server_close()


def post_run_when_free(port, body):
    for _ in range(50):
        status, _ = post_run(port, body)
        if status != 409:
            return status
        time.sleep(0.02)
    raise AssertionError("run never freed up")


def test_live_use_tools_without_a_configured_tool_step_is_503(monkeypatch):
    srv = live_hub_with_tools(monkeypatch, with_investigator=False)
    try:
        status, _ = post_run(
            srv.server_address[1], {"mode": "live", "question": "q", "use_tools": True}
        )
        assert status == 503
    finally:
        srv.shutdown()
        srv.server_close()


def test_config_says_whether_the_tool_step_is_available(server, monkeypatch):
    _, port = server
    cfg = json.loads(call(port, "GET", "/config")[1])
    assert cfg["tools_available"] is False
    assert {"tool_lookup", "tool_wrong_window"} <= set(cfg["scenarios"])


def test_page_has_the_tool_step_nodes_and_marks_empty_results_as_a_warning(server):
    _, port = server
    html = call(port, "GET", "/")[1].decode()
    for needle in ('id="n_inv"', 'id="n_tool"', 'id="e_inv_tool"', 'id="usetools"'):
        assert needle in html
    # An empty lookup is legal but suspicious (wrong window), so not green.
    assert '"warn"' in html and ".node.warn" in html


def test_streamed_model_replies_carry_the_served_by_field(monkeypatch):
    import findyfixy.console.server as console
    from findyfixy.triage.record import Reply

    monkeypatch.setattr(console, "ollama_ask", lambda m, p: Reply(scripted_reply(0.9), served_by=m))
    hub = RunHub(live_models=("s", "l"), sleep=lambda s: None, fake_delay=0)
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        post_run(port, {"mode": "live", "question": "q"})
        replies = [e["data"] for e in stream(port) if e["kind"] == "model_reply"]
        assert replies and replies[0]["served_by"] == "s" and replies[0]["mismatch"] is False
    finally:
        srv.shutdown()
        srv.server_close()


def test_fake_mode_says_the_served_model_is_not_reported(server):
    _, port = server
    replies = [e["data"] for e in run_fake(port, "stay_small") if e["kind"] == "model_reply"]
    assert replies[0]["served_by"] is None


def test_page_shows_and_flags_the_served_model(server):
    _, port = server
    html = call(port, "GET", "/")[1].decode()
    for needle in ("served by", "MISMATCH", "Served by", "not reported"):
        assert needle in html


def test_the_page_declares_an_inline_icon_so_browsers_do_not_ask_for_one(server):
    _, port = server
    html = call(port, "GET", "/")[1].decode()
    assert 'rel="icon"' in html and "data:image/svg+xml" in html


def test_favicon_requests_get_no_content_and_are_not_logged(server, capsys):
    _, port = server
    status, body = call(port, "GET", "/favicon.ico")
    call(port, "GET", "/config")
    err = capsys.readouterr().err
    assert status == 204 and body == b""
    assert "/config" in err and "favicon" not in err


# ---- stress ------------------------------------------------------------------


def stressed_hub(monkeypatch):
    """A live hub whose stressed `ask` is a recording fake. Returns (srv, seen)."""
    import findyfixy.console.server as console

    seen: list[str] = []

    def fake_make(stress):
        seen.append(stress)
        return lambda m, p: scripted_reply(0.9)

    monkeypatch.setattr(console, "make_ollama_ask", fake_make)
    hub = RunHub(live_models=("s", "l"), sleep=lambda s: None, fake_delay=0)
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_a_stressed_live_run_uses_that_stress_and_says_so(monkeypatch):
    srv, seen = stressed_hub(monkeypatch)
    port = srv.server_address[1]
    try:
        post_run(port, {"mode": "live", "question": "q", "stress": "vague_prompt"})
        events = stream(port)
        assert seen == ["vague_prompt"]
        assert events[0]["data"]["stress"] == "vague_prompt"
    finally:
        srv.shutdown(); srv.server_close()


def test_an_unstressed_live_run_does_not_build_a_stressed_ask(monkeypatch):
    srv, seen = stressed_hub(monkeypatch)
    port = srv.server_address[1]
    try:
        post_run(port, {"mode": "live", "question": "q"})
        stream(port)
        assert seen == []
    finally:
        srv.shutdown(); srv.server_close()


def test_hunt_can_be_combined_with_a_stress(monkeypatch):
    srv, seen = stressed_hub(monkeypatch)
    port = srv.server_address[1]
    try:
        status, _ = post_run(port, {"mode": "live", "question": "q", "hunt_max": 2,
                                    "stress": "no_format"})
        assert status == 202 and seen == ["no_format"]
        stream(port)
    finally:
        srv.shutdown(); srv.server_close()


@pytest.mark.parametrize("body", [
    {"mode": "live", "question": "q", "stress": "nonsense"},
    {"mode": "fake", "scenario": "stay_small", "stress": "no_format"},
])
def test_bad_stress_requests_are_422(monkeypatch, body):
    srv, _ = stressed_hub(monkeypatch)
    try:
        assert post_run(srv.server_address[1], body)[0] == 422
    finally:
        srv.shutdown(); srv.server_close()


def test_config_lists_the_stresses_with_descriptions(server):
    _, port = server
    cfg = json.loads(call(port, "GET", "/config")[1])
    assert {"none", "no_format", "vague_prompt", "percent_log", "small_context"} == set(cfg["stresses"])
    assert all(cfg["stresses"].values())


def test_page_defaults_to_live_mode_and_has_the_stress_control(server):
    _, port = server
    html = call(port, "GET", "/")[1].decode()
    assert '<option value="live" selected>' in html
    assert html.index('<option value="live"') < html.index('<option value="fake"')
    assert 'id="stress"' in html


# ---- recall: similar past incidents before the small model ------------------


def _serve(hub):
    srv = serve(hub, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_a_fake_run_shows_the_recall_step_when_the_hub_has_one():
    from findyfixy.console.server import FAKE_RECALL

    srv, port = _serve(RunHub(live_models=None, recall=lambda q: [], sleep=lambda s: None, fake_delay=0))
    try:
        post_run(port, {"mode": "fake", "scenario": "stay_small"})
        events = stream(port)
    finally:
        srv.shutdown()
        srv.server_close()
    assert events[0]["data"]["recall"] is True
    assert [e["data"]["node"] for e in events if e["kind"] == "node_entered"][0] == "recall"
    (done,) = [e for e in events if e["kind"] == "node_exited" and e["data"]["node"] == "recall"]
    # A fake run never calls the real recall: it shows the fixed, invented ones.
    assert done["data"]["incidents"] == [p.incident_id for p in FAKE_RECALL]


def test_a_live_run_uses_the_hubs_recall_and_a_hub_without_one_has_no_step(monkeypatch):
    import findyfixy.console.server as console
    from findyfixy.triage.record import Prior

    monkeypatch.setattr(console, "ollama_ask", lambda model, prompt: scripted_reply(0.95))
    seen: list[str] = []

    def recall(question):
        seen.append(question)
        return [Prior("INC-9", "auth", "sev2", "s", "c", "r")]

    srv, port = _serve(RunHub(live_models=("s", "l"), recall=recall, sleep=lambda s: None, fake_delay=0))
    try:
        post_run(port, {"mode": "live", "question": "auth slow"})
        events = stream(port)
    finally:
        srv.shutdown()
        srv.server_close()
    assert seen == ["auth slow"]
    (done,) = [e for e in events if e["kind"] == "node_exited" and e["data"]["node"] == "recall"]
    assert done["data"]["incidents"] == ["INC-9"]

    srv, port = _serve(RunHub(live_models=None, sleep=lambda s: None, fake_delay=0))
    try:
        post_run(port, {"mode": "fake", "scenario": "stay_small"})
        events = stream(port)
    finally:
        srv.shutdown()
        srv.server_close()
    assert events[0]["data"]["recall"] is False
    assert "recall" not in [e["data"].get("node") for e in events]


# ---- eval run: the harness on the first N golden cases ------------------------

FAKE_SIZE = len(fake_cases())


def post_eval(port, payload, **headers):
    """POST /eval with a JSON body and any extra headers. Returns (status, body)."""
    h = {"Content-Type": "application/json", **headers}
    return call(port, "POST", "/eval", json.dumps(payload), h)


def no_wait(seconds):
    """A sleep that does not sleep: fake runs finish at once."""


def eval_result(n_cases=1):
    """A minimal report wrapped as a subset result, for runs that score nothing."""
    from findyfixy.console.eval_run import EvalResult

    return EvalResult(report={"aggregates": {"overall": {"n": n_cases, "n_run_errors": 0, "scores": {}}}}, subset=True)


def test_a_fake_eval_streams_both_sources_and_ends_on_a_subset_verdict(server):
    """Harness and app events share one tagged stream, ending on a subset verdict."""
    _, port = server
    assert post_eval(port, {"mode": "fake", "cases": 2})[0] == 202
    events = stream(port)
    assert {e["source"] for e in events} == {"app", "harness"}
    assert events[0]["kind"] == "eval_started" and events[0]["data"]["n"] == 2
    assert events[0]["data"]["system"]["data"] == "FAKE"
    # The graph inside each case reports through the same stream.
    assert [e["data"]["node"] for e in events if e["kind"] == "node_entered"][:2] == ["recall", "small"]
    last = events[-1]
    assert last["source"] == "harness" and last["kind"] == "eval_finished"
    assert last["data"]["subset"] is True and last["data"]["gate"] is None
    assert last["data"]["summary"]["n"] == 2


def test_a_full_fake_eval_is_gated_read_only_and_fails_as_not_comparable(tmp_path):
    """Fake scores never pass the real baseline, and the baseline is only read.

    The judge and the golden set differ, so the gate refuses to compare. The
    baseline file keeps its bytes and nothing else appears next to it.
    """
    baseline = tmp_path / "baseline.json"
    original = ROOT_BASELINE.read_bytes()
    baseline.write_bytes(original)
    srv, port = _serve(RunHub(live_models=None, sleep=no_wait, fake_delay=0, baseline_path=baseline))
    try:
        assert post_eval(port, {"mode": "fake", "cases": FAKE_SIZE})[0] == 202
        last = stream(port)[-1]
    finally:
        srv.shutdown()
        srv.server_close()
    assert last["kind"] == "eval_finished" and last["data"]["subset"] is False
    problems = last["data"]["gate"]
    assert any("not comparable" in p for p in problems)
    assert any("run error" in p for p in problems)  # the scripted large_fails cases
    assert baseline.read_bytes() == original and list(tmp_path.iterdir()) == [baseline]


def test_a_full_eval_without_a_baseline_ends_in_eval_failed(tmp_path):
    """A missing baseline is a failed run with its reason, not a hang or a pass."""
    srv, port = _serve(RunHub(live_models=None, sleep=no_wait, fake_delay=0, baseline_path=tmp_path / "none.json"))
    try:
        post_eval(port, {"mode": "fake", "cases": FAKE_SIZE})
        last = stream(port)[-1]
    finally:
        srv.shutdown()
        srv.server_close()
    assert last["kind"] == "eval_failed" and last["data"]["error_type"] == "BaselineMissing"
    assert last["data"]["reason"] == "error"


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "fake", "cases": 0},
        {"mode": "fake", "cases": FAKE_SIZE + 1},
        {"mode": "fake", "cases": "2"},
        {"mode": "fake", "cases": 2.5},
        {"mode": "fake", "cases": True},
        {"mode": "fake"},
        {"mode": "weird", "cases": 2},
        {"mode": "fake", "cases": 2, "scenario": "stay_small"},
    ],
)
def test_bad_eval_requests_are_422(server, payload):
    """Counts outside the fake set, non-integers, missing or extra fields are refused."""
    _, port = server
    assert post_eval(port, payload)[0] == 422


def test_the_live_limit_is_the_live_sets_own_size():
    """A live count is checked against the live suite's size, not the fake one's."""
    from findyfixy.console.eval_run import EvalSuite

    suite = EvalSuite(run=lambda n, **kw: eval_result(n), size=3)
    srv, port = _serve(RunHub(live_models=("s", "l"), live_eval=suite, sleep=no_wait, fake_delay=0))
    try:
        assert post_eval(port, {"mode": "live", "cases": 4})[0] == 422
        assert post_eval(port, {"mode": "live", "cases": 3})[0] == 202
        stream(port)
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_live_eval_without_models_is_503(server):
    """No live suite configured: the request is refused as not configured."""
    _, port = server
    assert post_eval(port, {"mode": "live", "cases": 2})[0] == 503


def test_an_eval_gets_the_same_header_checks_as_a_run(server):
    """Content type, origin, host and body size are checked as for /run."""
    _, port = server
    body = json.dumps({"mode": "fake", "cases": 2})
    assert call(port, "POST", "/eval", body, {"Content-Type": "text/plain"})[0] == 415
    assert post_eval(port, {"mode": "fake", "cases": 2}, Origin="http://evil.example")[0] == 403
    assert post_eval(port, {"mode": "fake", "cases": 2}, Host="evil.example")[0] == 403
    assert call(port, "POST", "/eval", "x" * 40000, {"Content-Type": "application/json"})[0] == 413


def test_an_eval_and_a_triage_run_never_overlap():
    """One run slot: an eval blocks a triage run and another eval, and the reverse."""
    gate = threading.Event()
    srv, port = _serve(RunHub(live_models=None, sleep=lambda s: gate.wait(5), fake_delay=1))
    try:
        assert post_eval(port, {"mode": "fake", "cases": 1})[0] == 202
        assert post_run(port, {"mode": "fake", "scenario": "stay_small"})[0] == 409
        assert post_eval(port, {"mode": "fake", "cases": 1})[0] == 409
        gate.set()
        stream(port)
        gate.clear()
        assert post_run_when_free(port, {"mode": "fake", "scenario": "stay_small"}) == 202
        assert post_eval(port, {"mode": "fake", "cases": 1})[0] == 409
    finally:
        gate.set()
        stream(port)
        srv.shutdown()
        srv.server_close()


def test_config_reports_each_eval_sets_size(server):
    """The page reads the sizes from /config; live is None when not configured."""
    _, port = server
    cfg = json.loads(call(port, "GET", "/config")[1])
    assert cfg["eval_sizes"] == {"live": None, "fake": FAKE_SIZE}
    assert cfg["eval_default"] == 2


# ---- stopping an eval run ---------------------------------------------------------


def test_stop_ends_a_fake_eval_after_the_case_in_flight():
    """POST /stop during case 1 of a full run: case 1 is answered, case 2 never starts.

    The judge pass never starts either (runner.py answers every case first),
    so case 1 is answered but never scored.
    """
    holder = {}

    def press_stop(seconds):
        """The fake model's pause: the moment someone presses stop."""
        holder["stopping"] = call(holder["port"], "POST", "/stop", headers={"Content-Length": "0"})

    srv, port = _serve(RunHub(live_models=None, sleep=press_stop, fake_delay=0))
    holder["port"] = port
    try:
        assert post_eval(port, {"mode": "fake", "cases": FAKE_SIZE})[0] == 202
        events = stream(port)
    finally:
        srv.shutdown()
        srv.server_close()
    assert holder["stopping"] == (200, b'{"stopping": true}')
    kinds = [e["kind"] for e in events]
    assert kinds.count("case_started") == 1 and kinds.count("case_answered") == 1
    assert kinds.count("metric_scored") == 0 and kinds.count("case_scored") == 0
    last = events[-1]
    assert last["kind"] == "eval_failed" and last["data"]["reason"] == "stopped"
    assert last["data"]["error"] == f"stopped after 1 of {FAKE_SIZE} cases"


def test_stop_says_false_when_nothing_can_stop(server):
    """Idle, or a single triage run, cannot stop: the answer says so."""
    _, port = server
    assert call(port, "POST", "/stop", headers={"Content-Length": "0"}) == (200, b'{"stopping": false}')
    post_run(port, {"mode": "fake", "scenario": "stay_small"})
    stream(port)
    assert call(port, "POST", "/stop", headers={"Content-Length": "0"}) == (200, b'{"stopping": false}')


def test_a_stop_left_over_from_a_hunt_does_not_end_the_next_eval():
    """The stop flag is cleared when a run starts, so an old press cannot leak."""
    hub = RunHub(live_models=None, sleep=no_wait, fake_delay=0)
    hub._stop.set()  # as if a press landed just after the previous run ended
    srv, port = _serve(hub)
    try:
        post_eval(port, {"mode": "fake", "cases": 2})
        last = stream(port)[-1]
    finally:
        srv.shutdown()
        srv.server_close()
    assert last["kind"] == "eval_finished"


# ---- the live eval path, without a model ------------------------------------------


def test_a_live_eval_hub_runs_the_injected_eval_and_streams_its_events():
    """The hub hands the live suite the count, both emitters and the stop flag."""
    from findyfixy.console.eval_run import EvalSuite

    seen = []

    def live(n, *, emit, on_event, should_stop):
        """A live run that reports one harness and one app event."""
        seen.append((n, should_stop()))
        on_event("eval_started", n=n, cases=["g01"], system={})
        emit("node_entered", node="small", model="s")
        return eval_result()

    hub = RunHub(live_models=("s", "l"), live_eval=EvalSuite(run=live, size=5), sleep=no_wait, fake_delay=0)
    srv, port = _serve(hub)
    try:
        assert post_eval(port, {"mode": "live", "cases": 3})[0] == 202
        events = stream(port)
    finally:
        srv.shutdown()
        srv.server_close()
    assert seen == [(3, False)]
    assert [(e["source"], e["kind"]) for e in events] == [
        ("harness", "eval_started"), ("app", "node_entered"), ("harness", "eval_finished")]


def test_live_eval_cuts_the_session_to_n_cases_and_hands_the_graph_its_emitter(monkeypatch, tmp_path):
    """The real `live_eval` over the real `live_runner`, with a fake session, judge, digest and git."""
    from contextlib import contextmanager

    from findyfixy.console.eval_run import live_eval
    from findyfixy.evals.adapter import FindyFixy
    from findyfixy.triage.graph import build_graph
    from wyld3rness.eval import cli
    from wyld3rness.eval.cli import Session
    from wyld3rness.testing import FakeJudge, case, embed

    real = cli.live_runner
    monkeypatch.setattr(cli, "live_runner", lambda open_session, **kw: real(
        open_session, **kw, make_judge=lambda m: FakeJudge(), digest=lambda m: "dig",
        git=lambda repo: ("abc1234", False)))
    controls = []

    @contextmanager
    def open_session(negative_control, *, emit):
        """A session of three cases whose graph reports to the run's `emit`."""
        controls.append(negative_control)

        def make_graph(recall):
            """A graph with a confident scripted model and this case's recall."""
            return build_graph(lambda m, p: scripted_reply(0.9), small_model="s", large_model="l",
                               emit=emit, recall=recall)

        sut = FindyFixy(make_graph=make_graph, candidates=lambda q, n: [], small_model="s", large_model="l")
        cases = [case(f"g{i}", source_id=f"S{i}", twin_id=None) for i in range(3)]
        yield Session(sut=sut, cases=cases, judge_model="j", embed=embed)

    golden = tmp_path / "golden.jsonl"
    golden.write_text("{}\n")
    suite = live_eval(open_session, golden_path=golden, repo_dir=tmp_path, progress=lambda m: None, size=3)
    assert suite.size == 3
    app_events, harness_events = [], []
    result = suite.run(2, emit=lambda kind, **d: app_events.append(kind),
                       on_event=lambda kind, **d: harness_events.append((kind, d)), should_stop=lambda: False)
    assert controls == [False]
    assert [c["id"] for c in result.report["cases"]] == ["g0", "g1"] and result.subset is True
    assert "route" in app_events
    assert harness_events[0] == ("eval_started", {"n": 2, "cases": ["g0", "g1"], "system": {
        "small_model": "s", "large_model": "l", "threshold": 0.7, "recall_k": 3}})
    assert result.report["provenance"]["judge_digest"] == "dig"
    full = suite.run(3, emit=lambda kind, **d: None, on_event=lambda kind, **d: None, should_stop=lambda: False)
    assert full.subset is False
    from wyld3rness.eval.runner import EvalStopped

    with pytest.raises(EvalStopped):
        suite.run(3, emit=lambda kind, **d: None, on_event=lambda kind, **d: None, should_stop=lambda: True)


def test_a_subset_result_is_never_gated(tmp_path):
    """The gate needs the full set: a subset verdict is None even with no baseline on disk."""
    from findyfixy.console.eval_run import EvalResult, verdict

    assert verdict(EvalResult(report={}, subset=True), tmp_path / "missing.json") is None
