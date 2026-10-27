"""Day 2: the live console server. Run the real graph, stream its events.

What it does
    Serves the architecture page and runs the triage graph on request. While
    a run is in flight it streams every event (app/events.py) to the browser
    over Server-Sent Events (SSE). The page animates from those events.

        GET  /         the page (docs/architecture.html)
        GET  /config   questions, fake scenarios, which modes are available
        POST /run      start one run (or a hunt): {"mode", "question"?, "scenario"?}
        POST /stop     ask a running hunt to stop after its current run
        GET  /events   SSE stream of the latest run, replayed from the start

Run it:   uv run python -m app.console      (then open the printed address)
Develop:  uv run python -m app.console --reload
          Restarts the server when code changes and reloads the page. See
          app/dev_reload.py. `--port N` picks another port.
Test it:  uv run pytest

Why the standard library, and not FastAPI
    This is a local dev tool with four routes. `http.server` adds nothing to
    the supply chain (no new package, no license, no advisory to track).
    The cost is hand-written plumbing and no async. At gateway scale (Day 8)
    use a real framework. Moving over is cheap: the logic lives in `RunHub`,
    and the HTTP layer is a thin shell around it.

Why SSE, and not websockets
    The flow is one-way: server to page. SSE is plain HTTP, the browser
    reconnects by itself, and there is no framing code. The page sends its
    one command (start a run) as an ordinary POST.

Security model (a local tool is still a web server)
    - Binds to 127.0.0.1 only. Nothing on the network can reach it.
    - Host header must be 127.0.0.1 or localhost on our port. That blocks
      DNS rebinding, where a hostile site points its own name at 127.0.0.1
      and drives this API from your browser.
    - POST needs `Content-Type: application/json`. A cross-site form cannot
      send that without a CORS preflight, and we answer no preflight.
    - Origin, when present, must match Host.
    - The question is capped at 8000 characters and the body at 32 KB.
    - One run at a time. Two 7B-class runs would not fit in 16 GB.

Failure modes
    - A browser that disconnects mid-stream is a broken pipe in the handler.
      It is caught and ends that stream only. The run keeps going.
    - Only the latest run is kept, in memory. A restart loses it. This is a
      viewer, not a record. Day 6 (Langfuse) is the record.
    - If Ollama is down, a live run ends with a `run_failed` event that names
      the error. The server stays up.
"""

import json
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, ValidationError

from app.dev_reload import fingerprint, snapshot
from app.events import Emit, Event, make_emit
from app.incidents import DEFAULT_PATH, Incident, load_incidents
from app.paths import DOCS, ROOT
from app.investigate import build_investigator
from app.stress import DESCRIPTIONS, STRESSES, Stress, make_ollama_ask
from app.tools import make_deploys_tool
from app.graph import QUESTIONS, THRESHOLD, Recall, build_graph, ollama_ask
from app.triage import Ask, Prior, TriageError, answer_options

Mode = Literal["live", "fake"]
Scenario = Literal[
    "stay_small",
    "retry_then_small",
    "escalate",
    "large_fails",
    "tool_lookup",
    "tool_wrong_window",
]

# The page lives next to the docs (app/paths.py resolves from the file, never
# from the current directory, so the server works from any clone).
PAGE = DOCS / "architecture.html"

# Realistic incidents are log excerpts of a few thousand characters (see
# app/incidents.py), so the question cap is far above a one-line question.
_MAX_BODY = 32 * 1024
_MAX_QUESTION = 8000
_KEEPALIVE_SECONDS = 10.0

_FORCE_THRESHOLD = 1.01

# How many prior incidents go into the prompt. Three is the lesson's number: more
# costs prompt tokens and, with a 1.5B model, starts to distract it.
RECALL_K = 3

# What the page shows for the recall step in a fake run: fixed, invented, and
# marked as such by its IDs. A fake run must never need a database or a model.
FAKE_RECALL: tuple[Prior, ...] = (
    Prior("FAKE-1", "auth", "sev2", "Latency rose after a certificate rotation.", "Peers lacked the new intermediate authority."),
    Prior("FAKE-2", "auth", "sev3", "Slow logins after a rotation.", "Trust was not tested before cutover."),
    Prior("FAKE-3", "gateway", "sev2", "Saturation at the daily peak.", "The autoscaler floor was too low."),
)

# A hunt repeats real model runs, so the cap is a cost and time bound. The
# server enforces it. The page default is lower than the limit on purpose.
HUNT_DEFAULT = 20
HUNT_LIMIT = 50

_RUN_TERMINAL = ("run_finished", "run_failed")
_HUNT_TERMINAL = ("hunt_finished",)


def _reply(confidence: float | str, severity: str = "sev2") -> str:
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
    "stay_small": {"small": [_reply(0.9)], "large": []},
    "retry_then_small": {"small": [_reply(75), _reply(0.85)], "large": []},
    "escalate": {"small": [_reply(0.3)], "large": [_reply(0.92, "sev1")]},
    "large_fails": {
        "small": [_reply(75), _reply(75)],
        "large": ["not json", "still not json"],
    },
    # Scenarios with the tool step. The tool and the ToolNode are the real
    # ones, reading data/deploys.json. Only the tool-calling model is scripted.
    "tool_lookup": {"small": [_reply(0.9)], "large": []},
    "tool_wrong_window": {"small": [_reply(0.9)], "large": []},
}

# The look-back window the scripted tool model asks for, per scenario.
# `tool_wrong_window` mirrors what the real 1.5B model did on a 24 hour
# question: it sent a legal but wrong 1 hour window, so the tool returns [].
TOOL_HOURS: dict[str, int] = {"tool_lookup": 24, "tool_wrong_window": 1}


class _ScriptedToolModel:
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
            return _reply(75)
        return ask(model, prompt)

    return wrapped


class RunRequest(BaseModel):
    """The body of POST /run, validated before anything starts."""

    mode: Mode
    question: str | None = Field(default=None, max_length=_MAX_QUESTION)
    scenario: Scenario | None = None
    # Live mode only. Overrides the route so the large model runs for real.
    force_escalation: bool = False
    # Live mode only. Replace the small model's FIRST reply with an illegal
    # one (confidence 75), so validation rejects it and the retry runs for real.
    force_retry: bool = False
    # Live mode only. Make the triage model fail on purpose (app/stress.py).
    stress: Stress = "none"
    # Live mode only. Run the tool step (look up deploys) before triage.
    use_tools: bool = False
    # Live mode only. Repeat real runs until a reply fails validation, or until
    # this many runs have been made. None means a single run.
    hunt_max: int | None = Field(default=None, ge=1, le=HUNT_LIMIT)


class Busy(Exception):
    """A run is already in flight. Only one at a time."""


class NotConfigured(Exception):
    """Live mode was asked for but no model names are configured."""


class _Watch:
    """Sees every event on its way to the sink and notes a failed validation.

    The hunt reads `invalid` after each run, then resets it. It lives on the
    run thread only, so it needs no lock.
    """

    def __init__(self, publish: Callable[[Event], None]) -> None:
        self._publish = publish
        self.invalid = False

    def sink(self, event: Event) -> None:
        """Note a failed validation, then pass the event on."""
        if event.kind == "validation" and not event.data["ok"]:
            self.invalid = True
        self._publish(event)


def _rotate(question: str, banks: list[list[str]]) -> list[str]:
    """The questions a hunt cycles through, starting at `question`.

    If `question` is in one of the banks (the lesson five, or the incident
    bank), rotate that bank so the cycle starts there. Any other question
    (typed by the user) is the only one used.
    """
    for bank in banks:
        if question in bank:
            i = bank.index(question)
            return bank[i:] + bank[:i]
    return [question]


class RunHub:
    """Owns the current run and its event history. Thread-safe.

    One lock and one condition guard `events` and `running`. The run thread
    appends and notifies. Each SSE handler waits on the condition for new
    events past the index it has already sent. Replay falls out of this: a
    late subscriber starts at index 0 and reads the same list.
    """

    def __init__(
        self,
        *,
        live_models: tuple[str, str] | None,
        incidents: list[Incident] | None = None,
        investigator: CompiledStateGraph | None = None,
        recall: Recall | None = None,
        sleep: Callable[[float], None] = time.sleep,
        fake_delay: float = 0.6,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._live_models = live_models
        self.incidents = incidents or []
        self._investigator = investigator
        # When set, live runs use it and fake runs use `FAKE_RECALL`. When None
        # no run has a recall step, and the graph is what it was before Day 4.
        self._recall = recall
        self._sleep = sleep
        self._fake_delay = fake_delay
        self._clock = clock
        self._cond = threading.Condition()
        self._events: list[Event] = []
        self._running = False
        self._run_id = 0
        self._stop = threading.Event()
        self._terminal: tuple[str, ...] = _RUN_TERMINAL

    @property
    def tools_available(self) -> bool:
        """True when a real tool step was configured (live mode only)."""
        return self._investigator is not None

    @property
    def live_available(self) -> bool:
        """True when `SMALL_MODEL` and `LARGE_MODEL` were configured."""
        return self._live_models is not None

    def start(self, req: RunRequest) -> int:
        """Start a run in a thread and return its id. Raises `Busy`.

        Validation of the combination happens here, before the thread, so a
        bad request is a clean HTTP error and not a failed run.
        """
        if req.mode == "live":
            if self._live_models is None:
                raise NotConfigured("SMALL_MODEL and LARGE_MODEL are not set")
            if not (req.question and req.question.strip()):
                raise ValueError("live mode needs a question")
            small, large = self._live_models
            # Unstressed live runs keep using `ollama_ask`, so tests and anything
            # else that replaces it behave as before.
            ask: Ask = ollama_ask if req.stress == "none" else make_ollama_ask(req.stress)
            question = req.question.strip()
            forced = req.force_escalation
            investigator = None
            if req.use_tools:
                if self._investigator is None:
                    raise NotConfigured("the tool step is not configured")
                investigator = self._investigator
            if req.hunt_max is not None and (forced or req.force_retry):
                raise ValueError("hunt cannot be combined with a forced fault")
        else:
            if req.hunt_max is not None or req.force_retry or req.stress != "none":
                raise ValueError("hunt, force_retry and stress need live mode")
            if req.scenario is None:
                raise ValueError("fake mode needs a scenario")
            small, large = "small", "large"
            ask = fake_ask(req.scenario, self._sleep, self._fake_delay)
            question = (req.question or "").strip() or FAKE_QUESTION
            forced = False  # fake scenarios already script their own routing
            investigator = None
            if req.scenario in TOOL_HOURS:
                investigator = build_investigator(
                    _ScriptedToolModel(TOOL_HOURS[req.scenario]),
                    make_deploys_tool(),
                )

        with self._cond:
            if self._running:
                raise Busy
            self._running = True
            self._run_id += 1
            self._events = []
            self._stop.clear()
            self._terminal = _HUNT_TERMINAL if req.hunt_max else _RUN_TERMINAL
            run_id = self._run_id

        threading.Thread(
            target=self._run,
            args=(
                run_id, ask, small, large, question, req.mode, forced,
                req.hunt_max, req.force_retry,
            ),
            kwargs={
                "investigator": investigator,
                "stress": req.stress,
                "recall": self._recall_for(req.mode),
            },
            daemon=True,
        ).start()
        return run_id

    def _recall_for(self, mode: Mode) -> Recall | None:
        """The recall step for a run of this mode, or None when the hub has none."""
        if self._recall is None:
            return None
        return self._recall if mode == "live" else (lambda question: FAKE_RECALL)

    def _publish(self, event: Event) -> None:
        """Sink for the run's emitter: append and wake the SSE handlers."""
        with self._cond:
            self._events.append(event)
            self._cond.notify_all()

    def _banks(self) -> list[list[str]]:
        """The question banks a hunt can rotate: the lesson five, then incidents."""
        return [QUESTIONS, [i.text for i in self.incidents]]

    def stop(self) -> None:
        """Ask a running hunt to stop after its current run. Harmless otherwise."""
        self._stop.set()

    def _run(
        self,
        run_id: int,
        ask: Ask,
        small: str,
        large: str,
        question: str,
        mode: Mode,
        forced: bool,
        hunt_max: int | None,
        force_retry: bool,
        *,
        investigator: CompiledStateGraph | None = None,
        stress: Stress = "none",
        recall: Recall | None = None,
    ) -> None:
        """The run thread. Always ends with exactly one terminal event.

        A single run ends with `run_finished` or `run_failed`. A hunt ends with
        `hunt_finished`. Any exception becomes a failure event, so a viewer
        never hangs waiting. `_running` is cleared last, after the terminal
        event is published, so a new run cannot start and wipe events a
        viewer has yet to read.
        """
        watch = _Watch(self._publish)
        emit = make_emit(watch.sink, self._clock)
        # Confidence is at most 1.0 and the route keeps a record only when
        # confidence >= threshold. A threshold above 1.0 therefore sends every
        # valid small answer to the large model. Both models still run for real.
        threshold = _FORCE_THRESHOLD if forced else THRESHOLD
        if force_retry:
            ask = inject_bad_first_reply(ask, small, emit)

        def once(q: str) -> str:
            return self._run_once(
                emit, ask, small, large, q, mode, forced, run_id, threshold,
                investigator, stress, recall,
            )

        try:
            if hunt_max is None:
                once(question)
            else:
                self._hunt(emit, watch, once, question, hunt_max)
        finally:
            with self._cond:
                self._running = False
                self._cond.notify_all()

    def _run_once(
        self,
        emit: Emit,
        ask: Ask,
        small: str,
        large: str,
        question: str,
        mode: Mode,
        forced: bool,
        run_id: int,
        threshold: float,
        investigator: CompiledStateGraph | None = None,
        stress: Stress = "none",
        recall: Recall | None = None,
    ) -> Literal["finished", "failed", "error"]:
        """One graph run. Returns how it ended, after emitting its event.

        `failed` means the model never gave a valid record (`TriageError`).
        `error` is anything else, usually infrastructure (Ollama down). A hunt
        treats the two differently: the first is what it is looking for, the
        second means stop and say so.
        """
        emit("run_started", run=run_id, mode=mode, question=question,
             small_model=small, large_model=large, threshold=threshold, forced=forced,
             tools=investigator is not None, stress=stress,
             recall=recall is not None)
        try:
            graph = build_graph(
                ask,
                small_model=small,
                large_model=large,
                threshold=threshold,
                emit=emit,
                investigator=investigator,
                investigate_model=small if investigator is not None and mode == "live" else None,
                recall=recall,
            )
            result = graph.invoke({"question": question})
            emit(
                "run_finished",
                model_used=result["model_used"],
                triage=result["triage"].model_dump(),
            )
            return "finished"
        except TriageError as err:
            emit("run_failed", error_type="TriageError", error=str(err))
            return "failed"
        except Exception as err:  # noqa: BLE001 - the viewer must hear about it
            emit("run_failed", error_type=type(err).__name__, error=str(err))
            return "error"

    def _hunt(
        self,
        emit: Emit,
        watch: "_Watch",
        once: Callable[[str], str],
        question: str,
        hunt_max: int,
    ) -> None:
        """Repeat real runs until a reply fails validation, or the cap is hit.

        A "bad value" is any `validation` event with `ok` false. The run that
        saw it is allowed to finish, so the retry and the model's second
        attempt are part of the result. The loop is bounded by `hunt_max`
        (the server caps it at HUNT_LIMIT) and by `stop()`. Finding nothing
        is a valid result: it measures how rarely this setup slips.

        Questions rotate through the fixed list, starting at the one chosen,
        for variety. A custom question is repeated as it is.
        """
        questions = _rotate(question, self._banks())
        emit("hunt_started", max=hunt_max, questions=len(questions))
        reason, runs = "exhausted", 0
        for i in range(hunt_max):
            if self._stop.is_set():
                reason = "stopped"
                break
            q = questions[i % len(questions)]
            emit("hunt_run", index=i + 1, max=hunt_max, question=q)
            watch.invalid = False
            outcome = once(q)
            runs = i + 1
            if outcome == "error":
                reason = "error"
                break
            if watch.invalid:
                reason = "found"
                break
        emit("hunt_finished", reason=reason, runs=runs, max=hunt_max)

    def wait_for(self, index: int, timeout: float) -> tuple[list[Event], bool]:
        """Block until there are events at or past `index`, or `timeout`.

        Returns (new events, finished). `finished` is true once a terminal
        event is among those already published. An empty list with
        `finished` false means the wait timed out: send a keepalive.
        """
        with self._cond:
            def finished() -> bool:
                return any(e.kind in self._terminal for e in self._events)

            # Wake on new events, or on a finished run (so a replay past the
            # end returns at once instead of waiting out the timeout).
            self._cond.wait_for(lambda: len(self._events) > index or finished(), timeout)
            return self._events[index:], finished()

    @property
    def terminal_kinds(self) -> tuple[str, ...]:
        """Event kinds that end the current stream (set when the run starts)."""
        with self._cond:
            return self._terminal

    def has_run(self) -> bool:
        """True once any run has started, so there is something to replay."""
        with self._cond:
            return self._run_id > 0


def make_handler(
    hub: RunHub, page: Path, dev_root: Path | None = None
) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class bound to `hub` and the page path.

    `dev_root`, when given, turns on dev mode: `/version` reports a fingerprint
    of the watched files and this process's start time, and `/config` says so.
    The page polls it and reloads itself. Off by default: the plain console
    exposes no file information.
    """
    boot = time.time_ns()  # changes on every restart, even with no file change

    class Handler(BaseHTTPRequestHandler):
        """Routes, header checks and SSE framing. Logic stays in `RunHub`."""

        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            """Log requests to stderr, except two that only add noise.

            The dev page polls `/version` once a second, and some browsers ask for
            `/favicon.ico` on every load. Logged, those lines bury the ones that
            matter (`POST /run`, `GET /events`). Only those routes are quiet;
            everything else logs as before.
            """
            if self.path.split("?")[0] in ("/version", "/favicon.ico"):
                return
            super().log_message(format, *args)

        # -- helpers ------------------------------------------------------
        def _send(self, status: HTTPStatus, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: object) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _host_ok(self) -> bool:
            port = self.server.server_address[1]
            return self.headers.get("Host") in {
                f"127.0.0.1:{port}",
                f"localhost:{port}",
            }

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return origin in {
                f"http://{self.headers.get('Host')}",
            }

        # -- routes -------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "bad host"})
            url = urlparse(self.path)
            if url.path == "/":
                if not page.is_file():
                    return self._send(
                        HTTPStatus.OK, b"page not built yet", "text/plain"
                    )
                return self._send(
                    HTTPStatus.OK, page.read_bytes(), "text/html; charset=utf-8"
                )
            if url.path == "/favicon.ico":
                # The page declares an inline icon, so browsers normally never ask.
                # Some still do. Answer "no content" instead of a 404 in the log.
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if url.path == "/version" and dev_root is not None:
                return self._json(
                    HTTPStatus.OK,
                    {"version": f"{boot}:{fingerprint(snapshot(dev_root))}"},
                )
            if url.path == "/config":
                return self._json(
                    HTTPStatus.OK,
                    {
                        "questions": QUESTIONS,
                        "incidents": [
                            {"id": i.id, "title": i.title, "text": i.text}
                            for i in hub.incidents
                        ],
                        "scenarios": list(SCENARIOS),
                        "tools_available": hub.tools_available,
                        "live_available": hub.live_available,
                        "threshold": THRESHOLD,
                        "hunt_default": HUNT_DEFAULT,
                        "hunt_limit": HUNT_LIMIT,
                        "stresses": {s: DESCRIPTIONS[s] for s in STRESSES},
                        "dev_reload": dev_root is not None,
                    },
                )
            if url.path == "/events":
                return self._events(parse_qs(url.query))
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            # Early rejects leave the body unread. Close the connection, or the
            # leftover bytes are parsed as the next request on a keep-alive socket.
            self.close_connection = True
            if not self._host_ok() or not self._origin_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "bad host or origin"})
            path = urlparse(self.path).path
            if path == "/stop":
                # No body is read, and none is needed. Close, as above.
                hub.stop()
                return self._json(HTTPStatus.OK, {"stopping": True})
            if path != "/run":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send JSON"}
                )
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 < length <= _MAX_BODY:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                  {"error": "bad body size"})
            self.close_connection = False  # body is about to be read in full
            try:
                req = RunRequest.model_validate_json(self.rfile.read(length))
                run_id = hub.start(req)
            except (ValidationError, ValueError) as err:
                return self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(err)})
            except NotConfigured as err:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(err)})
            except Busy:
                return self._json(HTTPStatus.CONFLICT, {"error": "a run is in flight"})
            self._json(HTTPStatus.ACCEPTED, {"run": run_id})

        def _events(self, query: dict[str, list[str]]) -> None:
            """Stream the latest run as SSE until its terminal event."""
            if not hub.has_run():
                return self._json(HTTPStatus.NOT_FOUND, {"error": "no run yet"})
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            # No Content-Length and a closed connection mark the stream's end.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                sent = int(query.get("from", ["0"])[0])
            except ValueError:
                sent = 0
            try:
                while True:
                    new, done = hub.wait_for(sent, _KEEPALIVE_SECONDS)
                    if not new and done:
                        return  # replay past the end of a finished run
                    if not new:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    for event in new:
                        data = json.dumps(asdict(event))
                        self.wfile.write(f"id: {sent}\ndata: {data}\n\n".encode())
                        sent += 1
                    self.wfile.flush()
                    if done and new[-1].kind in hub.terminal_kinds:
                        return
            except (BrokenPipeError, ConnectionResetError):
                return  # the viewer left; the run is unaffected

    return Handler


def serve(
    hub: RunHub,
    host: str = "127.0.0.1",
    port: int = 8765,
    dev_root: Path | None = None,
) -> ThreadingHTTPServer:
    """Build the server. Port 0 picks a free port (tests use it)."""
    server = ThreadingHTTPServer((host, port), make_handler(hub, PAGE, dev_root))
    server.daemon_threads = True
    return server


DEFAULT_PORT = 8765


def parse_args(argv: list[str]) -> tuple[bool, int]:
    """Read `--reload` and `--port N`. Returns (reload, port). Pure.

    Anything else is an error: a typo should stop the run, not be ignored.
    """
    reload, port, i = False, DEFAULT_PORT, 0
    while i < len(argv):
        if argv[i] == "--reload":
            reload = True
        elif argv[i] == "--port" and i + 1 < len(argv) and argv[i + 1].isdigit():
            port, i = int(argv[i + 1]), i + 1
        else:
            raise SystemExit(f"usage: python -m app.console [--reload] [--port N] (got {argv[i]!r})")
        i += 1
    return reload, port


def lazy_recall() -> Recall:
    """The real recall step, built on first use instead of at startup.

    Building it loads two models (about 1.3 GB) and opens the database. Doing
    that at startup would slow every start and fail it when the database is
    off, though fake mode needs neither. On first use a failure becomes a
    `run_failed` event, which the page shows. The retriever is kept after
    the first build. One lock keeps two runs from building it twice.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    from app.ingest import database_url, make_embedder
    from app.retrieve import Retriever, load_passages, make_reranker, postgres_vector_search

    lock = threading.Lock()
    built: list[Retriever] = []

    def recall(question: str) -> list[Prior]:
        with lock:
            if not built:
                conn = psycopg.connect(database_url(), autocommit=True)
                register_vector(conn)
                built.append(
                    Retriever(
                        load_passages(conn),
                        postgres_vector_search(conn, make_embedder()),
                        make_reranker(),
                    )
                )
            retriever = built[0]
            found = retriever.hybrid_rerank(question, RECALL_K)
            return [retriever.brief(f.incident_id) for f in found]

    return recall


def main(argv: list[str] | None = None) -> None:
    """Read the environment, build the hub, serve until Ctrl-C.

    `--reload` runs this same command under a supervisor that restarts it when
    code changes (app/dev_reload.py). The child is started without the flag and
    with `FINDYFIXY_DEV` set, which turns on the page's auto-reload.

    Like `app.graph.main`, this is the only place that reads `.env`. Missing
    model names do not stop the server: fake mode needs none. Live mode then
    reports "not configured" instead of failing at startup.
    """
    args = sys.argv[1:] if argv is None else argv
    reload, port = parse_args(args)
    if reload:
        from app import dev_reload

        dev_reload.run(ROOT, port)
        return

    from dotenv import load_dotenv

    load_dotenv()
    small, large = os.environ.get("SMALL_MODEL"), os.environ.get("LARGE_MODEL")
    investigator = None
    if small and large:
        from langchain_ollama import ChatOllama

        tool = make_deploys_tool()
        investigator = build_investigator(
            ChatOllama(model=small, **answer_options()).bind_tools([tool]), tool
        )
    hub = RunHub(
        live_models=(small, large) if small and large else None,
        incidents=load_incidents(DEFAULT_PATH),
        investigator=investigator,
        recall=lazy_recall() if small and large else None,
    )
    dev = ROOT if os.environ.get("FINDYFIXY_DEV") else None
    server = serve(hub, port=port, dev_root=dev)
    port = server.server_address[1]
    print(f"Console: http://127.0.0.1:{port}  (Ctrl-C to stop)", flush=True)
    live = "on" if hub.live_available else "off (set SMALL_MODEL, LARGE_MODEL)"
    print(f"Live mode: {live}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
