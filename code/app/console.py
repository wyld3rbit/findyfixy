"""Day 2: the live console server. Run the real graph, stream its events.

What it does
    Serves the architecture page and runs the triage graph on request. While
    a run is in flight it streams every event (app/events.py) to the browser
    over Server-Sent Events (SSE). The page animates from those events.

    It also runs the evaluation harness on the first N golden cases (an "eval
    run", app/console_eval.py). The stream then carries the harness's events
    (harness/eval/events.py) next to the graph's events for each case. Every
    streamed event is tagged with its `source`, "app" or "harness", so the
    page can tell the two vocabularies apart.

        GET  /         the page (docs/architecture.html)
        GET  /config   questions, fake scenarios, which modes are available
        POST /run      start one run (or a hunt): {"mode", "question"?, "scenario"?}
        POST /eval     start one eval run: {"mode", "cases": 1..N}, N the set's size
        POST /stop     ask a hunt or an eval run to stop after its current run or
                       case; answers {"stopping": false} when nothing can stop
        GET  /events   SSE stream of the latest run, replayed from the start

    The console is a viewer. An eval run here writes no report and no
    baseline; it applies the gate read-only, and only when every case ran.

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
    - An eval run's case count must be a JSON integer from 1 to the size of
      the chosen set (the golden set live, the invented set fake). Strict:
      "2", 2.0 and true are refused, and so is any unknown field.
    - One run at a time, triage or eval. Two 7B-class runs would not fit in 16 GB.

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
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from app.console_eval import (
    EVAL_DEFAULT,
    EvalRun,
    HarnessEvent,
    EvalSuite,
    live_eval,
    run_and_judge,
)
from app.console_fakes import (
    FAKE_QUESTION,
    FAKE_RECALL,
    SCENARIOS,
    TOOL_HOURS,
    Scenario,
    ScriptedToolModel,
    fake_ask,
    fake_eval,
    inject_bad_first_reply,
)
from app.dev_reload import fingerprint, snapshot
from app.events import Emit, Event, make_emit, make_stamped_emit
from app.incidents import DEFAULT_PATH, Incident, load_incidents
from app.paths import DOCS, EVALS, ROOT
from app.settings import load_env, optional_models
from app.wiring import SharedRetrieval
from app.investigate import build_investigator
from app.stress import DESCRIPTIONS, STRESSES, Stress, make_ollama_ask
from app.tools import make_deploys_tool
from app.graph import QUESTIONS, THRESHOLD, Recall, build_graph
from app.llm import answer_options, ollama_ask
from app.triage import RECALL_K, Ask, TriageError

Mode = Literal["live", "fake"]

# The page lives next to the docs (app/paths.py resolves from the file, never
# from the current directory, so the server works from any clone).
PAGE = DOCS / "architecture.html"

# Realistic incidents are log excerpts of a few thousand characters (see
# app/incidents.py), so the question cap is far above a one-line question.
_MAX_BODY = 32 * 1024
_MAX_QUESTION = 8000
_KEEPALIVE_SECONDS = 10.0

_FORCE_THRESHOLD = 1.01


# A hunt repeats real model runs, so the cap is a cost and time bound. The
# server enforces it. The page default is lower than the limit on purpose.
HUNT_DEFAULT = 20
HUNT_LIMIT = 50

_RUN_TERMINAL = ("run_finished", "run_failed")
_HUNT_TERMINAL = ("hunt_finished",)
# Harness kinds. The app and harness vocabularies share no kind name
# (tests/test_page_truth.py checks it), so a kind alone says which run ended.
_EVAL_TERMINAL = ("eval_finished", "eval_failed")

# Read-only: the console compares a full eval run to it and never writes it.
BASELINE_PATH = EVALS / "baseline.json"

StreamEvent = Event | HarnessEvent


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


class EvalRequest(BaseModel):
    """The body of POST /eval, validated before anything starts.

    `cases` is a strict int: the first N cases run, so a string or a float
    has no meaning here and is refused rather than coerced. Unknown fields are
    refused too, so a typo cannot silently fall back to a default. The upper
    bound depends on the mode's case set, so `RunHub.start_eval` checks it.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Mode
    cases: StrictInt = Field(ge=1)


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
        live_eval: EvalSuite | None = None,
        baseline_path: Path = BASELINE_PATH,
    ) -> None:
        """Hold the wiring. Nothing runs until `start` or `start_eval`.

        `live_eval` is the real eval run and its golden set's size (built in
        `main` when models are configured); None means live eval runs are
        refused as not configured. Fake eval runs are always available and
        need no model.
        """
        self._live_models = live_models
        self._live_eval = live_eval
        self._fake_eval = fake_eval(sleep, fake_delay)
        self._baseline_path = baseline_path
        self.incidents = incidents or []
        self._investigator = investigator
        # When set, live runs use it and fake runs use `FAKE_RECALL`. When None
        # no run has a recall step, and the graph is what it was before Day 4.
        self._recall = recall
        self._sleep = sleep
        self._fake_delay = fake_delay
        self._clock = clock
        self._cond = threading.Condition()
        self._events: list[StreamEvent] = []
        self._running = False
        self._run_id = 0
        self._stop = threading.Event()
        self._terminal: tuple[str, ...] = _RUN_TERMINAL
        # Set by `_publish` when a terminal event lands, so `wait_for` reads a
        # flag instead of rescanning every event on each wake.
        self._finished = False
        # True while a run that honours /stop (a hunt, an eval) is in flight.
        self._stoppable = False

    @property
    def tools_available(self) -> bool:
        """True when a real tool step was configured (live mode only)."""
        return self._investigator is not None

    @property
    def live_available(self) -> bool:
        """True when `SMALL_MODEL` and `LARGE_MODEL` were configured."""
        return self._live_models is not None

    def eval_sizes(self) -> dict[str, int | None]:
        """How many cases each mode's eval set has. None for live when it is not configured."""
        return {"live": self._live_eval.size if self._live_eval else None, "fake": self._fake_eval.size}

    def _begin(self, terminal: tuple[str, ...], *, stoppable: bool) -> int:
        """Claim the one run slot and reset the stream. Raises `Busy`. Returns the run id.

        Shared by triage and eval runs, so neither can start while the other is
        in flight. The terminal kinds are set here, under the lock, so a viewer
        never reads a stream with the previous run's end condition.
        """
        with self._cond:
            if self._running:
                raise Busy
            self._running = True
            self._run_id += 1
            self._events = []
            self._stop.clear()
            self._terminal = terminal
            self._finished = False
            self._stoppable = stoppable
            return self._run_id

    def start_eval(self, req: EvalRequest) -> int:
        """Start an eval run of the first `req.cases` cases in a thread. Raises `Busy`.

        Live needs the real wiring (`NotConfigured` otherwise). Fake runs the
        real harness on invented cases with scripted models. A count above the
        chosen set's size is a `ValueError` (422), checked before the slot is
        taken.
        """
        if req.mode == "live":
            if self._live_eval is None:
                raise NotConfigured("live eval needs SMALL_MODEL and LARGE_MODEL")
            suite = self._live_eval
        else:
            suite = self._fake_eval
        if req.cases > suite.size:
            raise ValueError(f"cases must be between 1 and {suite.size} for {req.mode} mode")
        run_id = self._begin(_EVAL_TERMINAL, stoppable=True)
        threading.Thread(target=self._eval, args=(suite.run, req.cases), daemon=True).start()
        return run_id

    def _eval(self, run: EvalRun, n: int) -> None:
        """The eval run thread. Always ends with exactly one terminal event.

        Two emitters feed one stream: the graph's events (app vocabulary) and
        the runner's (harness vocabulary). Both stamp time from this moment,
        with the same emitter code. The runner asks the stop flag before each
        case. `_running` is cleared last, as in `_run`.
        """
        emit = make_emit(self._publish, self._clock)
        on_event = make_stamped_emit(self._publish, self._clock, HarnessEvent)
        try:
            run_and_judge(run, n, emit=emit, on_event=on_event, should_stop=self._stop.is_set,
                          baseline_path=self._baseline_path)
        finally:
            with self._cond:
                self._running = False
                self._cond.notify_all()

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
                    ScriptedToolModel(TOOL_HOURS[req.scenario]),
                    make_deploys_tool(),
                )

        run_id = self._begin(_HUNT_TERMINAL if req.hunt_max else _RUN_TERMINAL, stoppable=bool(req.hunt_max))

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

    def _publish(self, event: StreamEvent) -> None:
        """Sink for the run's emitter: append and wake the SSE handlers."""
        with self._cond:
            self._events.append(event)
            if event.kind in self._terminal:
                self._finished = True
            self._cond.notify_all()

    def _banks(self) -> list[list[str]]:
        """The question banks a hunt can rotate: the lesson five, then incidents."""
        return [QUESTIONS, [i.text for i in self.incidents]]

    def stop(self) -> bool:
        """Ask a running hunt or eval run to stop after its current run or case.

        Returns True when such a run is in flight and will stop, False when
        there is nothing that can stop (idle, or a single triage run), so the
        page can say what really happens.
        """
        with self._cond:
            if not (self._running and self._stoppable):
                return False
            self._stop.set()
            return True

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

    def wait_for(self, index: int, timeout: float) -> tuple[list[StreamEvent], bool]:
        """Block until there are events at or past `index`, or `timeout`.

        Returns (new events, finished). `finished` is true once a terminal
        event is among those already published. An empty list with
        `finished` false means the wait timed out: send a keepalive.
        """
        with self._cond:
            # Wake on new events, or on a finished run (so a replay past the
            # end returns at once instead of waiting out the timeout).
            self._cond.wait_for(lambda: len(self._events) > index or self._finished, timeout)
            return self._events[index:], self._finished

    @property
    def terminal_kinds(self) -> tuple[str, ...]:
        """Event kinds that end the current stream (set when the run starts)."""
        with self._cond:
            return self._terminal

    def has_run(self) -> bool:
        """True once any run has started, so there is something to replay."""
        with self._cond:
            return self._run_id > 0


def wire(event: StreamEvent) -> dict[str, object]:
    """One stream event as the JSON object the page reads, tagged with its source.

    The tag is added here, at the edge, so neither vocabulary needs to know
    about the other or about the page.
    """
    source = "harness" if isinstance(event, HarnessEvent) else "app"
    return {"source": source, **asdict(event)}


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
                        "eval_sizes": hub.eval_sizes(),
                        "eval_default": EVAL_DEFAULT,
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
                return self._json(HTTPStatus.OK, {"stopping": hub.stop()})
            if path not in ("/run", "/eval"):
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
                body = self.rfile.read(length)
                if path == "/eval":
                    run_id = hub.start_eval(EvalRequest.model_validate_json(body))
                else:
                    run_id = hub.start(RunRequest.model_validate_json(body))
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
                        data = json.dumps(wire(event))
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

    load_env()
    models = optional_models()
    investigator = None
    if models:
        from langchain_ollama import ChatOllama

        tool = make_deploys_tool()
        investigator = build_investigator(
            ChatOllama(model=models.small, **answer_options()).bind_tools([tool]), tool
        )
    # One database connection, embedder and reranker for the whole process:
    # triage recall and the eval session share it, so a live eval loads no copy.
    retrieval = SharedRetrieval() if models else None
    live_eval_suite = None
    if retrieval is not None:
        from app.eval_answers import GOLDEN_PATH, open_live_session
        from app.golden import load_golden

        live_eval_suite = live_eval(
            partial(open_live_session, retrieval=retrieval),
            golden_path=GOLDEN_PATH,
            repo_dir=ROOT,
            progress=lambda msg: print(msg, file=sys.stderr, flush=True),
            size=len(load_golden(GOLDEN_PATH)),
        )
    hub = RunHub(
        live_models=(models.small, models.large) if models else None,
        incidents=load_incidents(DEFAULT_PATH),
        investigator=investigator,
        recall=retrieval.recall(RECALL_K) if retrieval else None,
        live_eval=live_eval_suite,
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
