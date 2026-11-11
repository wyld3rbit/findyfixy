"""Scores on traces: the seam, and a Langfuse implementation that sends in the background.

What it does
    `ScoreSink` is what the eval runner calls with one score for one trace.
    `LangfuseScores` queues each score and a background thread POSTs it to
    Langfuse's `/api/public/scores`. `close()` sends what is left, within a
    time limit.

Why a seam
    Spans are portable (OTLP), scores are not: OpenTelemetry has no standard
    for "this trace scored 0.8 on faithfulness". Each backend has its own API.
    The runner only knows `ScoreSink`, so another backend is a new class here,
    and the runner and the app do not change.

Failure model
    A score that cannot be sent is dropped and counted (`dropped`), never
    raised: an eval run's result is its report, and the scores in the backend
    are a view of it. The queue is bounded (`MAX_QUEUE`): when the backend is
    down the thread keeps failing fast and new scores drop, but the run never
    waits. Each score has a deterministic id (trace id and score name), so a
    resent score replaces the old one instead of adding a duplicate.

    A failed metric (score None) is never sent as a score: 0 would say the
    answer was bad when the judge never answered. The runner marks it on the
    metric's span instead.

Credentials
    The key pair travels only in the Authorization header. Nothing here
    prints it; `TraceConfig.__repr__` hides it.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Literal, Protocol

from harness.trace.tracing import EXPORT_TIMEOUT_S, TraceConfig

ScoreType = Literal["NUMERIC", "BOOLEAN"]
MAX_QUEUE = 10_000
SCORES_PATH = "/api/public/scores"


class ScoreSink(Protocol):
    """Where the eval runner sends a score. Must not block or raise.

    The app's composition root builds the sink and owns its lifetime; the
    harness only calls `score`.
    """

    def score(self, trace_id: str, name: str, value: float, kind: ScoreType, comment: str | None = None) -> None:
        """Record `value` as score `name` on trace `trace_id`."""
        ...


class LangfuseScores:
    """Sends scores to Langfuse from one background thread. Start it, use it, `close` it."""

    def __init__(self, config: TraceConfig, *, timeout_s: float = EXPORT_TIMEOUT_S) -> None:
        """Start the sending thread. Nothing is sent until `score` is called."""
        self._config = config
        self._timeout_s = timeout_s
        self._queue: queue.Queue[dict[str, object] | None] = queue.Queue(MAX_QUEUE)
        self.sent = 0
        self.dropped = 0
        self._count_lock = threading.Lock()  # `dropped` is counted from callers' threads too
        self._thread = threading.Thread(target=self._run, name="langfuse-scores", daemon=True)
        self._thread.start()

    def score(self, trace_id: str, name: str, value: float, kind: ScoreType, comment: str | None = None) -> None:
        """Queue one score. Returns at once; a full queue drops it."""
        body: dict[str, object] = {
            "id": f"{trace_id}-{name}",  # deterministic: a resend replaces, never duplicates
            "traceId": trace_id,
            "name": name,
            "value": value,
            "dataType": kind,
        }
        if comment:
            body["comment"] = comment
        try:
            self._queue.put_nowait(body)
        except queue.Full:
            with self._count_lock:
                self.dropped += 1

    def _run(self) -> None:
        """Send queued scores until `close` puts the stop marker."""
        while True:
            body = self._queue.get()
            if body is None:
                return
            try:
                self._config.request("POST", SCORES_PATH, body, timeout_s=self._timeout_s)
                self.sent += 1
            except Exception:  # noqa: BLE001 - any failure drops this score; the thread must live on
                with self._count_lock:
                    self.dropped += 1

    def close(self, timeout_s: float = 30.0) -> None:
        """Send what is queued, then stop. Waits about `timeout_s` in all; what is left is counted as dropped.

        Counting the leftovers keeps `dropped` true when the backend is slow:
        the caller's warning then says how many scores are missing.
        """
        deadline = time.monotonic() + timeout_s
        try:
            self._queue.put(None, timeout=timeout_s)
        except queue.Full:
            pass
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive() or not self._queue.empty():
            left = sum(1 for item in list(self._queue.queue) if item is not None)
            with self._count_lock:
                self.dropped += left
