"""Dev mode: restart the console when code changes, and reload the page.

What it does
    `supervise` runs the console as a child process and watches a set of files.
    When one changes, it restarts the child, so new Python code is picked up
    without retyping the command. `snapshot`, `diff` and `fingerprint` are the
    pure parts: they say what is on disk and what changed. The page polls the
    console's `/version` (a fingerprint plus the process start time) and
    reloads itself when it changes.

Why polling and the standard library
    A file-watching package (`watchfiles`, or `uvicorn --reload`) would add
    dependencies and, through them, supply-chain surface. Polling about thirty
    files twice a second costs almost nothing, and it works the same on every
    system. The cost: a change is noticed up to half a second late, and a file
    that changes and changes back within one poll is missed. Neither matters
    for a dev loop.

Why the child is a separate process
    A running Python process does not re-read modules it has imported. The only
    reliable way to run new code is a new process. The supervisor holds no
    sockets, so the child can bind the same port again on restart.

Debounce
    Editors and formatters often write several files in a burst. Restarting on
    the first write would start the child against half-saved code. `settle`
    waits until the snapshot has stopped changing for one full interval, then
    restarts once.

A broken edit does not crash-loop
    If the child exits on its own (a syntax error, an import error), the
    supervisor prints that and WAITS. It restarts only after the next change,
    so a typo costs one message, not a loop of failing starts.

What is not watched
    pyproject.toml and uv.lock. A new dependency needs `uv sync` first, and a
    restart cannot do that. In-flight runs die with the child. For development
    only: the plain `python -m app.console` is unchanged.
"""

import hashlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

# Relative to the project root (code/). A change here needs a NEW PROCESS: Python
# does not re-read modules it has imported, and model names are read from `.env`
# at startup. Only `.env`'s modification time is read, never its contents.
RESTART_PATTERNS = ("app/**/*.py", "data/*.json", ".env")

# The page is read from disk on every request, so editing it needs no restart,
# only a browser reload. It is part of the fingerprint the page polls, and not
# a reason to restart the server.
PAGE_PATTERNS = ("docs/*.html",)

WATCH_PATTERNS = RESTART_PATTERNS + PAGE_PATTERNS

Snapshot = dict[str, int]


class Child(Protocol):
    """The part of `subprocess.Popen` the supervisor uses."""

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def snapshot(root: Path, patterns: Sequence[str] = WATCH_PATTERNS) -> Snapshot:
    """Map each watched file (relative path) to its modification time in ns.

    A file that disappears between listing and stat is skipped, not an error:
    editors save by writing a temp file and renaming it.
    """
    out: Snapshot = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            try:
                out[str(path.relative_to(root))] = path.stat().st_mtime_ns
            except OSError:
                continue
    return out


def diff(old: Snapshot, new: Snapshot) -> list[str]:
    """Paths that were added, removed or modified. Pure. Sorted."""
    return sorted(
        p for p in old.keys() | new.keys() if old.get(p) != new.get(p)
    )


def fingerprint(snap: Snapshot) -> str:
    """A short stable hash of a snapshot. Changes when any watched file does."""
    blob = "\n".join(f"{p}:{t}" for p, t in sorted(snap.items()))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def wait_until_settled(
    read: Callable[[], Snapshot],
    current: Snapshot,
    *,
    interval: float,
    sleep: Callable[[float], None],
) -> Snapshot:
    """Return the snapshot once it has stopped changing for one interval.

    `current` is the latest reading, already known to differ from the one the
    child started with. Keep sleeping while it keeps changing.
    """
    while True:
        sleep(interval)
        newer = read()
        if newer == current:
            return newer
        current = newer


def _stop(child: Child, *, timeout: float = 3.0) -> None:
    """Ask the child to exit, and kill it if it does not."""
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def supervise(
    cmd: Sequence[str],
    root: Path,
    *,
    poll: float = 0.5,
    settle: float = 0.4,
    env: dict[str, str] | None = None,
    spawn: Callable[..., Child] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    read: Callable[[], Snapshot] | None = None,
    out: Callable[[str], None] = lambda m: print(m, flush=True),
) -> None:
    """Run `cmd`, restart it on file changes, stop it on Ctrl-C.

    Everything that touches the world is injected (`spawn`, `sleep`, `read`,
    `out`), so the tests drive it with fakes and no real processes or time.
    The loop ends when `sleep` raises KeyboardInterrupt.
    """
    read = read or (lambda: snapshot(root, RESTART_PATTERNS))
    start = lambda: spawn(list(cmd), cwd=str(root), env=env)  # noqa: E731
    snap = read()
    child = start()
    crashed = False
    try:
        while True:
            sleep(poll)
            now = read()
            if now != snap:
                now = wait_until_settled(read, now, interval=settle, sleep=sleep)
                out(f"[reload] changed: {', '.join(diff(snap, now))}. Restarting.")
                _stop(child)
                snap, crashed, child = now, False, start()
            elif child.poll() is not None and not crashed:
                crashed = True
                out(
                    f"[reload] server exited with code {child.poll()}. "
                    "Waiting for a change to try again."
                )
    except KeyboardInterrupt:
        _stop(child)
        out("[reload] stopped.")


def dev_env() -> dict[str, str]:
    """The child's environment: ours, plus the flag that turns on /version."""
    return {**os.environ, "FINDYFIXY_DEV": "1"}


def _stop_on_signal(signum: int, frame: object) -> None:
    """Turn SIGTERM and SIGHUP into the same clean stop as Ctrl-C.

    Without this, killing the supervisor (or closing its terminal) would end
    it at once and leave the server it started running and holding the port.
    """
    raise KeyboardInterrupt


def run(root: Path, port: int) -> None:
    """Start the console under the supervisor. Used by `console --reload`."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _stop_on_signal)
    out = sys.stdout
    out.write("[reload] restarts on app/, data/ and .env; docs/ only reloads the page. Ctrl-C to stop.\n")
    out.flush()
    supervise(
        [sys.executable, "-m", "app.console", "--port", str(port)],
        root,
        env=dev_env(),
    )
