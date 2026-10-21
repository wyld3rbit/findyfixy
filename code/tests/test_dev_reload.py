"""Tests for dev-mode reload (app/dev_reload.py) and its console hooks.

What these tests cover
    The pure parts (what is on disk, what changed, a stable fingerprint), and
    the supervisor's decisions: restart once after a burst of saves, wait and
    do not crash-loop after a broken edit, and stop the child on Ctrl-C.

Why fakes
    `supervise` takes its process, clock, file reader and output as arguments.
    Here the child is a stub, `sleep` returns instantly (and ends the loop by
    raising KeyboardInterrupt, the real exit path), and `read` replays a script
    of snapshots. No real process starts and nothing waits.

What a staff+ reader should notice
    - The burst test asserts ONE restart for several quick writes. Without the
      debounce the child would start against half-saved code.
    - The crash test asserts the supervisor does not respawn a child that exits
      on its own. A typo in an edit must cost one message, not a restart loop.
"""

import json
import subprocess
import threading
from pathlib import Path

import pytest

from app.console import RunHub, serve
from app.dev_reload import (
    RESTART_PATTERNS,
    WATCH_PATTERNS,
    diff,
    fingerprint,
    snapshot,
    supervise,
    wait_until_settled,
)

from test_console import call  # noqa: E402  (helper shared with the console tests)


# ---- pure parts -------------------------------------------------------------


def make_tree(root: Path) -> None:
    (root / "app" / "sub").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "data").mkdir()
    (root / "app" / "a.py").write_text("a = 1\n")
    (root / "app" / "sub" / "b.py").write_text("b = 1\n")
    (root / "docs" / "page.html").write_text("<p>x</p>")
    (root / "data" / "d.json").write_text("[]")
    (root / "notes.txt").write_text("not watched")


def test_snapshot_lists_the_watched_files_only(tmp_path):
    make_tree(tmp_path)
    assert sorted(snapshot(tmp_path)) == [
        "app/a.py", "app/sub/b.py", "data/d.json", "docs/page.html"
    ]


def test_editing_the_page_does_not_trigger_a_restart_but_does_change_the_version(tmp_path):
    # The page is read per request: no restart needed. But the page polls the
    # fingerprint of everything, so it must still notice and reload itself.
    import os

    make_tree(tmp_path)
    restart_before = snapshot(tmp_path, RESTART_PATTERNS)
    watch_before = snapshot(tmp_path, WATCH_PATTERNS)
    os.utime(tmp_path / "docs" / "page.html", ns=(7, 7))
    assert snapshot(tmp_path, RESTART_PATTERNS) == restart_before
    assert fingerprint(snapshot(tmp_path, WATCH_PATTERNS)) != fingerprint(watch_before)
    os.utime(tmp_path / "app" / "a.py", ns=(7, 7))
    assert snapshot(tmp_path, RESTART_PATTERNS) != restart_before


def test_diff_reports_added_removed_and_modified(tmp_path):
    make_tree(tmp_path)
    before = snapshot(tmp_path)
    (tmp_path / "app" / "a.py").write_text("a = 2\n")
    import os
    os.utime(tmp_path / "app" / "a.py", ns=(1, 1))        # modified
    (tmp_path / "app" / "new.py").write_text("")           # added
    (tmp_path / "data" / "d.json").unlink()                 # removed
    assert diff(before, snapshot(tmp_path)) == ["app/a.py", "app/new.py", "data/d.json"]


def test_diff_of_identical_snapshots_is_empty():
    assert diff({"a": 1}, {"a": 1}) == []


def test_fingerprint_is_stable_and_changes_with_any_file():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    assert fingerprint({"a": 1}) != fingerprint({"a": 1, "b": 1})


def test_wait_until_settled_returns_only_when_readings_stop_changing():
    readings = iter([{"a": 2}, {"a": 3}, {"a": 3}])
    sleeps: list[float] = []
    out = wait_until_settled(lambda: next(readings), {"a": 1},
                             interval=0.4, sleep=sleeps.append)
    assert out == {"a": 3} and sleeps == [0.4, 0.4, 0.4]


# ---- the supervisor with fakes ----------------------------------------------


class FakeChild:
    def __init__(self, exit_code=None, ignore_terminate=False):
        self.exit_code = exit_code
        self.terminated = self.killed = False
        self._ignore = ignore_terminate

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        if not self._ignore:
            self.exit_code = -15

    def kill(self):
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout=None):
        if self._ignore and not self.killed:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.exit_code


class Harness:
    """Runs `supervise` against scripted snapshots and fake children."""

    def __init__(self, reads, children, polls):
        self._reads = list(reads)
        self.children: list[FakeChild] = []
        self._to_spawn = list(children)
        self.out: list[str] = []
        self._polls = polls
        self.sleeps = 0

    def read(self):
        return self._reads.pop(0) if len(self._reads) > 1 else self._reads[0]

    def spawn(self, cmd, **kw):
        child = self._to_spawn.pop(0) if self._to_spawn else FakeChild()
        self.children.append(child)
        return child

    def sleep(self, seconds):
        self.sleeps += 1
        if self.sleeps > self._polls:
            raise KeyboardInterrupt

    def run(self):
        supervise(["x"], Path("."), spawn=self.spawn, sleep=self.sleep,
                  read=self.read, out=self.out.append)


A, B, C = {"f.py": 1}, {"f.py": 2}, {"f.py": 3}


def test_a_burst_of_saves_causes_exactly_one_restart():
    # initial A; then B (changed), C (still changing), C, C (settled).
    h = Harness(reads=[A, A, B, C, C, C], children=[FakeChild(), FakeChild()], polls=8)
    h.run()
    assert len(h.children) == 2                    # the original, and ONE restart
    assert h.children[0].terminated
    changes = [m for m in h.out if "changed" in m]
    assert len(changes) == 1 and "f.py" in changes[0]


def test_no_change_means_no_restart():
    h = Harness(reads=[A], children=[FakeChild()], polls=5)
    h.run()
    assert len(h.children) == 1
    assert not [m for m in h.out if "changed" in m]


def test_a_child_that_exits_on_its_own_is_not_respawned():
    h = Harness(reads=[A], children=[FakeChild(exit_code=1)], polls=6)
    h.run()
    assert len(h.children) == 1                    # no crash loop
    assert sum("exited with code 1" in m for m in h.out) == 1   # said once only


def test_after_a_crash_the_next_change_starts_a_fresh_child():
    h = Harness(reads=[A, A, B, B], children=[FakeChild(exit_code=1), FakeChild()], polls=6)
    h.run()
    assert len(h.children) == 2
    assert [("exited" in m, "changed" in m) for m in h.out[:2]] == [(True, False), (False, True)]


def test_ctrl_c_stops_the_child():
    h = Harness(reads=[A], children=[FakeChild()], polls=0)
    h.run()
    assert h.children[0].terminated
    assert h.out[-1] == "[reload] stopped."


def test_a_child_that_ignores_terminate_is_killed():
    h = Harness(reads=[A], children=[FakeChild(ignore_terminate=True)], polls=0)
    h.run()
    assert h.children[0].killed


# ---- the console hooks ------------------------------------------------------


def start(dev_root):
    hub = RunHub(live_models=None, sleep=lambda s: None, fake_delay=0)
    srv = serve(hub, port=0, dev_root=dev_root)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_version_is_not_exposed_unless_dev_mode_is_on():
    srv = start(None)
    try:
        port = srv.server_address[1]
        assert call(port, "GET", "/version")[0] == 404
        assert json.loads(call(port, "GET", "/config")[1])["dev_reload"] is False
    finally:
        srv.shutdown(); srv.server_close()


def test_version_changes_when_a_watched_file_changes(tmp_path):
    make_tree(tmp_path)
    srv = start(tmp_path)
    try:
        port = srv.server_address[1]
        assert json.loads(call(port, "GET", "/config")[1])["dev_reload"] is True
        v1 = json.loads(call(port, "GET", "/version")[1])["version"]
        assert json.loads(call(port, "GET", "/version")[1])["version"] == v1  # stable
        import os
        os.utime(tmp_path / "docs" / "page.html", ns=(5, 5))
        v2 = json.loads(call(port, "GET", "/version")[1])["version"]
        assert v2 != v1
    finally:
        srv.shutdown(); srv.server_close()


def test_the_version_differs_after_a_restart_even_with_no_file_change(tmp_path):
    # A new server process has a new boot token, so the page reloads.
    make_tree(tmp_path)
    first, second = start(tmp_path), start(tmp_path)
    try:
        v = lambda s: json.loads(call(s.server_address[1], "GET", "/version")[1])["version"]
        assert v(first) != v(second)
    finally:
        for s in (first, second):
            s.shutdown(); s.server_close()


def test_the_page_waits_for_a_run_to_end_before_reloading():
    html = (Path(__file__).resolve().parents[1] / "docs" / "architecture.html").read_text()
    assert "startDevReload" in html and "reloadPending" in html
    assert "if (running) reloadPending = true" in html


def test_command_line_arguments():
    from app.console import DEFAULT_PORT, parse_args

    assert parse_args([]) == (False, DEFAULT_PORT)
    assert parse_args(["--reload"]) == (True, DEFAULT_PORT)
    assert parse_args(["--port", "9001", "--reload"]) == (True, 9001)
    for bad in (["--prot", "1"], ["--port"], ["--port", "x"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_the_version_poll_is_not_logged_but_other_requests_still_are(tmp_path, capsys):
    # The page polls /version every second in dev mode. Logging each poll would
    # bury the lines that matter.
    make_tree(tmp_path)
    srv = start(tmp_path)
    try:
        port = srv.server_address[1]
        for _ in range(3):
            call(port, "GET", "/version")
        call(port, "GET", "/config")
        err = capsys.readouterr().err
        assert "/config" in err
        assert "/version" not in err
    finally:
        srv.shutdown(); srv.server_close()
