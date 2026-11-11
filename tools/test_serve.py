"""Tests for tools/serve.py.

What they cover
    Each route answers with the right type and content. Unknown lessons and
    paths are 404. A foreign Host header is refused (DNS rebinding). Writes are
    refused. Asset paths cannot escape assets/ by `..`, by a symlink, or by
    naming a directory.

Real server on a free loopback port, real files. No fakes needed: nothing here
is slow or non-deterministic.

Run from the training root:
    uv run --no-project --with markdown --with pytest pytest tools/test_serve.py
"""

import http.client
import json
import threading
from collections.abc import Iterator

import pytest

import serve


@pytest.fixture()
def conn() -> Iterator[http.client.HTTPConnection]:
    """A client connected to a throwaway server."""
    server = serve.make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    yield client
    client.close()
    server.shutdown()
    server.server_close()


def get(client: http.client.HTTPConnection, path: str, headers: dict[str, str] | None = None):
    client.request("GET", path, headers=headers or {})
    resp = client.getresponse()
    return resp.status, resp.getheader("Content-Type") or "", resp.read()


def test_root_serves_the_page(conn):
    status, ctype, body = get(conn, "/")
    assert status == 200
    assert ctype.startswith("text/html")
    assert b"fetch(\"/api/progress\"" in body


def test_progress_is_the_file_as_json(conn):
    status, ctype, body = get(conn, "/api/progress")
    assert status == 200
    assert ctype == "application/json"
    assert json.loads(body) == json.loads(serve.PROGRESS.read_text())


def test_lesson_renders_markdown_without_self_check(conn):
    status, ctype, body = get(conn, "/api/lesson/5")
    assert status == 200
    assert ctype.startswith("text/html")
    assert b"<h2>Concept" in body
    assert b"Self-check" not in body


@pytest.mark.parametrize("tail", ["11", "-1", "abc", "5/extra", ""])
def test_unknown_lesson_is_404(conn, tail):
    status, _, _ = get(conn, f"/api/lesson/{tail}")
    assert status == 404


def test_foreign_host_is_refused(conn):
    status, _, _ = get(conn, "/api/progress", {"Host": "evil.example"})
    assert status == 403


def test_writes_are_refused(conn):
    conn.request("POST", "/api/progress", body=b"{}")
    assert conn.getresponse().status == 405


def test_traversal_over_http_is_404(conn):
    status, _, _ = get(conn, "/assets/../record/progress.json")
    assert status == 404


def test_security_headers_present(conn):
    conn.request("GET", "/api/progress")
    resp = conn.getresponse()
    resp.read()
    assert resp.getheader("X-Content-Type-Options") == "nosniff"
    csp = resp.getheader("Content-Security-Policy") or ""
    assert "default-src 'self'" in csp
    # The visualization is the only other origin, for frames and the probe.
    assert "frame-src http://127.0.0.1:8765 http://localhost:8765" in csp
    assert "connect-src 'self' http://127.0.0.1:8765" in csp
    assert "script-src 'self' 'unsafe-inline'" in csp and "http://127.0.0.1:8765" not in csp.split("script-src")[1].split(";")[0]
    assert resp.getheader("Cache-Control") == "no-store"


def test_asset_inside_is_found(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    assert serve.asset_path("a.png", tmp_path) == (tmp_path / "a.png").resolve()


def test_asset_dotdot_is_refused(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "secret.txt").write_text("s")
    assert serve.asset_path("../secret.txt", assets) is None


def test_asset_symlink_escape_is_refused(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "secret.txt").write_text("s")
    (assets / "link.txt").symlink_to(tmp_path / "secret.txt")
    assert serve.asset_path("link.txt", assets) is None


def test_asset_directory_is_refused(tmp_path):
    (tmp_path / "sub").mkdir()
    assert serve.asset_path("sub", tmp_path) is None


def test_adr_index_lists_each_record_with_its_day(conn):
    """Every record is listed, and a rebuilt phase ("Day 5, rebuilt") still maps to its day."""
    status, ctype, body = get(conn, "/api/adrs")
    assert status == 200 and ctype == "application/json"
    by_n = {a["n"]: a for a in json.loads(body)}
    assert by_n[1]["day"] == 1 and by_n[6]["day"] == 5
    assert all(a["title"] and isinstance(a["day"], int) for a in by_n.values())


def test_adr_renders_with_its_diagram_and_no_dead_links(conn):
    """The diagram points at the image route; links to repo files become plain text."""
    status, ctype, body = get(conn, "/api/adr/6")
    html = body.decode()
    assert status == 200 and ctype.startswith("text/html")
    assert 'src="/api/adr-img/0006-architecture.svg"' in html
    assert "<h1" not in html and 'href="0005-' not in html and "../../../lessons" not in html
    assert "ADR 0005" in html  # the link text stays


def test_adr_diagram_is_served_as_svg(conn):
    """The image route serves exactly the record diagrams."""
    status, ctype, body = get(conn, "/api/adr-img/0001-architecture.svg")
    assert status == 200 and ctype == "image/svg+xml" and b"<svg" in body


@pytest.mark.parametrize("path", ["/api/adr/0", "/api/adr/abc", "/api/adr/9999",
                                  "/api/adr-img/../README.md", "/api/adr-img/0001-architecture.png",
                                  "/api/adr-img/%2e%2e%2fREADME.md"])
def test_unknown_adr_or_image_is_404(conn, path):
    """No other file under code/docs can be reached through these routes."""
    assert get(conn, path)[0] == 404
