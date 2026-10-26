"""Local server for the tracker. Standard library plus `markdown` (tools/lessons.py).

Why: the page holds code only. The record and the lessons are files on disk, and
browsers will not read files from a page opened by double-click (file://). This
server is the only reader of those files.

Run from the training root:
    uv run --no-project --with markdown python tools/serve.py [port]
then open http://127.0.0.1:8766

Routes (GET only):
    /                    tracker.html
    /api/progress        record/progress.json, never cached
    /api/lesson/<n>      lesson n as HTML, rendered from its .md file
    /assets/<path>       media under assets/, for lessons and the record

Security. It binds to 127.0.0.1 only. It checks the Host header, so a web page
on another site cannot reach it through DNS rebinding (the attacker's name
resolves to 127.0.0.1 but the Host header keeps the attacker's name). Asset
paths are resolved and must stay inside assets/. Nothing is writable.
What changes at scale: this is a single-user tool. A shared deployment would
need auth, TLS and a real web server in front.
"""

import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lessons import LESSON_NUMBERS, ROOT, render

DEFAULT_PORT = 8766
PROGRESS = ROOT / "record" / "progress.json"
PAGE = ROOT / "tracker.html"
ASSETS = ROOT / "assets"

# Port of the live visualization (code/app/console.py). The page embeds it in a
# frame and probes it with fetch, so both need the origin allowed, and only it.
CONSOLE_ORIGINS = "http://127.0.0.1:8765 http://localhost:8765"

# The page has inline script and style by design. Everything else is same-origin
# except the visualization above.
CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; "
    f"connect-src 'self' {CONSOLE_ORIGINS}; frame-src {CONSOLE_ORIGINS}"
)


def asset_path(rel: str, assets: Path = ASSETS) -> Path | None:
    """Resolve `rel` under `assets`, or None if it escapes or is not a file.

    Resolving first means `..` and symlinks are followed before the check, so
    `assets/../record/progress.json` cannot slip through as a prefix match.
    """
    base = assets.resolve()
    target = (base / rel).resolve()
    if base in target.parents and target.is_file():
        return target
    return None


def make_handler(port: int) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class bound to the port it is served on."""
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        """Read-only routes. Any other method is refused."""

        def _send(self, status: int, body: bytes, ctype: str, *, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _text(self, status: int, message: str) -> None:
            self._send(status, message.encode(), "text/plain; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 (name fixed by the base class)
            """Route a GET. Unknown paths are 404, bad Host headers 403."""
            if self.headers.get("Host") not in allowed_hosts:
                return self._text(403, "bad host")
            path = self.path.split("?", 1)[0]
            if path in ("/", "/tracker.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/progress":
                return self._send(200, PROGRESS.read_bytes(), "application/json")
            if path.startswith("/api/lesson/"):
                tail = path.removeprefix("/api/lesson/")
                if tail.isdecimal() and int(tail) in LESSON_NUMBERS:
                    return self._send(200, render(int(tail)).encode(), "text/html; charset=utf-8")
                return self._text(404, "no such lesson")
            if path.startswith("/assets/"):
                found = asset_path(path.removeprefix("/assets/"))
                if found:
                    ctype = mimetypes.guess_type(found.name)[0] or "application/octet-stream"
                    return self._send(200, found.read_bytes(), ctype, cache=True)
            return self._text(404, "not found")

        def _refuse(self) -> None:
            self._text(405, "read only")

        do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Keep the terminal quiet. Errors still surface as HTTP statuses."""

    return Handler


def make_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create the server bound to loopback. Port 0 picks a free one (tests use it)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(port))
    if port == 0:
        # The handler needs the real port for the Host check.
        server.RequestHandlerClass = make_handler(server.server_address[1])
    return server


def main(argv: list[str]) -> None:
    """Serve until Ctrl-C."""
    port = int(argv[1]) if len(argv) > 1 else DEFAULT_PORT
    server = make_server(port)
    print(f"Tracker: http://127.0.0.1:{server.server_address[1]}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main(sys.argv)
