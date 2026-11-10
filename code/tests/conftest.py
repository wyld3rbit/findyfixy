"""Test-session settings shared by every test.

Tracing is off for the whole session. Some tests start real subprocesses (the
MCP server over stdio) or real entry points, which read the tracing settings
and would find the local Langfuse stack's keys in `infra/langfuse/.env`. The
test run then sent its fake traffic to the developer's live backend: 48 MCP
tool-call traces from one `make test` (Day 6, step 5). The variable is set
here, at import, before any test or subprocess starts, and every subprocess
inherits it. Tests that check tracing build their own provider with an
in-memory exporter, or set the variable themselves with `monkeypatch`.
"""

import os

os.environ["TRACING"] = "off"
