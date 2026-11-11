"""One OpenTelemetry tracer provider that exports spans in the background over OTLP/HTTP.

What it does
    `Tracing.start(config)` builds a `TracerProvider` with a `BatchSpanProcessor`
    and an `OTLPSpanExporter`, authenticated for a Langfuse project. `tracer()`
    hands out the tracer that `SpanTree` (spans.py) uses. `close()` flushes what
    is queued and stops the export thread.

Why a batch exporter, and what it costs
    A span that ends goes into an in-memory queue and returns at once. A
    background thread sends the queue every `SCHEDULE_DELAY_MS`, or sooner when
    `MAX_BATCH` spans wait. So a slow or stopped backend never slows a run: the
    run only appends. The price is loss, by design. When the queue is full
    (`MAX_QUEUE` spans), new spans are dropped, and spans still queued when the
    process is killed are lost. `close()` flushes them on a normal exit. For a
    debugging tool that is the right trade; an audit log needs a durable queue.

    Each export call has a timeout (`EXPORT_TIMEOUT_S`), so a backend that hangs
    holds up only the export thread, and only for that long.

Why OTLP and not a vendor SDK
    The spans carry standard OpenTelemetry attributes and travel over the
    standard protocol. Langfuse accepts OTLP at `/api/public/otel`; another
    backend (Jaeger, Tempo, Phoenix) needs a different endpoint and headers,
    not different code. The vendor-specific parts are the endpoint path, the
    Basic authentication, and the few attribute names in spans.py.

Configuration
    Read at the edge (findyfixy/settings.py builds `TraceConfig`). No config means no
    `Tracing`: callers hold `None`, and nothing here is ever built or imported.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the SDK is imported only when tracing is turned on
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter
    from opentelemetry.trace import Tracer

# Send a batch every second, so a trace appears in the UI about a second after
# its run ends. The SDK default is 5 s. Shorter means more, smaller requests.
SCHEDULE_DELAY_MS = 1000
MAX_BATCH = 512
# About 2,000 spans queued at most. A FindyFixy run makes about 15 spans, so
# this is over a hundred runs of backlog before anything is dropped.
MAX_QUEUE = 2048
EXPORT_TIMEOUT_S = 5.0
# Langfuse's OTLP path. OTLP/HTTP appends nothing: this is the full traces URL.
LANGFUSE_OTLP_PATH = "/api/public/otel/v1/traces"


@dataclass(frozen=True)
class TraceConfig:
    """Where traces go. `host` is the backend's base URL, such as http://127.0.0.1:3000.

    `public_key` and `secret_key` are the project's API key pair. The secret is
    a credential: it travels only in the Authorization header and is never
    printed. `repr` hides it, so a config in a traceback or a log line leaks
    nothing. `project_id` is the backend project the keys belong to; it is
    only needed to link to a trace in the backend's web page (`trace_url`).
    """

    host: str
    public_key: str
    secret_key: str
    service_name: str
    user_id: str = "local"
    project_id: str | None = None

    def __repr__(self) -> str:
        """Everything but the secret."""
        return (f"TraceConfig(host={self.host!r}, public_key={self.public_key!r}, "
                f"secret_key='***', service_name={self.service_name!r}, user_id={self.user_id!r}, "
                f"project_id={self.project_id!r})")

    def trace_url(self, trace_id: str) -> str | None:
        """Where the backend's web page shows one trace, or None when the project is not known."""
        if not self.project_id:
            return None
        from urllib.parse import quote

        return f"{self.host.rstrip('/')}/project/{quote(self.project_id, safe='')}/traces/{quote(trace_id, safe='')}"

    def request(self, method: str, path: str, body: object = None, *, timeout_s: float) -> Any:
        """One authenticated call to the backend's API at `path`; the JSON reply, or None if empty.

        The one place the harness talks HTTP to the backend outside OTLP
        (scores, model prices), so auth headers and the call shape change in
        one place. HTTP and network errors propagate: each caller decides
        whether to drop (scores, during a run) or fail loudly (an operator
        command).
        """
        import json
        import urllib.request

        data = json.dumps(body).encode() if body is not None else None
        headers = {**self.headers, "Content-Type": "application/json"}
        request = urllib.request.Request(self.host.rstrip("/") + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - our configured host
            raw = response.read()
        return json.loads(raw) if raw else None

    @property
    def endpoint(self) -> str:
        """The OTLP traces URL."""
        return self.host.rstrip("/") + LANGFUSE_OTLP_PATH

    @property
    def headers(self) -> dict[str, str]:
        """HTTP Basic authentication with the key pair, as Langfuse's OTLP endpoint expects."""
        token = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
        return {"Authorization": f"Basic {token}"}


class Tracing:
    """A running tracer provider. Build with `start`; close once, at exit."""

    def __init__(self, provider: TracerProvider, config: TraceConfig) -> None:
        """Hold the provider. Use `start` instead of calling this directly."""
        self._provider = provider
        self.config = config  # what scores (scores.py) are sent with: the same backend and keys

    @classmethod
    def start(cls, config: TraceConfig, exporter: SpanExporter | None = None,
              export_timeout_s: float = EXPORT_TIMEOUT_S) -> Tracing:
        """Build the provider and its export thread. `exporter` replaces OTLP in tests.

        `export_timeout_s` bounds each export and so the wait in `close`. A
        short-lived process that must exit fast (an MCP server started per
        tool call) passes a short one.

        The provider is local to this object, not installed as the global one,
        so two of them (a test and the code under test) cannot interfere.
        """
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": config.service_name}),
            shutdown_on_exit=False,  # `close` flushes; an atexit hook would hide a missing call
        )
        sender = exporter or OTLPSpanExporter(
            endpoint=config.endpoint, headers=config.headers, timeout=export_timeout_s
        )
        provider.add_span_processor(_trace_user(config.user_id))
        provider.add_span_processor(BatchSpanProcessor(
            sender,
            max_queue_size=MAX_QUEUE,
            schedule_delay_millis=SCHEDULE_DELAY_MS,
            max_export_batch_size=MAX_BATCH,
            export_timeout_millis=export_timeout_s * 1000,
        ))
        return cls(provider, config)

    def tracer(self) -> Tracer:
        """The tracer spans are made with. The name says which component made them."""
        return self._provider.get_tracer("wyld3rness.trace")

    def flush(self, timeout_s: float = EXPORT_TIMEOUT_S) -> bool:
        """Send everything queued now. True when it was all sent within the timeout."""
        return self._provider.force_flush(int(timeout_s * 1000))

    def close(self) -> None:
        """Flush and stop the export thread. Safe to call twice.

        With the backend down this waits up to `EXPORT_TIMEOUT_S` while the
        exporter gives up, so a Ctrl-C can take that long. That bound is the
        price of not losing the last runs' spans on a normal exit.
        """
        self._provider.shutdown()


def _trace_user(user_id: str) -> Any:
    """A span processor that puts `user.id` on every trace's top span.

    In the provider, not at each call site, so no command can forget it.
    Langfuse reads `user.id` from the top span and groups traces by it
    (its Users view). The value comes from settings (`TRACE_USER`, default
    "local") until Day 7 knows the authenticated engineer.
    """
    from opentelemetry.sdk.trace import SpanProcessor

    class TraceUser(SpanProcessor):
        """Sets `user.id` on spans that start a trace."""

        def on_start(self, span: Any, parent_context: Any = None) -> None:
            """Only top spans: a child already belongs to a trace that has the user."""
            if span.parent is None:
                span.set_attribute("user.id", user_id)

    return TraceUser()
