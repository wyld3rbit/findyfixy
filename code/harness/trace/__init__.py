"""Tracing: the harness component that turns any app's run into an OpenTelemetry trace.

    tracing.py     `TraceConfig`, and `Tracing`: one OpenTelemetry tracer provider
                   with a batch exporter over OTLP/HTTP. Built at the edge.
    spans.py       `SpanTree`: the spans of one run, opened and closed by an
                   app's adapter as its events arrive. Attribute names follow the
                   OpenTelemetry GenAI conventions; the few the backend needs on
                   top are named here, in one place.
    command.py     `CommandTrace`: one command as one trace, with a span per step.
    scores.py      `ScoreSink`, and `LangfuseScores`: eval scores sent onto their
                   traces in the background.

Why this is the harness's and not the app's
    Every app needs the same three things: a provider that never blocks a run,
    a tree of spans whose parent is the step in progress, and model-call
    attributes in a standard vocabulary. The app knows only its own events and
    maps them onto `SpanTree` calls (FindyFixy: app/trace_adapter.py). The
    harness never imports an app (tests/test_architecture.py).

Nothing is imported here, so importing the package costs nothing.
"""
