"""`make obs-prices`: register FindyFixy's reference price table with the trace backend.

Why here and not in the harness
    The harness knows how to parse a price table and talk to Langfuse
    (`wyld3rness/trace/prices.py`). Which table, which backend and which keys
    are this app's settings, read at the edge: here.

The command is itself traced, like every command that touches the network.

It refuses to run when a model the app is configured to use (small, large or
judge) has no price in the table: that model's calls would carry no cost, and
cost per run on the dashboard would quietly under-report.
"""

from __future__ import annotations

from collections.abc import Iterable

from findyfixy.paths import PRICES


def unpriced(configured: Iterable[str], priced: Iterable[str]) -> list[str]:
    """The configured model names with no entry in the table, compared as the backend does (any case)."""
    known = {name.casefold() for name in priced}
    return sorted({name for name in configured if name.casefold() not in known})


def main() -> None:
    """Parse `config/prices.toml`, sync it to Langfuse, print one line per model."""
    from findyfixy.settings import load_env, optional_models, read_tracing
    from findyfixy.observability.tracing import command_trace
    from wyld3rness.trace.prices import LangfuseModels, PriceConflict, parse_prices, sync

    prices = parse_prices(PRICES.read_text(encoding="utf-8"))  # fails before anything is sent
    load_env()
    models = optional_models()
    configured = [models.small, models.large, models.judge] if models else []
    missing = unpriced(configured, (p.name for p in prices))
    if missing:
        raise SystemExit(f"no price in {PRICES.name} for {', '.join(missing)}: add it, or its calls carry no cost")
    config = read_tracing()
    if config is None:
        raise SystemExit("tracing is off (no Langfuse keys, or TRACING=off): nothing to register")
    with command_trace("obs-prices", input=f"{PRICES.name}: {len(prices)} models") as command:
        try:
            plan = sync(prices, LangfuseModels(config))
        except PriceConflict as conflict:
            raise SystemExit(f"nothing changed: {conflict}. Remove or rename that definition in Langfuse "
                             "(Settings, Models), then re-run.") from None
        lines = [f"{'registered' if p in plan.create else 'unchanged '} {p.name}: "
                 f"${p.input_per_token * 1e6:.2f} in, ${p.output_per_token * 1e6:.2f} out per 1M tokens "
                 f"(reference {p.reference})" for p in prices]
        command.output("\n".join(lines))
    print("\n".join(lines))
    if plan.delete:
        print(f"replaced {len(plan.delete)} older definition(s)")


if __name__ == "__main__":
    main()
