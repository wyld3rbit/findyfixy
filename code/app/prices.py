"""`make obs-prices`: register FindyFixy's reference price table with the trace backend.

Why here and not in the harness
    The harness knows how to parse a price table and talk to Langfuse
    (`harness/trace/prices.py`). Which table, which backend and which keys
    are this app's settings, read at the edge: here.

The command is itself traced, like every command that touches the network.
"""

from __future__ import annotations

from app.paths import PRICES


def main() -> None:
    """Parse `config/prices.toml`, sync it to Langfuse, print one line per model."""
    from app.settings import load_env, read_tracing
    from app.tracing import command_trace
    from harness.trace.prices import LangfuseModels, parse_prices, sync

    prices = parse_prices(PRICES.read_text(encoding="utf-8"))  # fails before anything is sent
    load_env()
    config = read_tracing()
    if config is None:
        raise SystemExit("tracing is off (no Langfuse keys, or TRACING=off): nothing to register")
    with command_trace("obs-prices", input=f"{PRICES.name}: {len(prices)} models") as command:
        plan = sync(prices, LangfuseModels(config))
        lines = [f"{'registered' if p in plan.create else 'unchanged '} {p.name}: "
                 f"${p.input_per_token * 1e6:.2f} in, ${p.output_per_token * 1e6:.2f} out per 1M tokens "
                 f"(reference {p.reference})" for p in prices]
        command.output("\n".join(lines))
    print("\n".join(lines))
    if plan.delete:
        print(f"replaced {len(plan.delete)} older definition(s)")


if __name__ == "__main__":
    main()
