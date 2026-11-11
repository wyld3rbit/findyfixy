"""Tests for FindyFixy's price table and its check of configured models (Day 6, step 7).

The parser and the sync are the harness's, tested in the harness suite. Here:
the committed table parses and prices the four pulled models, and
`make obs-prices` names any configured model without a price (`unpriced`).
"""

from __future__ import annotations

from findyfixy.observability.prices import unpriced
from findyfixy.paths import PRICES
from wyld3rness.trace.prices import parse_prices


def test_configured_models_without_a_price_are_named() -> None:
    """`make obs-prices` refuses these, so no configured model's calls go unpriced."""
    assert unpriced(["qwen3:4b", "Qwen3:4B", "qwen3:14b"], ["qwen3:4b"]) == ["qwen3:14b"]


def test_the_committed_table_parses_and_names_a_source_for_each_price() -> None:
    """The file `make obs-prices` reads is valid, and prices the four models pulled for this project."""
    prices = {p.name: p for p in parse_prices(PRICES.read_text(encoding="utf-8"))}
    assert {"qwen2.5:1.5b", "qwen3:4b-instruct", "qwen2.5:7b", "qwen3:8b"} <= set(prices)
    assert all(p.reference for p in prices.values())
