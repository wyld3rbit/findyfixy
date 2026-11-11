"""Tests for the price table and its sync to the trace backend (Day 6, step 7).

What these tests cover
    `parse_prices` rejects every malformed entry by name instead of defaulting
    to zero, and names that differ only in case or spaces. `plan_sync` touches
    only the project's own definitions, converges to one definition per
    model, and reports a definition under another name whose pattern would
    also price a table model. `sync` refuses on such a conflict and creates
    each model's new definition before deleting its old ones. The committed
    table parses and prices the four pulled models; `make obs-prices` checks
    the configured ones at run time (`unpriced`).

What a staff+ reader should notice
    The backend is a fake `ModelsApi`: no network. The match pattern is
    tested against a near-miss name, because a loose pattern would silently
    price the wrong model.
"""

from __future__ import annotations

import re

import pytest

from app.paths import PRICES
from app.prices import unpriced
from harness.trace.prices import ModelPrice, PriceConflict, PriceTableError, parse_prices, plan_sync, sync

ONE = """
[[model]]
name = "qwen3:4b"
input_per_token = 0.00000005
output_per_token = 0.0000004
reference = "gpt-5-nano"
"""


def test_a_valid_table_parses() -> None:
    """Every field lands where it belongs."""
    assert parse_prices(ONE) == [ModelPrice("qwen3:4b", 5e-8, 4e-7, "gpt-5-nano")]


@pytest.mark.parametrize("change, message", [
    (("input_per_token = 0.00000005\n", ""), "missing"),
    (('reference = "gpt-5-nano"\n', 'reference = "gpt-5-nano"\nprice = 1\n'), "unknown"),
    (("0.00000005", "-1"), ">= 0"),
    (("0.00000005", "true"), ">= 0"),
    (("0.00000005", '"cheap"'), ">= 0"),
    (("0.00000005", "nan"), ">= 0"),
    (('"gpt-5-nano"', '""'), "reference"),
    (('"qwen3:4b"', '" "'), "name"),
    (('"qwen3:4b"', '"qwen3:4b "'), "spaces"),
])
def test_a_malformed_entry_is_rejected_by_name(change: tuple[str, str], message: str) -> None:
    """No default fills a gap: a typo cannot price a model at zero."""
    with pytest.raises(PriceTableError, match=message):
        parse_prices(ONE.replace(*change))


def test_a_model_listed_twice_and_an_empty_table_are_rejected() -> None:
    """Two prices for one model would leave the backend to pick one; it matches names in any case."""
    with pytest.raises(PriceTableError, match="twice"):
        parse_prices(ONE + ONE)
    with pytest.raises(PriceTableError, match="twice"):
        parse_prices(ONE + ONE.replace('"qwen3:4b"', '"Qwen3:4B"'))
    with pytest.raises(PriceTableError, match="table"):
        parse_prices("model = [1]")
    with pytest.raises(PriceTableError, match="no"):
        parse_prices("")
    with pytest.raises(PriceTableError, match="TOML"):
        parse_prices("[[model]\n")


def test_the_match_pattern_is_exact() -> None:
    """Anchored and escaped: the 4B price never matches the 4B-instruct model or a lookalike."""
    pattern = ModelPrice("qwen2.5:7b", 0, 0, "x").match_pattern
    assert re.fullmatch(pattern[4:], "qwen2.5:7b")  # strip (?i) for Python; Postgres reads it
    assert not re.fullmatch(pattern[4:], "qwen2x5:7b")
    assert not re.fullmatch(pattern[4:], "qwen2.5:7b-instruct")


def own(price: ModelPrice, id_: str, **changes: object) -> dict[str, object]:
    """A backend definition for `price`, as the API returns it."""
    return {"id": id_, "modelName": price.name, "matchPattern": price.match_pattern, "unit": "TOKENS",
            "inputPrice": price.input_per_token, "outputPrice": price.output_per_token,
            "isLangfuseManaged": False, **changes}


PRICE = ModelPrice("qwen3:4b", 5e-8, 4e-7, "gpt-5-nano")


def test_a_matching_definition_is_left_alone() -> None:
    """A re-run changes nothing."""
    plan = plan_sync([PRICE], [own(PRICE, "a")])
    assert (plan.create, plan.delete, plan.unchanged) == ([], [], [PRICE])


@pytest.mark.parametrize("definitions, deleted", [
    ([], []),
    ([own(PRICE, "a", inputPrice=1e-6)], ["a"]),
    ([own(PRICE, "a", matchPattern="qwen3")], ["a"]),
    ([own(PRICE, "a"), own(PRICE, "b")], ["a", "b"]),
])
def test_a_missing_changed_or_duplicated_definition_is_replaced(definitions, deleted) -> None:
    """Every case ends at exactly one definition with the table's prices."""
    plan = plan_sync([PRICE], definitions)
    assert plan.create == [PRICE] and plan.delete == deleted and plan.conflicts == []


def test_built_in_and_unlisted_definitions_are_never_touched() -> None:
    """The backend's hosted-model prices and a hand-added definition for another model stay."""
    managed = own(PRICE, "m", isLangfuseManaged=True, inputPrice=9.0)
    other = own(ModelPrice("other", 1, 1, "x"), "o")
    plan = plan_sync([PRICE], [managed, other])
    assert plan.delete == [] and plan.create == [PRICE] and plan.conflicts == []


def test_another_definition_that_also_matches_a_table_model_is_a_conflict() -> None:
    """A hand-added catch-all would price the model twice: reported, and `sync` changes nothing."""
    catch_all = {"id": "c", "modelName": "qwen-any", "matchPattern": "(?i)^qwen.*", "isLangfuseManaged": False}
    assert plan_sync([PRICE], [catch_all]).conflicts
    api = FakeModels([catch_all])
    with pytest.raises(PriceConflict, match="qwen-any"):
        sync([PRICE], api)
    assert api.calls == []


class FakeModels:
    """A `ModelsApi` in memory that records the order of calls."""

    def __init__(self, definitions: list[dict[str, object]]) -> None:
        self.definitions = definitions
        self.calls: list[str] = []

    def list(self) -> list[dict[str, object]]:
        return list(self.definitions)

    def create(self, price: ModelPrice) -> None:
        self.calls.append(f"create {price.name}")

    def delete(self, definition_id: str) -> None:
        self.calls.append(f"delete {definition_id}")


def test_sync_creates_before_it_deletes_model_by_model() -> None:
    """A failure part-way leaves every model priced: old or new, never none."""
    other = ModelPrice("qwen3:8b", 2.5e-7, 2e-6, "gpt-5-mini")
    api = FakeModels([own(PRICE, "a", outputPrice=1.0), own(other, "b", inputPrice=1.0)])
    sync([PRICE, other], api)
    assert api.calls == ["create qwen3:4b", "delete a", "create qwen3:8b", "delete b"]


def test_configured_models_without_a_price_are_named() -> None:
    """`make obs-prices` refuses these, so no configured model's calls go unpriced."""
    assert unpriced(["qwen3:4b", "Qwen3:4B", "qwen3:14b"], ["qwen3:4b"]) == ["qwen3:14b"]


def test_the_committed_table_parses_and_names_a_source_for_each_price() -> None:
    """The file `make obs-prices` reads is valid, and prices the four models pulled for this project."""
    prices = {p.name: p for p in parse_prices(PRICES.read_text(encoding="utf-8"))}
    assert {"qwen2.5:1.5b", "qwen3:4b-instruct", "qwen2.5:7b", "qwen3:8b"} <= set(prices)
    assert all(p.reference for p in prices.values())
