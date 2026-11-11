"""Model prices: a committed price table, and a sync that registers it with Langfuse.

What it does
    `parse_prices` reads a TOML price table (one `[[model]]` per model: name,
    input and output price per token, and the hosted model the price is
    taken from). `plan_sync` compares it with the backend's own model
    definitions and says what to create, replace or leave. `LangfuseModels`
    is the backend's models API; `sync` applies a plan through it.

Why the backend computes the cost
    Langfuse already prices each generation it ingests: a model definition is
    a match pattern on the model name plus a price per input and output
    token. Registering prices there, instead of multiplying tokens ourselves,
    puts cost on every trace, on the dashboard and in its filters, with no
    code in the hot path. The trade-off: cost is computed at ingestion, so a
    price change applies to new traces only, and stored traces keep the cost
    they arrived with. The table in git stays the source of truth; the
    backend holds a copy.

Failure model
    This is an operator command, not part of a run. It fails loudly: a bad
    table raises `PriceTableError` naming the problem before anything is
    sent; a definition already in the backend whose pattern would also price
    a table model raises `PriceConflict` before anything changes; an HTTP
    error from the backend propagates. Each model's new definition is created
    before its old one is deleted, so a failure part-way leaves every model
    with at least one price. A run never waits on this and never depends on it.

What changes at scale
    Real per-token prices vary by tier, region and contract. A larger system
    keeps a price table per provider with start dates, so a cost report can
    be recomputed for any past month. Langfuse definitions support
    `startDate` for that; this table does not need it yet.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any, Protocol

from wyld3rness.trace.tracing import TraceConfig

MODELS_PATH = "/api/public/models"
PAGE_SIZE = 100
TIMEOUT_S = 10.0
_FIELDS = {"name", "input_per_token", "output_per_token", "reference"}


class PriceTableError(ValueError):
    """The price table is malformed. The message names the entry and the problem."""


class PriceConflict(RuntimeError):
    """A backend definition outside the table would also price a table model. Nothing was changed."""


@dataclass(frozen=True)
class ModelPrice:
    """One model's reference price per token, in US dollars, and where it comes from."""

    name: str
    input_per_token: float
    output_per_token: float
    reference: str

    @property
    def match_pattern(self) -> str:
        """The exact, case-insensitive pattern the backend matches the model name against.

        Escaped and anchored, so `qwen3:4b` can never also price `qwen3:4b-instruct`.
        """
        return f"(?i)^({re.escape(self.name)})$"


def _price(entry: dict[str, Any], key: str, where: str) -> float:
    """One price, checked: a finite number, zero or more. bool is rejected (it is an int in Python)."""
    value = entry[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise PriceTableError(f"{where}: {key} must be a number >= 0, got {value!r}")
    return float(value)


def parse_prices(text: str) -> list[ModelPrice]:
    """The price table in `text` (TOML), validated. Raises `PriceTableError` on any problem.

    Every entry needs exactly the four fields; a missing or unknown field is
    an error, not a default, so a typo cannot silently price a model at zero.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PriceTableError(f"not valid TOML: {exc}") from None
    entries = data.get("model")
    if not isinstance(entries, list) or not entries:
        raise PriceTableError("no [[model]] entries")
    prices: list[ModelPrice] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"model {index + 1}"
        if not isinstance(entry, dict):
            raise PriceTableError(f"{where}: must be a [[model]] table, got {entry!r}")
        if set(entry) != _FIELDS:
            missing, unknown = sorted(_FIELDS - set(entry)), sorted(set(entry) - _FIELDS)
            raise PriceTableError(f"{where}: missing {missing}, unknown {unknown}")
        name, reference = entry["name"], entry["reference"]
        if not isinstance(name, str) or not name or name != name.strip():
            raise PriceTableError(f"{where}: name must be a non-empty string with no surrounding spaces, "
                                  f"got {name!r}")
        if not isinstance(reference, str) or not reference.strip():
            raise PriceTableError(f"{where} ({name}): reference must name the source of the price")
        # The backend matches case-insensitively, so "Qwen3:8b" and "qwen3:8b" are one model.
        if name.casefold() in seen:
            raise PriceTableError(f"{where}: {name} is listed twice")
        seen.add(name.casefold())
        prices.append(ModelPrice(name, _price(entry, "input_per_token", where),
                                 _price(entry, "output_per_token", where), reference))
    return prices


@dataclass(frozen=True)
class SyncPlan:
    """What a sync does. `replace` maps each model to create to the backend ids of the definitions it replaces."""

    create: list[ModelPrice] = field(default_factory=list)
    replace: dict[str, list[str]] = field(default_factory=dict)
    unchanged: list[ModelPrice] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def delete(self) -> list[str]:
        """Every backend id this plan removes."""
        return [definition_id for ids in self.replace.values() for definition_id in ids]


def _same(price: ModelPrice, definition: dict[str, Any]) -> bool:
    """True when a backend definition already says what the table says."""
    def close(a: object, b: float) -> bool:
        return isinstance(a, int | float) and math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)

    return (definition.get("matchPattern") == price.match_pattern
            and definition.get("unit") == "TOKENS"
            and close(definition.get("inputPrice"), price.input_per_token)
            and close(definition.get("outputPrice"), price.output_per_token))


def _matches(pattern: object, name: str) -> bool:
    """Whether a backend match pattern matches `name`, read with Python's regex engine.

    The backend evaluates patterns in its database; Python's engine agrees on
    the anchored, escaped patterns this module writes and on common hand-written
    ones. A pattern Python cannot read is treated as a match, so it is reported
    as a conflict for a person to look at rather than ignored.
    """
    if not isinstance(pattern, str):
        return False
    flags = re.IGNORECASE if pattern.startswith("(?i)") else 0
    try:
        return re.search(pattern.removeprefix("(?i)"), name, flags) is not None
    except re.error:
        return True


def plan_sync(prices: list[ModelPrice], definitions: list[dict[str, Any]]) -> SyncPlan:
    """Compare the table with the backend's definitions. Pure: sends nothing.

    Only the project's own definitions count (`isLangfuseManaged` false): the
    backend's built-in prices for hosted models are never touched. A model
    with exactly one definition of its name that says what the table says is
    unchanged; anything else (none, different prices, duplicates) is
    replaced, so a re-run converges to one definition per model. A project
    definition under another name whose pattern also matches a table model
    is a conflict: the model would be priced twice, and which price the
    backend uses is not defined. Conflicts are listed for `sync` to refuse.
    """
    own: dict[str, list[dict[str, Any]]] = {}
    for definition in definitions:
        if not definition.get("isLangfuseManaged"):
            own.setdefault(str(definition.get("modelName", "")).casefold(), []).append(definition)
    plan = SyncPlan()
    for price in prices:
        current = own.get(price.name.casefold(), [])
        if len(current) == 1 and _same(price, current[0]):
            plan.unchanged.append(price)
        else:
            plan.replace[price.name] = [d["id"] for d in current]
            plan.create.append(price)
    table = {price.name.casefold() for price in prices}
    for name, others in own.items():
        if name in table:
            continue
        for definition in others:
            hit = [p.name for p in prices if _matches(definition.get("matchPattern"), p.name)]
            if hit:
                plan.conflicts.append(f"{definition.get('modelName')!r} (pattern {definition.get('matchPattern')!r}) "
                                      f"also matches {', '.join(hit)}")
    return plan


class ModelsApi(Protocol):
    """The backend's model definitions. `LangfuseModels` is the real one; tests pass a fake."""

    def list(self) -> list[dict[str, Any]]:
        """Every model definition the backend holds."""
        ...

    def create(self, price: ModelPrice) -> None:
        """Add a definition for `price`."""
        ...

    def delete(self, definition_id: str) -> None:
        """Remove one definition."""
        ...


class LangfuseModels:
    """Langfuse's `/api/public/models`, authenticated with the project's key pair."""

    def __init__(self, config: TraceConfig, *, timeout_s: float = TIMEOUT_S) -> None:
        """Hold the config. Sends nothing until called."""
        self._config = config
        self._timeout_s = timeout_s

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        """One request through `TraceConfig.request`. HTTP errors propagate."""
        return self._config.request(method, path, body, timeout_s=self._timeout_s)

    def list(self) -> list[dict[str, Any]]:
        """Every definition, page by page (the backend holds about 190 built-in ones)."""
        definitions: list[dict[str, Any]] = []
        page = 1
        while True:
            reply = self._call("GET", f"{MODELS_PATH}?limit={PAGE_SIZE}&page={page}")
            definitions.extend(reply["data"])
            if page >= reply["meta"]["totalPages"]:
                return definitions
            page += 1

    def create(self, price: ModelPrice) -> None:
        """Add a token-priced definition that matches the model name exactly."""
        self._call("POST", MODELS_PATH, {
            "modelName": price.name,
            "matchPattern": price.match_pattern,
            "unit": "TOKENS",
            "inputPrice": price.input_per_token,
            "outputPrice": price.output_per_token,
        })

    def delete(self, definition_id: str) -> None:
        """Remove one of the project's own definitions."""
        from urllib.parse import quote

        self._call("DELETE", f"{MODELS_PATH}/{quote(definition_id, safe='')}")


def sync(prices: list[ModelPrice], api: ModelsApi) -> SyncPlan:
    """Make the backend's definitions match the table, and return what was done.

    Refuses with `PriceConflict`, changing nothing, when another definition
    would also price a table model. Otherwise, model by model, it creates
    the new definition, then deletes the old ones. A failure part-way leaves
    every model priced (old or new, never none); in the moment between
    create and delete a model has two definitions, which an operator command
    run between eval runs can afford, and a re-run converges.
    """
    plan = plan_sync(prices, api.list())
    if plan.conflicts:
        raise PriceConflict("; ".join(plan.conflicts))
    for price in plan.create:
        api.create(price)
        for definition_id in plan.replace[price.name]:
            api.delete(definition_id)
    return plan
