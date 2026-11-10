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
    sent, and an HTTP error from the backend propagates. A run never waits on
    it and never depends on it.

What changes at scale
    Real per-token prices vary by tier, region and contract. A larger system
    keeps a price table per provider with start dates, so a cost report can
    be recomputed for any past month. Langfuse definitions support
    `startDate` for that; this table does not need it yet.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from harness.trace.tracing import TraceConfig

MODELS_PATH = "/api/public/models"
PAGE_SIZE = 100
TIMEOUT_S = 10.0
_FIELDS = {"name", "input_per_token", "output_per_token", "reference"}


class PriceTableError(ValueError):
    """The price table is malformed. The message names the entry and the problem."""


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
        if set(entry) != _FIELDS:
            missing, unknown = sorted(_FIELDS - set(entry)), sorted(set(entry) - _FIELDS)
            raise PriceTableError(f"{where}: missing {missing}, unknown {unknown}")
        name, reference = entry["name"], entry["reference"]
        if not isinstance(name, str) or not name.strip():
            raise PriceTableError(f"{where}: name must be a non-empty string")
        if not isinstance(reference, str) or not reference.strip():
            raise PriceTableError(f"{where} ({name}): reference must name the source of the price")
        if name in seen:
            raise PriceTableError(f"{where}: {name} is listed twice")
        seen.add(name)
        prices.append(ModelPrice(name, _price(entry, "input_per_token", where),
                                 _price(entry, "output_per_token", where), reference))
    return prices


@dataclass(frozen=True)
class SyncPlan:
    """What a sync does. `delete` holds backend ids of definitions that `create` replaces."""

    create: list[ModelPrice] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)
    unchanged: list[ModelPrice] = field(default_factory=list)


def _same(price: ModelPrice, definition: dict[str, Any]) -> bool:
    """True when a backend definition already says what the table says."""
    def close(a: object, b: float) -> bool:
        return isinstance(a, int | float) and math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)

    return (definition.get("matchPattern") == price.match_pattern
            and definition.get("unit") == "TOKENS"
            and close(definition.get("inputPrice"), price.input_per_token)
            and close(definition.get("outputPrice"), price.output_per_token))


def plan_sync(prices: list[ModelPrice], definitions: list[dict[str, Any]]) -> SyncPlan:
    """Compare the table with the backend's definitions. Pure: sends nothing.

    Only the project's own definitions count (`isLangfuseManaged` false): the
    backend's built-in prices for hosted models are never touched. Only
    names in the table are managed, so a definition someone added by hand
    for another model stays. A model with exactly one matching definition is
    unchanged; anything else (none, different prices, duplicates) is
    replaced, so a re-run converges to one definition per model.
    """
    own: dict[str, list[dict[str, Any]]] = {}
    for definition in definitions:
        if not definition.get("isLangfuseManaged"):
            own.setdefault(definition.get("modelName", ""), []).append(definition)
    plan = SyncPlan()
    for price in prices:
        current = own.get(price.name, [])
        if len(current) == 1 and _same(price, current[0]):
            plan.unchanged.append(price)
        else:
            plan.delete.extend(d["id"] for d in current)
            plan.create.append(price)
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
        """Hold the URL and headers. Sends nothing until called."""
        self._url = config.host.rstrip("/") + MODELS_PATH
        self._headers = {**config.headers, "Content-Type": "application/json"}
        self._timeout_s = timeout_s

    def _call(self, method: str, url: str, body: dict[str, Any] | None = None) -> Any:
        """One request; the JSON reply, or None for an empty one. HTTP errors propagate."""
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, headers=self._headers, method=method)
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:  # noqa: S310 - our configured host
            raw = response.read()
        return json.loads(raw) if raw else None

    def list(self) -> list[dict[str, Any]]:
        """Every definition, page by page (the backend holds about 190 built-in ones)."""
        definitions: list[dict[str, Any]] = []
        page = 1
        while True:
            reply = self._call("GET", f"{self._url}?limit={PAGE_SIZE}&page={page}")
            definitions.extend(reply["data"])
            if page >= reply["meta"]["totalPages"]:
                return definitions
            page += 1

    def create(self, price: ModelPrice) -> None:
        """Add a token-priced definition that matches the model name exactly."""
        self._call("POST", self._url, {
            "modelName": price.name,
            "matchPattern": price.match_pattern,
            "unit": "TOKENS",
            "inputPrice": price.input_per_token,
            "outputPrice": price.output_per_token,
        })

    def delete(self, definition_id: str) -> None:
        """Remove one of the project's own definitions."""
        from urllib.parse import quote

        self._call("DELETE", f"{self._url}/{quote(definition_id, safe='')}")


def sync(prices: list[ModelPrice], api: ModelsApi) -> SyncPlan:
    """Make the backend's definitions match the table, and return what was done.

    Deletes before creates: a model briefly has no price, never two. A trace
    that arrives in that gap stays unpriced; for an operator command run
    between runs that is the cheaper failure than a double definition.
    """
    plan = plan_sync(prices, api.list())
    for definition_id in plan.delete:
        api.delete(definition_id)
    for price in plan.create:
        api.create(price)
    return plan
