"""The seam between the harness and the system it measures.

An application implements `SystemUnderTest` once. The harness then runs it on
every case, scores the `Answer`, and never imports the application. This is the
dependency inversion that keeps the harness reusable: the policy (what a good
answer is, when a release is blocked) depends on an interface, and the
application depends on the same interface, never the other way round.

Contract, read before implementing it
    - `answer` is called from several threads when the run has `workers` > 1.
      Keep per-call state local, and put shared clients behind a lock.
    - An expected failure (the model never produced a valid record) is returned
      as `Answer.failed(...)`, with whatever meta is known, so the record still
      shows what the system saw. Any exception that escapes `answer` is also
      caught by the harness and recorded as the case's run error, but then
      the record has no answer-side meta.
    - `contexts` must be exactly what the model saw, rendered as the prompt
      rendered it. The judge never sees more than the model saw. Otherwise
      faithfulness would reward facts the model could not have used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from harness.eval.case import Case


@dataclass(frozen=True)
class Answer:
    """What the system under test produced for one case.

    `text` is the rendered answer the judge scores. `contexts` is what the
    model saw, one string per item. `action` is the proposed next step for
    `action_quality`; None scores as a failed metric, never as 0. `labels` are
    compared exactly with the case's `expected`. `meta` is copied flat into the
    record. `error` set means the run failed for this case: nothing is scored,
    and the harness records it as `run_error`.
    """

    text: str
    contexts: Sequence[str]
    action: str | None
    labels: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        """Freeze the collections, so a record cannot change after it is built."""
        object.__setattr__(self, "contexts", tuple(self.contexts))
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @classmethod
    def failed(cls, error: str, meta: Mapping[str, Any] | None = None) -> Answer:
        """An expected failure with the meta known so far. One constructor for the shape."""
        return cls(text="", contexts=(), action=None, meta=meta or {}, error=error)


class SystemUnderTest(Protocol):
    """What the harness needs from an application. Three methods, no base class."""

    def describe(self) -> Mapping[str, Any]:
        """The app settings that shaped the answers, for provenance (models, thresholds).

        Merged flat into the report's provenance. A key that collides with a
        harness key is refused, because it would overwrite the judge or git stamp.
        """
        ...

    def answer(self, case: Case) -> Answer:
        """Answer one case. Thread-safe. Expected failures come back as `Answer.failed`."""
        ...

    def summarise(self, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Extra aggregates over a list of case records, merged into each aggregate.

        Called once for "overall" and once per group. Records with a run error
        are included; filter them if the number should not count them.
        """
        ...
