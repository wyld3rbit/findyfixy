"""One golden case, in the harness's own terms.

What it is: a question, the reference answer the judge compares against, and the
labels a correct answer must carry (for example `{"severity": "sev2"}`). Every
application converts its own golden rows into `Case` at the seam, so the
harness never sees an app type.

Why labels are a mapping and not fields: the harness scores any label the same
way, as an exact-match rate. One app has a severity, another a route or a
category. A field per label would put app vocabulary into the harness.

`meta` is app data the harness does not read. It is copied flat into the case
record, so a saved report carries it (for example the source incident id).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Case:
    """One golden case: what is asked, what a right answer says, and the labels it must carry.

    `group` names the slice the case is aggregated in (besides "overall").
    Groups are reported in the order they first appear in the case list.

    Frozen, with read-only mappings: under `workers` > 1 the same case objects
    are read from several threads, and a mutation there would be a data race
    that shows up as a wrong record, not a crash.
    """

    id: str
    group: str
    question: str
    reference: str
    expected: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy the mappings into read-only views. The caller's dicts stay theirs."""
        object.__setattr__(self, "expected", MappingProxyType(dict(self.expected)))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
