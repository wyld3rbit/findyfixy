"""Fakes for the harness tests: a judge, an embedder and a system under test.

They use harness types only. The system under test has one label, `kind`, so
the tests prove the harness works for an app that is not FindyFixy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import threading
from typing import Any

from pydantic import BaseModel

from harness.eval.case import Case
from harness.eval.judge import JudgeError
from harness.eval.metrics import ActionRating, Claims, Questions, Verdict
from harness.eval.runner import run_eval
from harness.eval.sut import Answer

NOW = datetime(2026, 10, 26, 12, 30, 5, tzinfo=timezone.utc)
PROV = {"judge_digest": "d1", "prompt_versions": {"m": "h"}, "golden_sha256": "g1", "git_sha": "abc1234"}


def case(cid: str, group: str = "a", kind: str | None = "x", **meta: Any) -> Case:
    """A case expecting label `kind` (none when `kind` is None)."""
    return Case(
        id=cid,
        group=group,
        question=f"question {cid}",
        reference="Root cause: a. Remediation: b.",
        expected={} if kind is None else {"kind": kind},
        meta=meta,
    )


class FakeJudge:
    """Answers every question. Records prompts. Fails when told to."""

    def __init__(self, supported: bool = True, fail: bool = False) -> None:
        """`supported` sets every verdict; `fail` makes every call raise `JudgeError`."""
        self.supported, self.fail, self.prompts = supported, fail, []

    def ask(self, prompt: str, schema: type[BaseModel]) -> Any:
        """A canned reply of the right schema."""
        self.prompts.append(prompt)
        if self.fail:
            raise JudgeError("fake failure")
        if schema is Claims:
            return Claims(claims=["it is auth"])
        if schema is Questions:
            return Questions(questions=["q1", "q2", "q3"])
        if schema is ActionRating:
            return ActionRating(rating="helpful" if self.supported else "unhelpful", reason="fake")
        return Verdict(supported=self.supported)


def embed(texts: Any) -> list[list[float]]:
    """Every text maps to the same unit vector: cosine 1.0."""
    return [[1.0, 0.0] for _ in texts]


class FakeSUT:
    """A system that always answers `kind=label`, with knobs for every failure path."""

    def __init__(
        self,
        label: str = "x",
        *,
        fail_on: Sequence[str] = (),
        raise_on: Sequence[str] = (),
        action: str | None = "roll back",
        meta: Mapping[str, Any] | None = None,
        barrier: threading.Barrier | None = None,
    ) -> None:
        """`fail_on` ids return `Answer.failed`; `raise_on` ids raise; `barrier` waits."""
        self.label, self.action, self.barrier = label, action, barrier
        self.fail_on, self.raise_on = set(fail_on), set(raise_on)
        self.meta = dict(meta) if meta is not None else {"model_used": "small"}

    def describe(self) -> dict[str, Any]:
        """The fake's settings."""
        return {"fake_model": "m1", "fake_threshold": 0.5}

    def answer(self, case: Case) -> Answer:
        """A canned answer whose context names the case, so a leak across cases shows."""
        if self.barrier is not None:
            self.barrier.wait()
        if case.id in self.raise_on:
            raise RuntimeError("model never produced a valid record")
        if case.id in self.fail_on:
            return Answer.failed("TriageError: no valid record", {**self.meta, "model_used": None})
        return Answer(
            text=f"Kind {self.label}. Next action: {self.action}",
            contexts=[f"context for {case.id}"],
            action=self.action,
            labels={"kind": self.label},
            meta=self.meta,
        )

    def summarise(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """One extra aggregate: the share of records that ran."""
        ran = [r for r in records if r["run_error"] is None]
        return {"ran_rate": len(ran) / len(records) if records else None}


def run(cases: Sequence[Case], sut: Any = None, judge: Any = None, **kw: Any) -> dict[str, Any]:
    """`run_eval` with the fakes and a fixed provenance."""
    return run_eval(
        cases,
        sut=sut or FakeSUT(),
        judge=judge or FakeJudge(),
        embed=kw.pop("embed", embed),
        provenance=kw.pop("provenance", PROV),
        **kw,
    )
