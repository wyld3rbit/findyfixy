"""FindyFixy as a system under test for the evaluation harness.

What it does
    Implements `harness.eval.sut.SystemUnderTest` for the triage graph:
    converts each golden row into a harness `Case`, runs the graph with that
    case's own incident held out of recall, and returns the answer, what the
    model saw, the severity label and the twin data the report keeps.

Why this is the only app module the harness touches
    The harness owns the policy (metrics, gate, baseline) and knows nothing
    about incidents, severities or twins. Everything FindyFixy-specific lives
    here: the holdout, the `severity` label, the `twin`/`no_twin` groups, and
    `twin_recall_rate`. Keeping it in one module means a second app needs one
    adapter of its own and no change to the harness.

Why holdout
    The golden question is built from one incident. If recall returns that
    incident, the model reads the answer. The score then measures lookup of
    the exact record, not triage. Recall fetches k + 1 candidates, drops the
    source id, keeps k. The twin (same incident, other voice) stays eligible:
    that is what the twin group measures.

What the judge sees
    `contexts` are the prior incidents the model saw, one string per
    incident, formatted and capped by `format_prior`, the same function the
    prompt uses. The judge never sees more than the model saw.

Thread safety
    `answer` keeps its per-case state (the recall sink) local, so several
    workers can call it at once. The injected `candidates` must be thread-safe;
    the live wiring in app/eval_answers.py puts it behind a lock.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from app.golden import GoldenCase
from app.graph import THRESHOLD, Recall
from app.triage import RECALL_K, Prior, Triage, format_prior
from harness.eval.case import Case
from harness.eval.sut import Answer

if TYPE_CHECKING:
    from app.trace_adapter import EventSpans, RunTracer

# (question, n) -> up to n prior incidents, best first. The live one wraps
# `Retriever.hybrid_rerank` and `Retriever.brief`.
Candidates = Callable[[str, int], Sequence[Prior]]

# The one label FindyFixy is scored on. Its record keys are `severity`,
# `severity_match` and `expected_severity`, as in every saved report.
SEVERITY = "severity"


class Invokable(Protocol):
    """What the adapter needs from a compiled graph: `invoke(state) -> state`."""

    def invoke(self, state: dict[str, Any]) -> Mapping[str, Any]:
        """Run the graph on `{"question": ...}` and return the final state."""
        ...


# ---- Golden rows into harness cases -----------------------------------------


def to_case(golden: GoldenCase) -> Case:
    """One golden row as a harness case. The ids the adapter needs travel in `meta`.

    `source_id` and `twin_id` are copied into the record, so a saved report
    shows which incident was held out and which twin could be recalled.
    """
    return Case(
        id=golden.id,
        group=golden.group,
        question=golden.question,
        reference=golden.reference,
        expected={SEVERITY: golden.expected_severity},
        meta={"source_id": golden.source_id, "twin_id": golden.twin_id},
    )


# ---- Holdout recall ---------------------------------------------------------


def holdout(candidates: Sequence[Prior], source_id: str, k: int) -> list[Prior]:
    """Drop the case's own incident, keep the first `k` of the rest. Pure.

    The caller must fetch `k + 1` candidates, so that after the drop `k`
    remain. The twin is not special: it stays if the retriever returned it.
    """
    return [p for p in candidates if p.incident_id != source_id][:k]


def make_holdout_recall(
    candidates: Candidates, source_id: str, k: int, sink: list[Prior]
) -> Recall:
    """Build the recall step for one case, bound to that case's source id.

    Every call asks for `k + 1` candidates and filters. The kept priors are
    also written to `sink`, so the adapter knows what the model saw. Clearing
    the sink first means a graph that calls recall twice reports the last call,
    which is the one whose prompt the model answered.
    """

    def recall(question: str) -> list[Prior]:
        """Recall minus the source incident."""
        kept = holdout(candidates(question, k + 1), source_id, k)
        sink.clear()
        sink.extend(kept)
        return kept

    return recall


# ---- What the judge sees ----------------------------------------------------


def render_answer(triage: Triage) -> str:
    """Plain-text answer for the judge. Confidence is left out: it is not a claim."""
    return (
        f"Severity {triage.severity}. Component {triage.component}. "
        f"Next action: {triage.next_action}"
    )


def render_contexts(priors: Sequence[Prior]) -> list[str]:
    """One string per recalled incident, formatted and capped as in the prompt."""
    return [format_prior([p]) for p in priors]


# ---- The adapter ------------------------------------------------------------


def twin_recall_rate(records: Sequence[Mapping[str, Any]]) -> float | None:
    """Share of twin cases (no run error) whose twin came back from recall. Pure.

    None when there is no such case, so a group without twins shows `n/a`, not 0.
    """
    twin_cases = [r for r in records if r["run_error"] is None and r.get("twin_id") is not None]
    hits = sum(1 for r in twin_cases if r.get("twin_recalled"))
    return (hits / len(twin_cases)) if twin_cases else None


class FindyFixy:
    """The triage graph behind the `SystemUnderTest` seam.

    `make_graph` compiles a graph with recall bound to one case. Compiling is
    cheap; the model calls are not. The negative control is the caller's
    choice of `make_graph` (one that ignores the recall it is given), so the
    adapter runs the same way in both cases.
    """

    def __init__(
        self,
        *,
        make_graph: Callable[[Recall], Invokable],
        candidates: Candidates,
        small_model: str,
        large_model: str,
        k: int = RECALL_K,
        threshold: float = THRESHOLD,
        run_tracer: RunTracer | None = None,
    ) -> None:
        """Hold the wiring. Nothing runs until `answer`.

        `run_tracer`, when tracing is on, opens a FindyFixy run trace around
        each case's graph run; it nests under the harness's case span. The
        graph's events must reach it through the emitter `make_graph` binds
        (`RunTracer.wrap`).
        """
        self._run_tracer = run_tracer
        self._make_graph = make_graph
        self._candidates = candidates
        self._small_model = small_model
        self._large_model = large_model
        self._k = k
        self._threshold = threshold

    def describe(self) -> dict[str, Any]:
        """The settings that shape an answer, for the report's provenance."""
        return {
            "small_model": self._small_model,
            "large_model": self._large_model,
            "threshold": self._threshold,
            "recall_k": self._k,
        }

    def answer(self, case: Case) -> Answer:
        """Run the graph on one case with its source incident held out of recall.

        Any graph error (for example `TriageError`) comes back as a failed
        answer that still says what was recalled. The catch is broad on
        purpose: whatever the graph raised, the record must show the context
        the model had when it failed.
        """
        if self._run_tracer is None:
            return self._answer(case, None)
        with self._run_tracer.run(case.question, {"findyfixy.case": case.id}) as spans:
            return self._answer(case, spans)

    def _answer(self, case: Case, spans: EventSpans | None) -> Answer:
        """`answer`, ending the run trace (when there is one) with the triage or the error."""
        seen: list[Prior] = []
        twin_id = case.meta["twin_id"]
        recall = make_holdout_recall(self._candidates, case.meta["source_id"], self._k, seen)
        try:
            state = self._make_graph(recall).invoke({"question": case.question})
            triage = state.get("triage")
            if triage is None:
                raise ValueError("graph returned no triage")
        except Exception as err:
            meta = {"recalled_ids": [p.incident_id for p in seen], "twin_recalled": False, "model_used": None}
            if spans is not None:
                spans.finish_run(error=f"{type(err).__name__}: {err}")
            return Answer.failed(f"{type(err).__name__}: {err}", meta)
        if spans is not None:
            spans.finish_run(output=triage.model_dump(), model_used=state.get("model_used"))
        recalled = [p.incident_id for p in seen]
        return Answer(
            text=render_answer(triage),
            contexts=render_contexts(seen),
            action=triage.next_action,
            labels={SEVERITY: triage.severity},
            meta={
                "recalled_ids": recalled,
                "twin_recalled": twin_id is not None and twin_id in recalled,
                "model_used": state.get("model_used"),
            },
        )

    def summarise(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """FindyFixy's extra aggregate: how often the twin was recalled."""
        return {"twin_recall_rate": twin_recall_rate(records)}
