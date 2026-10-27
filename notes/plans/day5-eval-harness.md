# Plan: Day 5 eval harness

Status: BUILT 2026-10-26. The baseline is not frozen yet.

## What differs from this plan, as built

- Questions are the alert line only (the first timeline entry). The plan said summary plus timeline, which leaks the cause and the fix.
- The answering model runs at temperature 0 with thinking off and a 120 s read timeout, defined once in `app/triage.py`. The plan did not cover this; the first evaluations showed run-to-run swings of 9 to 16 of 30 on severity without it.
- Recalled incidents carry their severity (`Prior` and `Passage` gained a validated `severity` field).
- The runner gained `--workers` (default 1; no speedup on this machine), `--no-recall`, `--show-case`, `--baseline-from`; reports store each case's answer, contexts and the judge's per-claim decisions.
- Work package C ran in the main checkout, not a worktree: worktrees branched from an older commit without the Day 4 code.

The sections below are the original plan, kept as the record of what was approved.

## Goal

Score graph answers on 30 holdout cases. Gate a release on absolute thresholds and on a regression against a committed baseline.

## Research (facts from the code, 2026-10-26)

- Graph state returns `triage` (`Triage`: severity, component, next_action, confidence) and `model_used` (`app/graph.py`).
- Recall seam: `Recall = Callable[[str], Sequence[Prior]]`. The real one is `lazy_recall()` in `app/console.py`. It uses `Retriever.hybrid_rerank(question, k)` and `retriever.brief(id)`.
- Corpus: `data/incidents.jsonl`, 200 incidents. Fields: summary, timeline, root_cause, remediation, severity, `twin_of`. 20 have a twin.
- `ragas` 0.4.3 rejected: PYSEC-2026-3046, no fix, 31 new packages. DeepEval/TruLens/openevals not adopted. No new dependency.
- Ollama `format=` with a Pydantic schema is already used by `ask_validated` (`app/triage.py`).

## Decisions made (learner)

1. Holdout. The case's own incident is removed from recall. Fetch k+1, filter, keep k.
2. 20 twin cases, 10 no-twin cases. Metrics reported per group.
3. A failed judge call is excluded from the mean and counted as `judge_failures`. The gate fails above a failure-rate cap.
4. Own the metrics. No eval library.
5. Regression gate compares against `evals/baseline.json` with bootstrap 95% CIs.

## Interfaces (fixed here so packages can run in parallel)

```python
# app/judge.py
class Judge(Protocol):
    def ask(self, prompt: str, schema: type[T]) -> T: ...   # raises JudgeError after retries
class JudgeError(Exception): ...

# app/metrics.py   (pure; Judge and Embedder injected)
class MetricResult(NamedTuple): score: float | None; failed: bool
def faithfulness(judge, answer, contexts) -> MetricResult
def response_relevancy(judge, embed, question, answer, n=3) -> MetricResult
def context_recall(judge, reference, contexts) -> MetricResult
PROMPT_VERSIONS: dict[str, str]   # metric -> sha256 of its prompt text

# app/golden.py
class GoldenCase(BaseModel): id, source_id, group: Literal["twin","no_twin"],
                             question, reference, expected_severity
def build_golden(incidents) -> list[GoldenCase]   # 20 twin + 10 no-twin, seeded
```

## Work packages

| WP | Files | Depends on | Parallel |
|---|---|---|---|
| A | `app/golden.py`, `tests/test_golden.py`, `data/golden.jsonl` | none | yes |
| B | `app/judge.py`, `app/metrics.py`, `tests/test_judge.py`, `tests/test_metrics.py` | none | yes |
| C | `app/eval_answers.py`, `tests/test_eval_answers.py` (holdout recall, runner, report with provenance, bootstrap CI, baseline compare) | A, B | no |
| D | `tests/test_quality.py`, `Makefile` (`eval`, `eval-baseline`), `pyproject.toml` (marker), `.gitignore`, `CHANGELOG.md` | C | no |

## Acceptance per package (the gate, run by the lead, not the minion)

- `uv run pytest` passes. New tests use fakes only.
- One mutation per package, shown failing, then restored.
- `scan_private` clean. No new dependency, so no `make audit` change expected.
- B: free-text judge output that is invalid JSON triggers a retry, then `JudgeError`, never a crash.
- C: removing the case's own incident from recall is tested. A baseline-drop test fails the gate. A failure rate above the cap fails the gate.

## Report provenance (C)

Judge model digest, prompt-version hashes, golden-set hash, graph model names, `THRESHOLD`, git SHA, timestamp.

## Open

- Judge calibration file (10 hand labels): deferred to the final lesson pass.
- Concurrency and timeout defaults for the judge: set in B, measured in the first live run.
