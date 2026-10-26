"""Day 5: three model-graded metrics as pure functions over injected seams.

What it does
    - `faithfulness`: is every claim in the answer supported by the contexts?
    - `response_relevancy`: does the answer address the question asked?
    - `context_recall`: do the contexts contain what the reference answer says?
    The definitions follow RAGAS (Retrieval-Augmented Generation Assessment),
    so scores stay comparable to published numbers. We implement them here
    because the project rejected the `ragas` dependency.

Seams
    `judge` is a `Judge` (app/judge.py). `embed` is an `Embedder`. Both are
    arguments, so tests use fakes and nothing loads at import time.

Failure model
    A metric returns `MetricResult(score, failed)`. `failed=True` means no
    usable score: a judge error, zero claims, a wrong number of questions, a
    zero vector, an empty reference. The score is then None. It is never 0.0
    or 1.0, because those are real verdicts. Means must exclude None. The
    runner counts failures separately and reports the count next to the mean,
    otherwise a judge that crashes on hard cases would quietly raise the
    average, or lower it, depending on what we substituted.

Empty contexts are legal
    Some cases retrieve only unrelated incidents or nothing. Then every claim
    is unsupported (faithfulness 0.0) and nothing can be attributed
    (context_recall 0.0). Those are real scores, not failures.

Judge failure modes (read before trusting a number)
    - Position bias: judges favour the first or last item in a prompt.
    - Verbosity bias: longer, fluent text is rated as more supported.
    - A 7B judge is weaker than a frontier one. Expect noise and leniency.
      Use scores to compare two systems on the same judge, not as absolutes.
    - Claim splitting is the weakest step. A claim that is too coarse hides an
      unsupported part. A claim that is not self-contained cannot be judged.
    - Temperature 0 reduces run-to-run drift but does not remove it.
    - Cost: faithfulness is 1 + claims calls per sample. At scale, sample.
    - Prompt text is part of the metric. `PROMPT_VERSIONS` hashes it so a
      result file can record which wording produced it.
"""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

from pydantic import BaseModel, Field

from app.judge import Judge, JudgeError

# Same shape as app.ingest.Embedder. Redefined so this module does not import
# psycopg and pgvector just to name a function type.
Embed = Callable[[Sequence[str]], list[list[float]]]


class MetricResult(NamedTuple):
    """One metric on one sample. `score` is None exactly when `failed`."""

    score: float | None
    failed: bool


def _failed() -> MetricResult:
    """The failed result. One constructor so the invariant has one home."""
    return MetricResult(None, True)


# ---- Prompt templates. Module-level so they can be hashed. -----------------

CLAIMS_PROMPT = (
    "Break the answer into atomic claims. Each claim states one fact and is "
    "self-contained: replace pronouns with the thing they name, so the claim "
    "makes sense alone. Do not add facts that are not in the answer.\n\n"
    "Answer:\n{answer}\n\n"
    "Reply with JSON: claims (a list of strings)."
)

SUPPORT_PROMPT = (
    "Decide whether the claim is supported by the context. Supported means "
    "the context states it or it follows directly. Use only the context, not "
    "outside knowledge.\n\n"
    "Context:\n{context}\n\n"
    "Claim: {claim}\n\n"
    "Reply with JSON: supported (true or false) and reason (one short "
    "sentence)."
)

QUESTIONS_PROMPT = (
    "Write {n} different questions that the answer below appears to address. "
    "Use only the answer.\n\n"
    "Answer:\n{answer}\n\n"
    "Reply with JSON: questions (a list of exactly {n} strings)."
)

ATTRIBUTION_PROMPT = (
    "Decide whether the sentence can be attributed to the context: the "
    "context states it or clearly implies it. Use only the context.\n\n"
    "Context:\n{context}\n\n"
    "Sentence: {sentence}\n\n"
    "Reply with JSON: supported (true or false) and reason (one short "
    "sentence)."
)


def _sha(*texts: str) -> str:
    """SHA-256 hex of the templates joined with a separator that cannot clash."""
    return hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()


# Metric name -> hash of the template text(s) it uses. Computed, never typed.
PROMPT_VERSIONS: dict[str, str] = {
    "faithfulness": _sha(CLAIMS_PROMPT, SUPPORT_PROMPT),
    "response_relevancy": _sha(QUESTIONS_PROMPT),
    "context_recall": _sha(ATTRIBUTION_PROMPT),
}


# ---- Judge reply schemas. ---------------------------------------------------


class Claims(BaseModel):
    """The answer split into claims."""

    claims: list[str]


class Verdict(BaseModel):
    """One support decision, with a reason kept for debugging."""

    supported: bool
    reason: str = Field(default="", description="one short sentence")


class Questions(BaseModel):
    """Questions the answer appears to address."""

    questions: list[str]


# ---- Pure helpers. ----------------------------------------------------------

# Split after . ! or ? followed by whitespace. Deterministic and model-free.
# Known limit: abbreviations ("e.g. ") split wrongly. Acceptable for short
# synthetic references; a real corpus would need a proper sentence tokenizer.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentences. Pure."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Raises `ValueError` on a zero vector or size mismatch.

    A zero vector has no direction. Returning 0.0 would pass a broken embedding
    off as "unrelated", so the caller must see it as a failure.
    """
    if len(a) != len(b):
        raise ValueError("vector sizes differ")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("zero vector")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _join(contexts: Sequence[str]) -> str:
    """Join contexts into one block. Empty contexts give an empty block."""
    return "\n\n".join(contexts)


def _fraction_supported(
    judge: Judge, template: str, key: str, items: Sequence[str], context: str
) -> float:
    """Ask one verdict per item and return supported / total. Raises JudgeError.

    One call per item, not one batch: a 7B model asked for many verdicts at
    once loses count and drifts toward the same answer for all of them.
    """
    supported = 0
    for item in items:
        prompt = template.format(context=context, **{key: item})
        if judge.ask(prompt, Verdict).supported:
            supported += 1
    return supported / len(items)


# ---- Metrics. ---------------------------------------------------------------


def faithfulness(
    judge: Judge, answer: str, contexts: Sequence[str]
) -> MetricResult:
    """Fraction of the answer's claims that the contexts support.

    Zero claims means failed (nothing to score). Any `JudgeError` fails the
    whole metric: dropping one claim would silently change the denominator.
    Empty contexts: every claim is unsupported, so the score is 0.0, not failed.
    """
    try:
        raw = judge.ask(CLAIMS_PROMPT.format(answer=answer), Claims).claims
        claims = [c.strip() for c in raw if c.strip()]
        if not claims:
            return _failed()
        score = _fraction_supported(
            judge, SUPPORT_PROMPT, "claim", claims, _join(contexts)
        )
    except JudgeError:
        return _failed()
    return MetricResult(score, False)


def response_relevancy(
    judge: Judge, embed: Embed, question: str, answer: str, n: int = 3
) -> MetricResult:
    """Mean cosine between the question and `n` questions the answer implies.

    An answer that drifts off topic implies questions unlike the real one.
    Failed when the judge errors, returns a count other than `n`, the embedder
    returns the wrong number of vectors, or any vector is zero.
    """
    try:
        generated = judge.ask(
            QUESTIONS_PROMPT.format(answer=answer, n=n), Questions
        ).questions
    except JudgeError:
        return _failed()
    if len(generated) != n:
        return _failed()
    vectors = embed([question, *generated])
    if len(vectors) != n + 1:
        return _failed()
    try:
        sims = [cosine(vectors[0], v) for v in vectors[1:]]
    except ValueError:
        return _failed()
    return MetricResult(sum(sims) / n, False)


def context_recall(
    judge: Judge, reference: str, contexts: Sequence[str]
) -> MetricResult:
    """Fraction of reference sentences the contexts support.

    Measures retrieval, not generation: did we fetch what the right answer
    needs? Empty reference means failed. Empty contexts give 0.0, not failed.
    """
    sentences = split_sentences(reference)
    if not sentences:
        return _failed()
    try:
        score = _fraction_supported(
            judge, ATTRIBUTION_PROMPT, "sentence", sentences, _join(contexts)
        )
    except JudgeError:
        return _failed()
    return MetricResult(score, False)
