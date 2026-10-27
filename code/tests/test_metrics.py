"""Tests for the three metrics (app/metrics.py), with a fake judge and embedder.

What these tests cover
    Hand-computed scores, every failure path, empty-context behaviour, the
    sentence splitter, cosine, and prompt hashes. Not judge quality: that is
    an empirical question for a live run.

How the fake judge works
    Replies are scripted per schema type, in order. A reply that is a
    `JudgeError` is raised. Every prompt is recorded, so a test can check what
    the judge was actually asked.
"""

import math

import pytest

import app.metrics as metrics
from app.judge import JudgeError
from app.metrics import (
    ACTION_SCORES,
    PROMPT_VERSIONS,
    ActionRating,
    Claims,
    Questions,
    Verdict,
    action_quality,
    context_recall,
    cosine,
    faithfulness,
    response_relevancy,
    split_sentences,
)


def pair(result):
    """The (score, failed) pair of a result, ignoring its trace."""
    return (result.score, result.failed)


class FakeJudge:
    """Plays scripted replies by schema type and records prompts."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.prompts: list[tuple[type, str]] = []

    def ask(self, prompt, schema):
        self.prompts.append((schema, prompt))
        reply = self.script[schema].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def verdicts(*flags):
    """Build a list of `Verdict` from booleans."""
    return [Verdict(supported=f, reason="r") for f in flags]


# ---- faithfulness ----------------------------------------------------------


def test_faithfulness_two_of_three_supported():
    judge = FakeJudge(
        {Claims: [Claims(claims=["a", "b", "c"])], Verdict: verdicts(True, False, True)}
    )
    result = faithfulness(judge, "ans", ["ctx one", "ctx two"])
    assert result.failed is False
    assert result.score == pytest.approx(2 / 3)
    # one split call plus one call per claim
    assert len(judge.prompts) == 4
    # contexts are joined into every support prompt
    assert "ctx one" in judge.prompts[1][1] and "ctx two" in judge.prompts[1][1]
    assert "Claim: b" in judge.prompts[2][1]


def test_faithfulness_all_unsupported_is_zero_not_failed():
    judge = FakeJudge({Claims: [Claims(claims=["a"])], Verdict: verdicts(False)})
    assert pair(faithfulness(judge, "ans", ["c"])) == (0.0, False)


def test_faithfulness_zero_claims_is_failed():
    judge = FakeJudge({Claims: [Claims(claims=[])], Verdict: []})
    assert pair(faithfulness(judge, "ans", ["c"])) == (None, True)


def test_faithfulness_blank_claims_count_as_zero_claims():
    judge = FakeJudge({Claims: [Claims(claims=["  ", ""])], Verdict: []})
    assert faithfulness(judge, "ans", ["c"]).failed is True


def test_faithfulness_judge_error_on_split_fails():
    judge = FakeJudge({Claims: [JudgeError("x")], Verdict: []})
    assert pair(faithfulness(judge, "ans", ["c"])) == (None, True)


def test_faithfulness_judge_error_midway_fails_whole_metric():
    judge = FakeJudge(
        {
            Claims: [Claims(claims=["a", "b", "c"])],
            Verdict: [*verdicts(True), JudgeError("x")],
        }
    )
    assert pair(faithfulness(judge, "ans", ["c"])) == (None, True)


def test_faithfulness_empty_contexts_is_scored_not_failed():
    """No context: every claim is unsupported. Only the claim split asks the judge."""
    judge = FakeJudge({Claims: [Claims(claims=["a", "b"])], Verdict: []})
    result = faithfulness(judge, "ans", [])
    assert pair(result) == (0.0, False)
    assert [schema for schema, _ in judge.prompts] == [Claims]
    assert [row["supported"] for row in result.trace] == [False, False]
    assert result.trace[0]["reason"] == "no context was retrieved"


# ---- response_relevancy ----------------------------------------------------


def fake_embed(vectors):
    """An embedder returning fixed vectors and recording what it was given."""
    seen = []

    def embed(texts):
        seen.append(list(texts))
        return vectors

    embed.seen = seen
    return embed


def test_relevancy_mean_cosine_hand_computed():
    # question vs generated: cos = 1.0, 0.0, 0.5 -> mean 0.5
    s = math.sqrt(3) / 2
    vectors = [[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.5, s]]
    judge = FakeJudge({Questions: [Questions(questions=["q1", "q2", "q3"])]})
    embed = fake_embed(vectors)
    result = response_relevancy(judge, embed, "orig", "ans")
    assert result.failed is False
    assert result.score == pytest.approx(0.5)
    assert embed.seen == [["orig", "q1", "q2", "q3"]]
    assert "3 different questions" in judge.prompts[0][1]


def test_relevancy_respects_n():
    judge = FakeJudge({Questions: [Questions(questions=["q1", "q2"])]})
    embed = fake_embed([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = response_relevancy(judge, embed, "orig", "ans", n=2)
    assert result.score == pytest.approx(0.5)


def test_relevancy_wrong_question_count_fails_before_embedding():
    judge = FakeJudge({Questions: [Questions(questions=["q1", "q2"])]})
    embed = fake_embed([])
    assert pair(response_relevancy(judge, embed, "o", "a")) == (None, True)
    assert embed.seen == []


def test_relevancy_judge_error_fails():
    judge = FakeJudge({Questions: [JudgeError("x")]})
    assert pair(response_relevancy(judge, fake_embed([]), "o", "a")) == (None, True)


def test_relevancy_zero_vector_is_failure_not_crash():
    judge = FakeJudge({Questions: [Questions(questions=["a", "b", "c"])]})
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    assert pair(response_relevancy(judge, fake_embed(vectors), "o", "a")) == (None, True)


def test_relevancy_zero_question_vector_is_failure():
    judge = FakeJudge({Questions: [Questions(questions=["a", "b", "c"])]})
    vectors = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert response_relevancy(judge, fake_embed(vectors), "o", "a").failed is True


def test_relevancy_embedder_wrong_count_fails():
    judge = FakeJudge({Questions: [Questions(questions=["a", "b", "c"])]})
    assert response_relevancy(judge, fake_embed([[1.0]]), "o", "a").failed is True


# ---- context_recall --------------------------------------------------------


def test_recall_one_of_two_sentences_attributed():
    judge = FakeJudge({Verdict: verdicts(True, False)})
    result = context_recall(judge, "First fact. Second fact.", ["ctx"])
    assert result.failed is False
    assert result.score == pytest.approx(0.5)
    assert "Sentence: First fact." in judge.prompts[0][1]
    assert "Sentence: Second fact." in judge.prompts[1][1]


def test_recall_empty_reference_is_failed():
    judge = FakeJudge({Verdict: []})
    assert pair(context_recall(judge, "   ", ["ctx"])) == (None, True)
    assert judge.prompts == []


def test_recall_judge_error_fails():
    judge = FakeJudge({Verdict: [*verdicts(True), JudgeError("x")]})
    assert pair(context_recall(judge, "One. Two.", ["c"])) == (None, True)


def test_recall_empty_contexts_is_zero_not_failed():
    """No context: nothing is attributable, and the judge is never asked."""
    judge = FakeJudge({Verdict: []})
    result = context_recall(judge, "One. Two.", [])
    assert pair(result) == (0.0, False)
    assert judge.prompts == []
    assert len(result.trace) == 2


# ---- helpers ---------------------------------------------------------------


def test_split_sentences():
    assert split_sentences("A b. C d!  E f?\nG h") == ["A b.", "C d!", "E f?", "G h"]
    assert split_sentences("") == []


def test_cosine_known_values_and_guards():
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 1], [2, 2]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine([0, 0], [1, 0])
    with pytest.raises(ValueError):
        cosine([1], [1, 0])


# ---- prompt hashes ---------------------------------------------------------


def test_prompt_versions_are_stable_hex_strings():
    assert set(PROMPT_VERSIONS) == {
        "faithfulness",
        "response_relevancy",
        "context_recall",
        "action_quality",
    }
    for digest in PROMPT_VERSIONS.values():
        assert len(digest) == 64 and int(digest, 16) >= 0
    assert len(set(PROMPT_VERSIONS.values())) == 4
    assert metrics._sha(metrics.QUESTIONS_PROMPT) == PROMPT_VERSIONS["response_relevancy"]


def test_prompt_hash_changes_when_a_template_changes(monkeypatch):
    before = metrics._sha(metrics.CLAIMS_PROMPT, metrics.SUPPORT_PROMPT)
    monkeypatch.setattr(metrics, "SUPPORT_PROMPT", metrics.SUPPORT_PROMPT + " ")
    after = metrics._sha(metrics.CLAIMS_PROMPT, metrics.SUPPORT_PROMPT)
    assert before == PROMPT_VERSIONS["faithfulness"]
    assert after != before


# ---- traces: the evidence behind each score ---------------------------------


def test_faithfulness_trace_lists_each_claim_with_its_verdict_and_reason():
    judge = FakeJudge(
        {
            Claims: [Claims(claims=["a", "b"])],
            Verdict: [Verdict(supported=True, reason="stated"), Verdict(supported=False, reason="not in context")],
        }
    )
    result = faithfulness(judge, "ans", ["ctx"])
    assert result.trace == (
        {"claim": "a", "supported": True, "reason": "stated"},
        {"claim": "b", "supported": False, "reason": "not in context"},
    )


def test_recall_trace_lists_each_reference_sentence():
    judge = FakeJudge({Verdict: verdicts(True, False)})
    result = context_recall(judge, "First point. Second point.", ["ctx"])
    assert [row["sentence"] for row in result.trace] == ["First point.", "Second point."]
    assert [row["supported"] for row in result.trace] == [True, False]


def test_relevancy_trace_lists_each_generated_question_with_its_similarity():
    judge = FakeJudge({Questions: [Questions(questions=["q1", "q2", "q3"])]})
    vectors = {"orig": [1.0, 0.0], "q1": [1.0, 0.0], "q2": [0.0, 1.0], "q3": [1.0, 0.0]}
    result = response_relevancy(judge, lambda texts: [vectors[t] for t in texts], "orig", "ans")
    assert result.trace == (
        {"question": "q1", "similarity": 1.0},
        {"question": "q2", "similarity": 0.0},
        {"question": "q3", "similarity": 1.0},
    )


def test_a_failed_result_has_no_trace():
    judge = FakeJudge({Claims: [JudgeError("x")]})
    assert faithfulness(judge, "ans", ["c"]).trace == ()


def test_a_runaway_judge_reason_is_capped_before_it_is_stored():
    from app.metrics import MAX_REASON_CHARS

    judge = FakeJudge({Claims: [Claims(claims=["a"])], Verdict: [Verdict(supported=True, reason="x" * 5000)]})
    (row,) = faithfulness(judge, "ans", ["c"]).trace
    assert len(row["reason"]) == MAX_REASON_CHARS and row["reason"].endswith("...")


# ---- why a result failed, and the sentence splitter -------------------------


def test_every_failure_says_why():
    """A timeout, a bad count and an empty answer must not look the same in a report."""
    judge_down = FakeJudge({Claims: [JudgeError("timed out")]})
    assert "timed out" in faithfulness(judge_down, "ans", ["c"]).error
    assert "no claims" in faithfulness(FakeJudge({Claims: [Claims(claims=[])]}), "a", ["c"]).error
    wrong_count = FakeJudge({Questions: [Questions(questions=["only one"])]})
    assert "1 questions, expected 3" in response_relevancy(wrong_count, fake_embed([]), "q", "a").error
    assert "no sentences" in context_recall(FakeJudge({}), "   ", ["c"]).error
    assert faithfulness(FakeJudge({Claims: [Claims(claims=["a"])], Verdict: verdicts(True)}), "a", ["c"]).error is None


def test_a_reference_split_is_the_same_whether_or_not_a_line_ends_in_a_full_stop():
    with_stop = "Root cause: A typo passed review.\nRemediation: Add bounds."
    without = "Root cause: A typo passed review\nRemediation: Add bounds"
    assert len(split_sentences(with_stop)) == len(split_sentences(without)) == 2


def test_the_splitter_is_part_of_the_context_recall_version():
    """Change how sentences are cut and the report must show the metric changed."""
    import hashlib

    from app.metrics import ATTRIBUTION_PROMPT, PROMPT_VERSIONS, _SENTENCE_END

    expected = hashlib.sha256("\x1f".join([ATTRIBUTION_PROMPT, _SENTENCE_END.pattern]).encode()).hexdigest()
    assert PROMPT_VERSIONS["context_recall"] == expected


# ---- action_quality ------------------------------------------------------------


@pytest.mark.parametrize("rating,score", [("helpful", 1.0), ("partial", 0.5), ("unhelpful", 0.0)])
def test_action_quality_maps_the_three_ratings_to_scores(rating: str, score: float) -> None:
    judge = FakeJudge({ActionRating: [ActionRating(rating=rating, reason="because")]})
    result = action_quality(judge, "Roll back the config", "Root cause: bad config.\nRemediation: validate in CI.")
    assert (result.score, result.failed, result.error) == (score, False, None)
    assert result.trace == ({"action": "Roll back the config", "rating": rating, "reason": "because"},)


def test_action_quality_shows_the_judge_the_reference_and_the_action_only() -> None:
    judge = FakeJudge({ActionRating: [ActionRating(rating="helpful")]})
    action_quality(judge, "Check the deploy", "Root cause: X.\nRemediation: Y.")
    ((_, prompt),) = judge.prompts
    assert "Check the deploy" in prompt and "Root cause: X." in prompt and "Remediation: Y." in prompt


def test_action_quality_fails_with_a_reason_when_there_is_no_action_or_the_judge_errors() -> None:
    assert "no next action" in action_quality(FakeJudge({}), "   ", "ref").error
    down = FakeJudge({ActionRating: [JudgeError("timed out")]})
    result = action_quality(down, "do it", "ref")
    assert result.failed and "timed out" in result.error and result.trace == ()


def test_an_unknown_rating_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValueError):
        ActionRating(rating="great")  # type: ignore[arg-type]


def test_action_scores_cover_every_rating() -> None:
    assert set(ACTION_SCORES) == {"helpful", "partial", "unhelpful"}

