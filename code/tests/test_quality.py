"""The answer-quality release gate, as a test.

What it does
    Runs every golden case through the real graph with live models, scores the
    answers, and compares the scores with the committed baseline
    (evals/baseline.json). It fails with the gate's own messages when a change
    made answers worse, when the judge or prompts changed (the comparison would
    be meaningless), or when judge failures or run errors are too high.

Why it is opt-in
    It needs Ollama and the database, and it takes many minutes: about 10 judge
    calls per case, run one case at a time. A default `pytest` run would be slow
    and would fail on a machine with the services off. Run it before a release:

        make up
        uv run pytest -m quality

    The logic of the gate itself is covered by fast tests in
    tests/test_eval_answers.py. This file only wires it to the live system.

What changes at scale
    On CI with a GPU runner this runs on every pull request that touches a
    prompt, a model name, retrieval or the graph. The baseline is then updated
    only by a reviewed commit.
"""

import pytest

from app.eval_answers import main


@pytest.mark.quality
def test_answer_quality_gate() -> None:
    """Exit code 0 means every gate rule passed. The report path is printed."""
    assert main(["--gate"]) == 0
