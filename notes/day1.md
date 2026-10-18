# Day 1 notes: setup and first graph

## Setup

- Python 3.13 (3.15 did not resolve `sentence-transformers` and `ragas`).
- Models: `qwen2.5:1.5b` (small), `qwen2.5:7b` (large), both on Ollama.
- Packages: langgraph 1.2.16, langchain-ollama 1.1.0, pydantic 2.14.1, python-dotenv 1.2.4.
- Threshold: escalate to the large model when confidence is below 0.7.

## Results

Run command: `uv run python -m app.graph`. The first run crashed on question 2 (the small model wrote a confidence of 75, my schema allowed 0 to 1 only). The range check was removed. Numbers below come from the second run, which finished all 5.

| # | Question (short) | Confidence written | Model used | Latency |
|---|---|---|---|---|
| 1 | Pod in CrashLoopBackOff | 75.0 | qwen2.5:1.5b | 2.8s |
| 2 | p95 auth latency doubled | 0.8 | qwen2.5:1.5b | 4.7s |
| 3 | HTTP 429 from vendor API | 90.0 | qwen2.5:1.5b | 1.5s |
| 4 | Feature flag disagreement | 0.742 | qwen2.5:1.5b | 1.2s |
| 5 | Voice-verification failures, one carrier | 95.0 | qwen2.5:1.5b | 4.2s |

- Escalation rate: 0 of 5 (0%).
- Mean latency: 2.9s. All 5 under the 5s small-path target.
- Sample is 5 runs, so no p95.

## What I learned

- The model's confidence is text it writes. Ollama gives no confidence score and no range. The docs do not say whether schema minimum or maximum values are enforced.
- The small model used two scales: 0 to 1 (two answers) and 0 to 100 (three answers), though the field description said 0 to 1.
- Rescaling would not have changed routing here. 75, 90 and 95 become 0.75, 0.9 and 0.95, all above 0.7.
- The model rated itself too high. Question 1 gave an intro and no checks (scored 75). Question 4 gave no steps (scored 0.742). Question 3 called 429 a "condition code" (scored 90).
- Self-reported confidence is cheap and noisy. A better signal is needed: rules, retrieval score, a judge model, or token probabilities.

## Not verified

- Routing is tested with a fake model function (`uv run pytest`). The large-model path was then checked against real models; see "Escalation check" below.
- Fixed after the first commit: LangGraph's drawing omitted the `small -> large` edge because `route()` had no `Literal` return type. Now annotated and covered by a test.

## Escalation check

All 5 normal runs stayed on the small model, so the large path had not run. To test it, I forced escalation on question 1 by setting the threshold impossibly high. No files changed.

| Step | Model | Time | Confidence written |
|---|---|---|---|
| 1 | qwen2.5:1.5b | 2.6s | 95.0 |
| 2 | qwen2.5:7b | 7.2s | 0.95 |

- Total 9.8s. Final `model_used` was `qwen2.5:7b`. The graph passed the question from `small` to `large` and returned the large answer.
- The escalated run took about 3.8 times as long as the small model alone. The 7.2s includes loading the 7B model. How much of it is load time was not measured.
- The 1.5B model wrote 95.0 and the 7B wrote 0.95 for the same question. Only the larger model followed "0 to 1".
- Both models returned only an intro sentence in `answer` for this question, with no list of checks. Cause not diagnosed. Check again on Day 2 with the `Triage` schema.

## Self-check

- [x] 5 answers, each tagged with the model used.
- [x] At least 2 of 5 stayed on the small model (5 of 5).
- [x] Can draw the graph from memory.
