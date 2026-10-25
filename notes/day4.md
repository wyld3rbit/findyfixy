# Day 4 notes: memory II, hybrid retrieval

Date: 2026-10-25. Reranker: `BAAI/bge-reranker-base` (revision and weights hash in `models.md`).

## Status of the numbers

The first table is an ASSISTANT run. The learner then ran `uv run python -m app.eval_retrieval` from `code/` on the same machine (2026-10-25). Recall@5, all@5 and MRR matched the assistant's run exactly. Latency differed a little (learner's table below). The analysis below uses the shared quality numbers.

## The eval set

- `data/retrieval_eval.jsonl`: 40 queries. 30 meaning (paraphrases of the seeded pairs, 19 from the summary, 11 from the root cause) and 10 exact-token (hostnames found in 1 to 3 incidents).
- Meaning labels are both twins, taken from `twin_of`. The two identical gateway pairs share one query with four labels.
- Token labels are computed: every incident whose text contains the hostname.

## Assistant run (k = 5, CPU)

| Retriever | Queries | n | recall@5 | all@5 | MRR | p50 ms |
|---|---|---|---|---|---|---|
| vector | all | 40 | 0.93 | 0.72 | 0.823 | 18 |
| vector | token | 10 | 0.70 | 0.70 | 0.700 | 16 |
| vector | meaning | 30 | 1.00 | 0.73 | 0.864 | 17 |
| hybrid | all | 40 | 0.97 | 0.68 | 0.821 | 18 |
| hybrid | token | 10 | 0.90 | 0.90 | 0.800 | 17 |
| hybrid | meaning | 30 | 1.00 | 0.60 | 0.828 | 19 |
| hybrid_rerank | all | 40 | 1.00 | 0.65 | 0.781 | 356 |
| hybrid_rerank | token | 10 | 1.00 | 0.90 | 0.833 | 322 |
| hybrid_rerank | meaning | 30 | 1.00 | 0.57 | 0.763 | 358 |

## Learner's run (same eval, same code)

| Retriever | Queries | n | recall@5 | all@5 | MRR | p50 ms |
|---|---|---|---|---|---|---|
| vector | all | 40 | 0.93 | 0.72 | 0.823 | 14 |
| vector | token | 10 | 0.70 | 0.70 | 0.700 | 13 |
| vector | meaning | 30 | 1.00 | 0.73 | 0.864 | 14 |
| hybrid | all | 40 | 0.97 | 0.68 | 0.821 | 15 |
| hybrid | token | 10 | 0.90 | 0.90 | 0.800 | 14 |
| hybrid | meaning | 30 | 1.00 | 0.60 | 0.828 | 16 |
| hybrid_rerank | all | 40 | 1.00 | 0.65 | 0.781 | 341 |
| hybrid_rerank | token | 10 | 1.00 | 0.90 | 0.833 | 312 |
| hybrid_rerank | meaning | 30 | 1.00 | 0.57 | 0.763 | 337 |

Identical quality numbers on a second run show the eval is deterministic. Only latency moves.

## What the table says

- Exact tokens: recall@5 goes 0.70 (vector), 0.90 (hybrid), 1.00 (rerank). Vector missed 3 of 10 hostnames in the top 5 (t03 at rank 10, t06 at rank 9, t09 at rank 8). BM25 finds them. This is the point of hybrid.
- Meaning queries: all three have recall@5 of 1.00. Hybrid and rerank do not improve it. They slightly lower MRR (0.864, 0.828, 0.763) and all@5 (0.73, 0.60, 0.57).
- Why meaning gets worse, from the per-query ranks (assistant diagnostic, not saved): each failure pattern appears in about nine incidents (different services and numbers). A paraphrase matches all of them. The twin's partner is often outranked by other incidents of the same pattern, at ranks 4 to 9. That is retrieval working on meaning, but the label only counts the twin, so all@5 and MRR punish it. The labels measure "find this exact incident", and the queries ask "find incidents like this".
- Rerank costs about 340 ms per query against about 18 ms. That is 20 times the latency for 0.07 of overall recall@5 (0.93 to 1.00), all of it on token queries.
- t06 (`notifications-001`): hybrid ranked it outside the top 10 (BM25 weight diluted), the reranker, which sees the top 30 chunks, put it at rank 1. A rerank stage can rescue a candidate that fusion left low, but only if it is in the pool.

## Limits of this measurement

- 40 queries. One query is 2.5 points. Differences of a few points are noise.
- Templated data, one writer for the meaning queries.
- Latency is a median over 40 calls on a laptop CPU, with Ollama idle.

## Recall in the prompt (assistant run)

- Query: "p95 latency on auth rose sharply after a certificate rotation. What should we check?" with `hybrid_rerank`, k = 3.
- Top 3: INC-0073, INC-0032, INC-0074. INC-0073 and INC-0074 are the seeded twins, so two of three prompt slots hold the same incident in two voices.
- Prompt size: 1190 characters, about 300 tokens of the 4096 window.
- Open for later: de-duplicate near-twins before filling the prompt (for example maximal marginal relevance). Not built.

## Other checks

- Audit: `filelock` 4.2.0 to 4.2.1 and `setuptools` 84.0.0 to 84.2.0 (approved), `make audit` clean afterwards.
- Tests: 371 pass (fast). Mutations, each caught: BM25 ignores the service filter, no incident collapsing, hybrid ignores BM25, rerank pool uncapped, prior incidents not passed to the models.
- `rank-bm25` 0.2.2: Apache-2.0, last release 2022-02-16, about 8.7 million downloads in the last month (pypistats, 2026-10-25).
