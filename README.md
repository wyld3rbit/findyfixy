# FindyFixy

FindyFixy is an incident-triage assistant that runs entirely on your machine.

Paste an incident report. It returns:

- A severity (`sev1` to `sev4`), the likely component and the first action to take.
- A confidence, so a weak answer can be escalated to a larger model.
- The most similar past incidents, found with hybrid search and used as context.

It runs on a 16 GB laptop with open-source parts. No data leaves the machine.

## Quick start

You need [uv](https://docs.astral.sh/uv/), Docker Compose with a runtime (Colima or Docker) and [Ollama](https://ollama.com/).

```bash
echo "SMALL_MODEL=qwen2.5:1.5b" >> code/.env
echo "LARGE_MODEL=qwen2.5:7b" >> code/.env
make setup      # once per clone
```

> `make setup` installs packages, pulls the two models (about 6 GB), starts the database and loads the incidents. It is safe to run again.

Every session, from this folder:

```bash
make up         # start the database (skip it for fake runs and the tracker)
make ui         # then open http://127.0.0.1:8766
make down       # when you are done
```

`make ui` serves one page. The tracker is the page, and the live visualization is embedded at the top. Ctrl-C stops it.

## Usage

```bash
make console          # live page: watch each step run
make ui               # one landing page: http://127.0.0.1:8766
make triage           # triage five fixed questions
make eval-retrieval   # compare the three retrievers
make test             # 371 tests
make up               # start the database (make down stops it)
make help             # everything else
```

Each triage run prints one record and how it was made:

```
A: <severity> | <component> | <next action>
model: <small or large model> | confidence: <0 to 1> | latency: <seconds>
```

## How it works

```
report
  -> Recall        hybrid search over past incidents, reranked, top 3
  -> Investigate   tool call: recent deploys for the service
  -> Small model   Qwen2.5 1.5B drafts the triage record
  -> Gate          confidence >= 0.7: answer
                   otherwise: Qwen2.5 7B, reusing the same evidence
```

| Layer | Implementation |
|---|---|
| Orchestration | LangGraph |
| Models | Ollama: Qwen2.5 1.5B first, 7B on escalation |
| Output contract | Pydantic record, validated in code. One retry that carries the validation error. At most 2 calls per model. |
| Memory | Postgres 16 with pgvector. 200 synthetic incidents, 800 chunks, `bge-small-en-v1.5` embeddings (384 dimensions). |
| Retrieval | BM25 top 50 and vector top 50, fused with Reciprocal Rank Fusion (k=60), reranked by `bge-reranker-base`. |
| Tools | `get_recent_deploys`, served over MCP (stdio). |

Routing is a policy, not a model property. The model's confidence is text it writes, so the 0.7 threshold is a placeholder until it is tuned against a measured quality metric.

## Results

Every number comes from a recorded run.

**Retrieval.** 40 labelled queries: 30 paraphrases of known incidents, 10 exact hostnames.

| Retriever | recall@5 | recall@5, hostnames | MRR | p50 latency |
|---|---|---|---|---|
| Vector | 0.93 | 0.70 | 0.823 | 14 ms |
| Hybrid | 0.97 | 0.90 | 0.821 | 15 ms |
| Hybrid + rerank | 1.00 | 1.00 | 0.781 | 341 ms |

- Embeddings miss exact tokens. BM25 recovers most of that. The reranker recovers the rest at about 24 times the latency.
- The reranker lowers MRR. Near-duplicate incidents fill two of the three recall slots. Not fixed.
- recall@5 is the share of queries with at least one correct incident in the top 5. MRR is the mean of 1 / rank of the first correct one.
- The corpus is templated, so absolute scores flatter. One query is 2.5 points, so small gaps are noise.

**Self-reported confidence is a weak routing signal.** Across 5 runs the small model wrote confidence on two scales: 0 to 1 twice, 0 to 100 three times. It wrote 75 for an answer that gave no checks. None escalated.

**A valid tool call can still be wrong.** The model called the deploys tool in 30 of 30 runs with `hours=1` for a 24-hour question. The call is legal under the schema. It answered correctly 0 of 30 times. Validation cannot catch this. An evaluation can.

**MCP costs process starts, not tokens.** One tool, same model and prompt, 10 runs per path:

| Path | Input tokens | Median time |
|---|---|---|
| Direct function | 499 | 0.57 s |
| MCP, persistent session | 499 | 0.57 s |
| MCP, new session per call | 499 | 1.14 s |

The extra 0.57 s is a server start paid on every call. Keep the session open. One tool understates token cost, and that was not measured with more.

## Decisions

| Decision | Alternative rejected | Reason |
|---|---|---|
| Validate model output in code | Trust Ollama's schema option alone | Its docs do not say bounds are enforced, and the model wrote 75 into a 0-to-1 field. |
| Hybrid retrieval with a reranker in the request path | Vector only | Hostname queries: recall 0.70 to 1.00. The 341 ms is small next to a 1.2 to 4.7 s end-to-end run. |
| Evaluate with each case's own incident held out of Recall | Leave it in the index | Recall would return the answer key and every metric would read high. |
| Write the judge metrics in-house | `ragas` 0.4.3 | An unfixed security advisory (PYSEC-2026-3046) and 31 new packages in a release gate. |
| Count failed judge calls separately | Score them 0 | Judge flakiness would read as worse answers. |
| 7-day delay on new package releases, vulnerability audit on every lock change | Latest versions | Fresh releases are where hijacks land. The audit found 7 known vulnerabilities in untouched packages. They were upgraded. |
| Pin container images by digest and scan with Grype | Tags only | The database image carried 8 critical and 37 high findings in OS packages. Accepted for local use with synthetic data. |
| Scan for local paths, emails and secrets in git hooks | Review by eye | The scanner reports rule and location, never the match. |

## Status

**Built:** routing, validated output, tools over MCP, ingestion, hybrid retrieval with rerank, the Recall step and a labelled retrieval evaluation. Logic is tested with fakes. New logic was broken on purpose to confirm the tests fail.

**Not built:** the answer-quality evaluation gate (designed), tracing, prompt-injection defences, cost budgets, approval for destructive actions and a one-command demo.

## Notes

- All data is synthetic. Secrets go in `code/.env`, which is git-ignored.
- `code/CHANGELOG.md` records what changed and what is open.
- Built with Claude Code.

## License

To be added by the repository owner.
