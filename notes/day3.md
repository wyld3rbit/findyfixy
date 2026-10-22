# Day 3 notes: memory I, ingestion

Date: 2026-10-21. Embedding model: `BAAI/bge-small-en-v1.5` (revision and weights hash in `models.md`).

## Self-check (learner's own runs, from `code/`)

| Item | Command | Result |
|---|---|---|
| About 800 chunks | `docker compose exec db psql -U postgres -d triage -c "SELECT count(*) FROM chunks;"` | 800 |
| Second ingestion run writes 0 | `uv run python -m app.ingest` | `chunks 800: written 0, deleted 0, unchanged 800` |
| Top 5 includes a seeded near-duplicate | `uv run python -m app.search` | 2 of 5 hits belong to a seeded pair; both members of one pair (INC-0073 and INC-0074) are in the top 5 |

The learner's search output was identical to the assistant's run, distance for distance.

## The corpus

- `data/incidents.jsonl`: 200 synthetic incidents, 8 services, 20 failure patterns told in two voices.
- 180 originals and 20 seeded twins. A twin is the same incident (same service, severity and numbers) told in the other voice, dated 1 to 4 days later.
- 800 chunks: 200 each of summary, timeline, root cause and remediation. All 384 dimensions, stored with the model name.

## First query (learner's run)

Query: "auth latency spike after certificate rotation"

| Rank | Distance | Chunk | Service | Pair |
|---|---|---|---|---|
| 1 | 0.089 | INC-0032:summary | auth | none |
| 2 | 0.091 | INC-0073:summary | auth | INC-0074 (both in results) |
| 3 | 0.098 | INC-0054:summary | auth | none |
| 4 | 0.101 | INC-0074:summary | auth | INC-0073 (both in results) |
| 5 | 0.157 | INC-0169:summary | notifications | none |

- All five hits are certificate-rotation summaries. The distances are 0.089 to 0.157.
- INC-0054 ("14 times normal") sits between the twins INC-0073 and INC-0074 ("41 times normal"), 0.003 away from INC-0074. The model ranks shared wording about as close as shared facts. It does not separate 14 from 41.
- One query on template data is close to a best case. Nine incidents share the certificate pattern, so the family as a whole ranks high. Day 4 measures recall across all 20 pairs.

## Assistant runs (not the learner's)

- First ingestion run: 800 chunks written in about 40 s on CPU, including the one-time model download. Re-run: 0.3 s, because the model is not loaded when nothing changed.
- Query plan at 800 rows: Postgres chose a sequential scan (1.27 ms, 800 rows read). With `enable_seqscan = off` it used the HNSW index (0.78 ms). At this size the index is not needed. Behaviour at larger sizes is not measured.
- Tests: 324 fast tests and 7 database tests pass.

## Images scanned before first use (Grype v0.120.1, 2026-10-21)

- `pgvector/pgvector:pg16` (Debian 12): 8 unique Critical, 37 unique High, all in base OS packages. Learner accepted: local only, synthetic data.
- `adminer:6.1.1` (Alpine): 0 Critical, 6 unique High, all in OS packages.
