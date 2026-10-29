# LLM Production Sprint

10 days in a row. 10 sessions of about 2 hours. Sat 17 Oct to Mon 26 Oct 2026.

## What you get out of it

1. **Every tool in the stack, built hands-on.** LangGraph, Ollama, hybrid-search memory, RAGAS, Langfuse, direct APIs over MCP, a token-exchange credential broker. You will have built each one and measured it.
2. **A working repo.** A small autonomous event detection, determination and mitigation agent, end to end: design, memory, evals, guardrails, tracing, cost control, deploy. Portfolio-ready.
3. **A show and tell.** One command starts the whole stack. A demo screen, a 3-minute script and 6 slides, built on Day 10.
4. **VP-level talking points.** Each day ends with one interview drill. You answer as "what I hold a team to", backed by a number you produced.
5. **An architecture decision record.** One page in Larson's format: diagnosis, guiding policies, coherent actions. Usable in a system-design round.

## What it is not

Not ML research. No model training or fine-tuning. The scope is the production stack around a model: what an engineering leader owns.

## The product

**FindyFixy.** An on-call engineer pastes an incident. In seconds it returns severity, the likely component, the first action, and the 3 most similar past incidents with their fixes. Safe actions run. Risky ones wait for a human to approve.

Built to production standards, measured on your machine:

| Bar | Target |
|---|---|
| Answer quality | Faithfulness at or above the Day 5 gate, checked in CI |
| Memory | Recall@5 measured on a labelled set |
| Latency | p95 per triage recorded, small-model path under 5 seconds |
| Cost | Cost per triage tracked; down 30% from baseline by Day 8 |
| Safety | 0 of 20 injection attacks cause an unauthorised action |
| Access | No long-lived secrets in the agent; per-call scoped tokens |
| Operability | Every run traced; nightly regression eval; one-flip rollback |
| Deploy | `make demo` starts the whole product from a fresh clone |

## Self-contained

- **One setup.** Day 1 checks the tools already on your Mac and pulls two models. Nothing else is installed system-wide.
- **Everything else lives in `code/`.** Python packages go in its `.venv` through `uv`. Every service (Postgres with pgvector, Langfuse, LiteLLM, the API, the demo screen) runs from compose files inside the repo.
- **Clean removal.** `make stop`, then delete the folder and `ollama rm` the two models. Nothing is left behind.

## How it runs

- **Read in the tracker.** Run `make ui` and open http://127.0.0.1:8766. Each day opens in place: concept, lab, drill, then the self-check you tick.
- **Build in Claude Code.** In a terminal, `cd` into this folder, run `claude`, and say "start day 1". It follows `CLAUDE.md`: one step at a time, code in `code/`. It runs on your Mac, so it reaches your local Ollama and containers.
- **Concept, then build, then explain.** You read why a component exists, build it, measure it, then say it aloud as a leader would. The number you measured is what makes the answer credible.
- **Progress.** Claude Code keeps the record in `record/progress.json` as you pass each check, and logs every session. Your browser ticks are a convenience on top.
- **Exit bar for each day:** the self-check passes and you can answer the drill in under 90 seconds without notes.

## The arc

| Day | Theme | You leave with |
|---|---|---|
| **Week 1: Build** |||
| 1 | Setup and first graph | Ollama plus LangGraph running; a small-model-first router that escalates to a larger model |
| 2 | Structured output and tools | Schema-validated outputs; a direct-API tool vs an MCP tool, cost compared |
| 3 | Memory I: ingestion | Synthetic incident corpus, chunked, embedded, stored in pgvector |
| 4 | Memory II: retrieval | BM25 plus vector, reciprocal-rank fusion, cross-encoder rerank; recall@5 measured |
| 5 | Evaluation | Golden set, RAGAS metrics, a CI gate that fails a bad change |
| **Week 2: Ship** |||
| 6 | Observability | Langfuse traces; cost, latency and token dashboards; OpenTelemetry GenAI fields |
| 7 | Security by design | OWASP LLM Top 10 walk-through, injection tests, a scoped short-lived credential broker |
| 8 | Cost and context control | Gateway budgets, caching, context limits, routing thresholds; before/after cost per run |
| 9 | Deploy and operate | API and demo screen in containers, human-on-the-loop approval, scheduled regression eval |
| 10 | Capstone and show and tell | One-command demo, the ADR, slides, a 3-minute live walkthrough |

## Open-source bill of materials

Everything that ships in the repo is open source. Python packages install into the repo with `uv`; services run as containers.

| Component | Role | License |
|---|---|---|
| uv | Python environments, locked with hashes | MIT / Apache-2.0 |
| Docker Compose | Runs every service from files in the repo | Apache-2.0 |
| Ollama | Local model runtime | MIT |
| Qwen2.5 1.5B and 7B | Small and large models | Apache-2.0 |
| LangGraph, LangChain | Orchestration | MIT |
| Postgres, pgvector | Database, vector search | PostgreSQL License |
| sentence-transformers, BGE models | Embeddings, reranker | Apache-2.0, MIT |
| rank-bm25 | Keyword search | Apache-2.0 |
| RAGAS | Answer-quality evals | Apache-2.0 |
| pytest | Tests and CI gates, including the injection suite | MIT |
| pip-audit | Known-vulnerability scan of Python dependencies | Apache-2.0 |
| Grype | Vulnerability scan of container images | Apache-2.0 |
| Langfuse | Tracing and dashboards (core; `/ee` features need a key, none used here) | MIT |
| OpenTelemetry | Trace standard | Apache-2.0 |
| LiteLLM proxy | Gateway, budgets, routing (open-source tier) | MIT |
| MCP Python SDK | Tool protocol | MIT |
| langchain-mcp-adapters | Connects MCP tools to LangChain and LangGraph | MIT |
| FastAPI, PyJWT | API, credential broker | MIT |
| Streamlit | Demo screen | Apache-2.0 |

**Not open source, and not part of the shipped stack:** Claude Code, your build partner. Docker Desktop, if that is your container runtime (free for personal use); the repo runs the same on Colima. A hosted model (Claude or another) is an optional swap on Day 8 to compare cost and quality.

**Two model choices made on purpose.** Qwen2.5 3B and 72B carry a separate Qwen license, so the course uses 1.5B and 7B, which also fit a 16 GB Mac. Llama models carry Meta's community license, so they are left out.

## Install safety

- `uv` writes a lockfile with hashes. Commit `uv.lock`.
- A release cooldown (`exclude-newer` in `pyproject.toml`) keeps freshly published, possibly hijacked versions out. It is a fixed date that moves forward deliberately, never to today, and it cannot be older than the packages already locked. A security fix newer than the date gets a documented per-package exception.
- `make audit` runs pip-audit against the lockfile and checks the environment for unexpected `.pth` startup hooks. It runs after every dependency change.
- LiteLLM versions 1.82.7 and 1.82.8 on PyPI were compromised on 24 March 2026. Day 8 excludes them explicitly.
- Container images are pinned by digest and scanned with Grype before first use. Never `latest`.
- Model IDs are recorded after each pull.

## Hardware

Built for your 16 GB Mac. The 7B model takes about 5 GB, the container runtime 6 GB (8 GB on Langfuse days), macOS the rest. Ollama loads one model at a time. Stop Langfuse when you are not using it, and close heavy apps during labs.

## Library drift

These libraries ship fast. If an import or signature in a lab fails, tell Claude Code to fix it against the installed version. That is part of the job you are training for.
