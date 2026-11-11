# FindyFixy

The example app the wyld3rness harness is tested on: an incident-triage agent. It is a uv workspace member with its own `code/examples/findyfixy/pyproject.toml`, which declares its libraries, its dependency on the harness and its commands (`findyfixy-console`, `findyfixy-eval` and the rest; the Makefile wraps the common ones).

Beside the package: `code/examples/findyfixy/tests/` (its tests), `code/examples/findyfixy/data/` (the synthetic corpus, golden set and deploy history), `code/examples/findyfixy/evals/` (the frozen baseline and the attestation CI checks), `code/examples/findyfixy/config/prices.toml` (reference prices per token) and `code/examples/findyfixy/docker-compose.yml` (Postgres with pgvector). Settings go in `code/examples/findyfixy/.env`, copied from `code/examples/findyfixy/.env.example`.

The package is `code/examples/findyfixy/src/findyfixy/`. Dependencies point inward: only entry points and `code/examples/findyfixy/src/findyfixy/observability/tracing.py` import the composition root `code/examples/findyfixy/src/findyfixy/wiring.py` or `code/examples/findyfixy/src/findyfixy/settings.py`, and `code/examples/findyfixy/tests/test_architecture.py` enforces it.

**Router and triage**
- `code/examples/findyfixy/src/findyfixy/triage/graph.py`: the small-model-first router graph.
- `code/examples/findyfixy/src/findyfixy/triage/record.py`: the triage record and validate-retry for model output.
- `code/examples/findyfixy/src/findyfixy/tools/investigate.py` and `code/examples/findyfixy/src/findyfixy/tools/deploys.py`: the tool-calling loop and the deploy-history tool.
- `code/examples/findyfixy/src/findyfixy/events.py`: the run events the graph reports.
- `code/examples/findyfixy/src/findyfixy/observability/adapter.py`: those events as OpenTelemetry spans, through the harness's `SpanTree`. The console turns it on when tracing is configured (`open_tracing` in `code/examples/findyfixy/src/findyfixy/observability/tracing.py`); `RunTracer` does the same for eval cases, which share one emitter across threads. `traced_sink` gives the console's event log a `run_traced` event per run, with the link to its trace.
- `code/examples/findyfixy/src/findyfixy/llm.py`: the one place that calls the answering model.

**Memory**
- `code/examples/findyfixy/src/findyfixy/retrieval/corpus.py` and `code/examples/findyfixy/src/findyfixy/retrieval/ingest.py`: build and ingest the synthetic incident corpus.
- `code/examples/findyfixy/src/findyfixy/retrieval/search.py` and `code/examples/findyfixy/src/findyfixy/retrieval/retrieve.py`: nearest-chunk search and the three retrievers.
- `code/examples/findyfixy/src/findyfixy/retrieval/incidents.py`: the small hand-written incident bank.
- `code/examples/findyfixy/src/findyfixy/model_cache.py`: loads the pinned embedder and reranker from the local cache, and downloads only when the cache does not hold them.

**Evaluation**
- `code/examples/findyfixy/src/findyfixy/evals/golden.py`: the golden set, built from the corpus.
- `code/examples/findyfixy/src/findyfixy/evals/adapter.py`: FindyFixy as a system under test for the harness in `code/src/wyld3rness/`: holdout recall, the severity label and the twin data.
- `code/examples/findyfixy/src/findyfixy/evals/answers.py`: the answer evaluation's entry point. It builds the live adapter and hands it to the harness command line. It also declares what the CI gate attests: the entry module whose imports make up the eval path, the data directory, the baseline, the lock file and the compose file, and the database check that stops a gate run on a stale ingest.
- `code/examples/findyfixy/src/findyfixy/evals/queries.py` and `code/examples/findyfixy/src/findyfixy/evals/retrieval.py`: the labelled queries and the retrieval evaluation.

**Entry points and wiring**
- `code/examples/findyfixy/src/findyfixy/wiring.py`, `code/examples/findyfixy/src/findyfixy/settings.py` and `code/examples/findyfixy/src/findyfixy/paths.py`: the composition root, the configuration read from the environment, and where files live.
- `code/examples/findyfixy/src/findyfixy/observability/tracing.py`: `command_trace`, how each command (triage, retrieval eval, ingest, search, the tool paths) becomes a trace, with the tracer built from settings. The MCP server and the console build their provider at startup and use the harness's `CommandTrace` per tool call or request. `code/examples/findyfixy/src/findyfixy/wiring.py` `stage_spans` turns the retriever's stages into spans.
- `code/examples/findyfixy/src/findyfixy/observability/prices.py`: `make obs-prices`, which registers `code/examples/findyfixy/config/prices.toml` with Langfuse through the harness's `code/src/wyld3rness/trace/prices.py`.
- `code/examples/findyfixy/src/findyfixy/console/server.py`, `code/examples/findyfixy/src/findyfixy/console/fakes.py`, `code/examples/findyfixy/src/findyfixy/console/dev_reload.py` and `code/examples/findyfixy/src/findyfixy/console/stress.py`: the live console, its scripted stand-ins, hot reload and stress settings.
- `code/examples/findyfixy/src/findyfixy/console/static/architecture.html`: the page the console serves, the live picture of both lanes. Each box's tooltip says what it does, and `code/examples/findyfixy/tests/test_page_truth.py` keeps the boxes true to the code.
- `code/examples/findyfixy/src/findyfixy/console/eval_run.py`: the console's eval run. It streams the harness's events next to the graph's for each case, and applies the gate read-only. It writes no report.
- `code/examples/findyfixy/src/findyfixy/mcp/server.py`, `code/examples/findyfixy/src/findyfixy/mcp/client.py`, `code/examples/findyfixy/src/findyfixy/mcp/run.py` and `code/examples/findyfixy/src/findyfixy/tools/direct_run.py`: the tool served over MCP and the two measured paths to it.
