# App

The `code/app/` package. Dependencies point inward: only entry points and `code/app/tracing.py` import the composition root `code/app/wiring.py` or `code/app/settings.py`, and `code/tests/test_architecture.py` enforces it.

**Router and triage**
- `code/app/graph.py`: the small-model-first router graph.
- `code/app/triage.py`: the triage record and validate-retry for model output.
- `code/app/investigate.py` and `code/app/tools.py`: the tool-calling loop and the deploy-history tool.
- `code/app/events.py`: the run events the graph reports.
- `code/app/trace_adapter.py`: those events as OpenTelemetry spans, through the harness's `SpanTree`. The console turns it on when tracing is configured (`open_tracing` in `code/app/tracing.py`); `RunTracer` does the same for eval cases, which share one emitter across threads. `traced_sink` gives the console's event log a `run_traced` event per run, with the link to its trace.
- `code/app/llm.py`: the one place that calls the answering model.

**Memory**
- `code/app/corpus.py` and `code/app/ingest.py`: build and ingest the synthetic incident corpus.
- `code/app/search.py` and `code/app/retrieve.py`: nearest-chunk search and the three retrievers.
- `code/app/incidents.py`: the small hand-written incident bank.
- `code/app/model_cache.py`: loads the pinned embedder and reranker from the local cache, and downloads only when the cache does not hold them.

**Evaluation**
- `code/app/golden.py`: the golden set, built from the corpus.
- `code/app/eval_adapter.py`: FindyFixy as a system under test for the harness in `code/harness/`: holdout recall, the severity label and the twin data.
- `code/app/eval_answers.py`: the answer evaluation's entry point. It builds the live adapter and hands it to the harness command line. It also declares what the CI gate attests: the entry module whose imports make up the eval path, the data directory, the baseline, the lock file and the compose file, and the database check that stops a gate run on a stale ingest.
- `code/app/eval_queries.py` and `code/app/eval_retrieval.py`: the labelled queries and the retrieval evaluation.

**Entry points and wiring**
- `code/app/wiring.py`, `code/app/settings.py` and `code/app/paths.py`: the composition root, the configuration read from the environment, and where files live.
- `code/app/tracing.py`: `command_trace`, how each command (triage, retrieval eval, ingest, search, the tool paths) becomes a trace, with the tracer built from settings. The MCP server and the console build their provider at startup and use the harness's `CommandTrace` per tool call or request. `code/app/wiring.py` `stage_spans` turns the retriever's stages into spans.
- `code/app/prices.py`: `make obs-prices`, which registers `code/config/prices.toml` with Langfuse through the harness's `code/harness/trace/prices.py`.
- `code/app/console.py`, `code/app/console_fakes.py`, `code/app/dev_reload.py` and `code/app/stress.py`: the live console, its scripted stand-ins, hot reload and stress settings.
- `code/app/console_eval.py`: the console's eval run. It streams the harness's events next to the graph's for each case, and applies the gate read-only. It writes no report.
- `code/app/mcp_server.py`, `code/app/mcp_client.py`, `code/app/mcp_run.py` and `code/app/direct_run.py`: the tool served over MCP and the two measured paths to it.
