# App

The `code/app/` package. Dependencies point inward: only entry points import the composition root `code/app/wiring.py` or `code/app/settings.py`, and `code/tests/test_architecture.py` enforces it.

**Router and triage**
- `code/app/graph.py`: the small-model-first router graph.
- `code/app/triage.py`: the triage record and validate-retry for model output.
- `code/app/investigate.py` and `code/app/tools.py`: the tool-calling loop and the deploy-history tool.
- `code/app/events.py`: the run events the graph reports.
- `code/app/llm.py`: the one place that calls the answering model.

**Memory**
- `code/app/corpus.py` and `code/app/ingest.py`: build and ingest the synthetic incident corpus.
- `code/app/search.py` and `code/app/retrieve.py`: nearest-chunk search and the three retrievers.
- `code/app/incidents.py`: the small hand-written incident bank.

**Evaluation**
- `code/app/golden.py`, `code/app/judge.py` and `code/app/metrics.py`: the golden set, the judge seam and the metrics.
- `code/app/eval_answers.py`, `code/app/eval_report.py`, `code/app/eval_stats.py` and `code/app/eval_provenance.py`: the answer evaluation, its reports and gate, its statistics and its provenance stamp.
- `code/app/eval_queries.py` and `code/app/eval_retrieval.py`: the labelled queries and the retrieval evaluation.

**Entry points and wiring**
- `code/app/wiring.py`, `code/app/settings.py` and `code/app/paths.py`: the composition root, the configuration read from the environment, and where files live.
- `code/app/console.py`, `code/app/console_fakes.py`, `code/app/dev_reload.py` and `code/app/stress.py`: the live console, its scripted stand-ins, hot reload and stress settings.
- `code/app/mcp_server.py`, `code/app/mcp_client.py`, `code/app/mcp_run.py` and `code/app/direct_run.py`: the tool served over MCP and the two measured paths to it.
