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
- `code/app/golden.py`: the golden set, built from the corpus.
- `code/app/eval_adapter.py`: FindyFixy as a system under test for the harness in `code/harness/`: holdout recall, the severity label and the twin data.
- `code/app/eval_answers.py`: the answer evaluation's entry point. It builds the live adapter and hands it to the harness command line.
- `code/app/eval_queries.py` and `code/app/eval_retrieval.py`: the labelled queries and the retrieval evaluation.

**Entry points and wiring**
- `code/app/wiring.py`, `code/app/settings.py` and `code/app/paths.py`: the composition root, the configuration read from the environment, and where files live.
- `code/app/console.py`, `code/app/console_fakes.py`, `code/app/dev_reload.py` and `code/app/stress.py`: the live console, its scripted stand-ins, hot reload and stress settings.
- `code/app/console_eval.py`: the console's eval run. It streams the harness's events next to the graph's for each case, and applies the gate read-only. It writes no report.
- `code/app/mcp_server.py`, `code/app/mcp_client.py`, `code/app/mcp_run.py` and `code/app/direct_run.py`: the tool served over MCP and the two measured paths to it.
