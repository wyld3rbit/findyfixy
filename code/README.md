# Code

FindyFixy, a prototype incident-triage assistant, and the harness around it. It runs on one 16 GB machine, with open-source models and synthetic data only.

- `code/examples/findyfixy/src/findyfixy//` is the FindyFixy package. `code/examples/findyfixy/README.md` maps its modules.
- `code/src/wyld3rness/` is the harness: evaluation (`code/src/wyld3rness/eval/`) and tracing (`code/src/wyld3rness/trace/`). It never imports the app. `code/src/wyld3rness/README.md` explains the seam.
- `code/tests/` holds the product tests, which use fakes and need no model or database.
- `code/examples/findyfixy/data/` holds the synthetic data the system reads.
- `code/examples/findyfixy/config/` holds committed settings that are not secrets: today the reference price per token of each model (`make obs-prices`).
- `code/examples/findyfixy/evals/` holds the frozen evaluation baseline and the attestation of the last passing gate run, which CI checks.
- `code/docs/` holds the architecture page and the decision records. `code/docs/README.md` lists them.
- `code/infra/` holds supporting services, each with its own compose file: today the Langfuse trace backend (`make obs-up`). `code/infra/README.md` explains it.
- `code/Makefile` holds the commands. Run `make help` from the folder above to list them.
- `code/CHANGELOG.md` records what was done in each step, what differs from the lessons, and what is open.
- `code/pyproject.toml` and `code/uv.lock` pin the dependencies.
