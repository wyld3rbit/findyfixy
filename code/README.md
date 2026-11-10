# Code

FindyFixy, an event detection, determination and mitigation agent, and the evaluation harness around it. It runs on 1 machine with 16 GB of memory provisioned, with open-source models and synthetic data only.

- `code/app/` is the FindyFixy package. `code/app/README.md` maps its modules.
- `code/harness/` is the evaluation harness. It never imports the app. `code/harness/README.md` explains the seam.
- `code/tests/` holds the product tests, which use fakes and need no model or database.
- `code/data/` holds the synthetic data the system reads.
- `code/evals/` holds the frozen evaluation baseline and the attestation of the last passing gate run, which CI checks.
- `code/docs/` holds the architecture page.
- `code/infra/` holds supporting services, each with its own compose file: today the Langfuse trace backend (`make obs-up`). `code/infra/README.md` explains it.
- `code/Makefile` holds the commands. Run `make help` to list them.
- `code/CHANGELOG.md` records what was done in each step, what differs from the lessons, and what is open.
- `code/pyproject.toml` and `code/uv.lock` pin the dependencies.
