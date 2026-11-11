# Code

A uv workspace: the wyld3rness harness at the root, and the example app it is tested on under `code/examples/`. One `code/uv.lock` pins every dependency of both, and one `code/.venv` holds them (`uv sync --all-packages`).

- `code/src/wyld3rness/` is the harness package: evaluation (`code/src/wyld3rness/eval/`), tracing (`code/src/wyld3rness/trace/`) and test doubles for apps (`code/src/wyld3rness/testing.py`). It never imports an app. `code/src/wyld3rness/README.md` explains the seam.
- `code/tests/` holds the harness's tests. `make test-harness` runs them with no app installed.
- `code/examples/findyfixy/` is FindyFixy: its package, tests, synthetic data, baseline and attestation, price table and database compose file. `code/examples/findyfixy/README.md` maps it.
- `code/infra/` holds the services the harness talks to, each with its own compose file: today the Langfuse trace backend (`make obs-up`). `code/infra/README.md` explains it.
- `code/docs/` holds the decision records and the README images. `code/docs/README.md` lists them.
- `code/pyproject.toml` declares the harness, the workspace, and the settings for pytest, Ruff and mypy.
- `code/Makefile` holds the commands. Run `make help` from the folder above to list them.
- `code/CHANGELOG.md` records what was done in each step, what differs from the lessons, and what is open.
