# Code

FindyFixy, an event detection, determination and mitigation agent, and the evaluation harness around it. It runs on 1 machine with 16 GB of memory provisioned, with open-source models and synthetic data only.

- `code/app/` is the Python package. `code/app/README.md` maps its modules.
- `code/tests/` holds the product tests, which use fakes and need no model or database.
- `code/data/` holds the synthetic data the system reads.
- `code/evals/` holds the frozen evaluation baseline.
- `code/docs/` holds the architecture page.
- `code/Makefile` holds the commands. Run `make help` to list them.
- `code/CHANGELOG.md` records what was done in each step, what differs from the lessons, and what is open.
- `code/pyproject.toml` and `code/uv.lock` pin the dependencies.
