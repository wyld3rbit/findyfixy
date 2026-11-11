"""Where things live on disk, defined once.

Why: ten modules used to compute the project root for themselves from their own
file location. That works, but a change (a moved module, a renamed folder) then
has to be made ten times, and two of them can disagree. One definition removes
the drift. A test fails if any other module computes the root again.

All paths are resolved from this file, never from the current directory, so
every command works from any clone and any working directory.

The package uses a `src/` layout inside a uv workspace:
    code/                            WORKSPACE: the harness, the lock, infra/
    code/examples/findyfixy/         ROOT: this app's data, evals, config, reports
    code/examples/findyfixy/src/findyfixy/   this package
The editable install keeps `__file__` inside the checkout, which is what these
paths assume. A wheel installed elsewhere would carry no data folder, and this
app is not shipped as one.

This module does no work at import time beyond building the path constants.
"""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]  # code/examples/findyfixy/
WORKSPACE = ROOT.parents[1]  # code/

DATA = ROOT / "data"  # committed inputs: the corpus, the golden set, the demo bank
EVALS = ROOT / "evals"  # committed evaluation baseline and attestation
REPORTS = ROOT / "reports"  # generated evaluation reports (git-ignored)
PRICES = ROOT / "config" / "prices.toml"  # committed reference prices per token (make obs-prices)
PAGE = PACKAGE / "console" / "static" / "architecture.html"  # the page the console serves
LANGFUSE_ENV = WORKSPACE / "infra" / "langfuse" / ".env"  # the local trace backend's secrets (git-ignored)
