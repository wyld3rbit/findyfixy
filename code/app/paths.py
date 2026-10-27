"""Where things live on disk, defined once.

Why: ten modules used to compute the project root for themselves from their own
file location. That works, but a change (a moved module, a renamed folder) then
has to be made ten times, and two of them can disagree. One definition removes
the drift. A test fails if any other module computes the root again.

All paths are resolved from this file, never from the current directory, so
every command works from any clone and any working directory.

This module does no work at import time beyond building the path constants.
"""

from pathlib import Path

# code/ : the folder that holds `app/`.
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"  # committed inputs: the corpus, the golden set, the demo bank
DOCS = ROOT / "docs"  # the architecture page the console serves
EVALS = ROOT / "evals"  # committed evaluation baseline
REPORTS = ROOT / "reports"  # generated evaluation reports (git-ignored)
