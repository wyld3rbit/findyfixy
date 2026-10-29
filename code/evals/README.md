# Evals

`code/evals/baseline.json` is the frozen baseline for the answer-quality gate: the scores of a chosen run, a floor for each measure and a cap on the judge failure rate. `make eval-gate` compares a new run against it, and `make eval-freeze` writes a new one from a saved report. Reports are written to `code/reports/`, which is git-ignored.
