# Evals

`code/evals/baseline.json` is the frozen baseline for the answer-quality gate: the scores of a chosen run, a floor for each measure and a cap on the judge failure rate. `make eval-gate` compares a new run against it, and `make eval-freeze` writes a new one from a saved report. Reports are written to `code/reports/`, which is git-ignored.

The gate, the baseline rules and the report format live in the harness (`code/harness/eval/report.py`). `code/app/eval_answers.py` runs FindyFixy through it. A baseline keeps every provenance key of its run except the per-run ones, so FindyFixy's models, threshold and recall depth are recorded in it.
