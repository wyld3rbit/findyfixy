# Evals

`code/evals/baseline.json` is the frozen baseline for the answer-quality gate: the scores of a chosen run, a floor for each measure and a cap on the judge failure rate. `make eval-gate` compares a new run against it, `make eval-freeze` writes a new one from a saved report, and `make eval-compare` checks two saved reports case by case. Reports are written to `code/reports/`, which is git-ignored.

The attestation, attestation.json in this folder, is the CI gate's record, written by `make eval-gate` when a full run on a clean tree passes. It holds the SHA-256 of every answer-affecting file, the report's name and hash, the verdict, and the models, judge digest and git SHA of the run. Commit it with the change it covers. The gate does not run when the database is stale (a `make ingest` would change it), because the run reads its passages from there, not from the file. `make verify-attestation` runs the same check as CI: it fails when any of those files differ from the attested ones. A failed or refused run writes nothing.

The gate, the baseline rules and the report format live in the harness (`code/harness/eval/report.py`). `code/app/eval_answers.py` runs FindyFixy through it. A baseline keeps every provenance key of its run except the per-run ones, so FindyFixy's models, threshold and recall depth are recorded in it.
