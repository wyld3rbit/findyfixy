# Harness

The `code/harness/` package is the evaluation harness: it scores an LLM system's answers and gates a release on the scores. It knows nothing about any one application. FindyFixy is its first consumer.

**The seam.** An application implements `SystemUnderTest` in `code/harness/eval/sut.py`: `describe()` returns the settings that shaped its answers, `answer(case)` returns an `Answer` (the text the judge scores, the contexts the model saw, the proposed action, the labels and app data), and `summarise(records)` returns any extra aggregates. Cases arrive as `Case` from `code/harness/eval/case.py`. FindyFixy's implementation is `code/app/eval_adapter.py`.

**The dependency rule.** An application may import the harness. Nothing under `code/harness/` imports an application, and no harness module works out a path from its own location: the app passes in its golden set, reports folder, baseline and repository directory. `code/tests/test_architecture.py` enforces both.

**Modules**
- `code/harness/eval/case.py` and `code/harness/eval/sut.py`: the case and the seam.
- `code/harness/eval/judge.py` and `code/harness/eval/metrics.py`: the judge over a local model, and the four metrics (faithfulness, response relevancy, context recall, action quality).
- `code/harness/eval/stats.py`: bootstrap intervals and aggregation. Each expected label `L` is scored as `L_match`.
- `code/harness/eval/provenance.py`: the stamp on every report. The app's `describe()` keys are merged in flat.
- `code/harness/eval/report.py`: report and baseline files, the gate, the table, the case viewer, and the case-by-case comparison of two saved reports (`--compare`).
- `code/harness/eval/events.py`: the harness's own event vocabulary, which the runner reports to an optional viewer while a run is in flight.
- `code/harness/eval/runner.py`: runs every case through a system under test and builds the report, in two passes: every case is answered, then every answer is judged, so a machine that holds one model at a time loads each model about once. The report's `timing` has each pass (`answer_s`, `judge_s`) and the wall time.
- `code/harness/eval/cli.py`: the command line an app's entry point calls with its own wiring. With an `AttestationSpec`, a passing `--gate` run also writes the attestation.
- `code/harness/eval/attest.py`: the CI gate. It fingerprints every answer-affecting file, writes the attestation after a passing full run on a clean tree, and verifies it in CI. It also holds the traceability rules the baseline shares. It is the harness's leaf: standard library only and no harness imports, so CI runs it alone with plain `python3` and no install.

**The CI gate.** A live run needs a GPU, so it runs on the developer's machine and CI checks its attestation. The answer-affecting files are computed, not listed: the import closure of the app's eval entry modules, read from the source with `ast`, plus the data and lock files the app declares (a declared directory, such as `code/data/` for FindyFixy, counts as every file in it). A file read at runtime is invisible to the closure, which is why the app declares those. State that no file shows, such as a database loaded from an older corpus, is the app's precondition: the gate checks it before the run and again before writing. The attestation is written only when the whole command exits 0. CI recomputes the hashes and fails when a file changed, was added or was removed since the attested run, and names each one. This is a process control, not a security control: CI trusts the local run, and anyone who can push can edit the attestation. Review and branch protection stop a dishonest change; the attestation stops an honest one from merging without a run.

Its tests are `code/tests/test_harness_eval_*.py`. They use a fake system that is not FindyFixy.
