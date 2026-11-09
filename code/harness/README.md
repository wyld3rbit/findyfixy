# Harness

The `code/harness/` package is the evaluation harness: it scores an LLM system's answers and gates a release on the scores. It knows nothing about any one application. FindyFixy is its first consumer.

**The seam.** An application implements `SystemUnderTest` in `code/harness/eval/sut.py`: `describe()` returns the settings that shaped its answers, `answer(case)` returns an `Answer` (the text the judge scores, the contexts the model saw, the proposed action, the labels and app data), and `summarise(records)` returns any extra aggregates. Cases arrive as `Case` from `code/harness/eval/case.py`. FindyFixy's implementation is `code/app/eval_adapter.py`.

**The dependency rule.** An application may import the harness. Nothing under `code/harness/` imports an application, and no harness module works out a path from its own location: the app passes in its golden set, reports folder, baseline and repository directory. `code/tests/test_architecture.py` enforces both.

**Modules**
- `code/harness/eval/case.py` and `code/harness/eval/sut.py`: the case and the seam.
- `code/harness/eval/judge.py` and `code/harness/eval/metrics.py`: the judge over a local model, and the four metrics (faithfulness, response relevancy, context recall, action quality).
- `code/harness/eval/stats.py`: bootstrap intervals and aggregation. Each expected label `L` is scored as `L_match`.
- `code/harness/eval/provenance.py`: the stamp on every report. The app's `describe()` keys are merged in flat.
- `code/harness/eval/report.py`: report and baseline files, the gate, the table and the case viewer.
- `code/harness/eval/events.py`: the harness's own event vocabulary, which the runner reports to an optional viewer while a run is in flight.
- `code/harness/eval/runner.py`: runs every case through a system under test and builds the report.
- `code/harness/eval/cli.py`: the command line an app's entry point calls with its own wiring.

Its tests are `code/tests/test_harness_eval_*.py`. They use a fake system that is not FindyFixy.
