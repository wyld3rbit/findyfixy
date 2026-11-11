# Harness tests

The tests of the `wyld3rness` harness. They drive it with the fakes in `code/src/wyld3rness/testing.py`, a judge, an embedder and a system under test that is not FindyFixy, so they need no model, database or network. `make test` runs them with the example app's tests. `make test-harness` runs them alone, in a throwaway environment that has no app installed, which proves the harness stands on its own.

- `code/tests/test_harness_rules.py` enforces the harness's architecture: it imports only the standard library, its declared dependencies and itself, has no import cycles or relative imports, and never works out a path from its own location.
- `code/tests/test_attest.py` tests the CI gate on temp trees, including a workspace with two source roots and the command line run alone with no site-packages.
- `code/tests/test_runner.py`, `code/tests/test_metrics.py`, `code/tests/test_judge.py`, `code/tests/test_report.py`, `code/tests/test_stats.py`, `code/tests/test_provenance.py` and `code/tests/test_cli.py` cover the eval runner, the metrics and their judge, the report and baseline rules, the statistics, the run's provenance and the command line.
- `code/tests/test_trace.py` tests the span tree and provider with an in-memory exporter, including that a hung or refusing backend does not slow the code that makes spans. `code/tests/test_eval_tracing.py` covers eval cases as traces with their scores.
- `code/tests/test_prices.py` covers the price table parser and its sync to Langfuse, and `code/tests/test_langfuse_env.py` checks the Langfuse secrets writer against its compose file.
