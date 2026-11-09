# Tests

The product tests, next to the behaviour they cover. They use fakes, so they need no model, database or network, and they run with `make test`.

- `code/tests/test_architecture.py` enforces the dependency rule: domain modules do not import vendor clients or the composition root, and the harness never imports the app.
- `code/tests/test_harness_eval_*.py` test the harness with a fake system under test from `code/tests/harness_fakes.py`. FindyFixy's adapter is tested in `code/tests/test_eval_adapter.py`, and its pre-harness reports in `code/tests/test_eval_legacy_reports.py`.
- `code/tests/integration/` needs a live model and database, so `make test` leaves it out.
- `code/tests/test_page_truth.py` keeps the architecture page honest: each box's `data-code` must match the code (built means it exists, planned means it does not), and every event kind the code can emit must have a handler on the page.
- `code/tests/js/replay.js` plays events through the page's own script, for `code/tests/test_page_lag.py` and `code/tests/test_page_truth.py`.
