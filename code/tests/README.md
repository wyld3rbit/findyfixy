# Tests

The product tests, next to the behaviour they cover. They use fakes, so they need no model, database or network, and they run with `make test`.

- `code/tests/test_architecture.py` enforces the dependency rule: domain modules do not import vendor clients or the composition root.
- `code/tests/integration/` needs a live model and database, so `make test` leaves it out.
- `code/tests/js/replay.js` is a helper for the page-lag test in `code/tests/test_page_lag.py`.
