# FindyFixy tests

The example app's tests, next to the behaviour they cover. They use fakes, so they need no model, database or network, and they run with `make test`. The harness's own tests are in `code/tests/`.

- `code/examples/findyfixy/tests/test_architecture.py` enforces the dependency rule: domain modules do not import vendor clients or the composition root, and only `code/examples/findyfixy/src/findyfixy/paths.py` works out where files live.
- `code/examples/findyfixy/tests/test_eval_adapter.py` tests FindyFixy as a system under test, with the harness's fakes from `code/src/wyld3rness/testing.py`, and `code/examples/findyfixy/tests/test_eval_legacy_reports.py` its pre-harness reports.
- `code/examples/findyfixy/tests/test_eval_attestation.py` checks what FindyFixy declares to the CI gate: the eval path holds no console module, every data file it reads is declared, and the Makefile and the workflow verify the file the gate writes.
- `code/examples/findyfixy/tests/test_trace_adapter.py` runs every fake console scenario through a traced console and checks the tree. `code/examples/findyfixy/tests/test_observe_everything.py` covers command traces, `user.id` on top spans, hunt sessions, the console's request traces, a tool path run and an MCP tool call.
- `code/examples/findyfixy/tests/test_page_truth.py` keeps the architecture page honest: each box's `data-code` must match the code (built means it exists, planned means it does not), and every event kind the code can emit must have a handler on the page.
- `code/examples/findyfixy/tests/js/replay.js` plays events through the page's own script, for `code/examples/findyfixy/tests/test_page_lag.py` and `code/examples/findyfixy/tests/test_page_truth.py`.
- `code/examples/findyfixy/tests/integration/` needs a live model and database, so `make test` leaves it out.
