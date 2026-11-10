# Docs

`code/docs/architecture.html` is the interactive architecture page, in two lanes: the harness on top (the product) and FindyFixy below (the app under test). It shows what is built and what is planned, and it follows a live run from the console's event stream: a triage run of one question, or an eval run of the first N golden cases, where each case moves through the app lane and its scores land on the eval gate. Start the console with `make console`, or run `make ui` for the tracker and the console together.

`code/docs/adr/` holds one architecture decision record per phase, each with a diagram, keyed by ADR number. The diagrams for ADRs 0001 to 0005 are frozen files checked by hash; from ADR 0006 on, `make adr-snapshot` draws them from the architecture page.

`code/docs/demo.gif` and `code/docs/space-needle.png` are the images used by the top-level README.
