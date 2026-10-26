# Everything you run, from the training folder (the one that holds CLAUDE.md).
#   make help
#
# The product lives in code/ and has its own Makefile. The targets here either
# hand off to it or run the tracker tools in tools/. Nothing here does work of
# its own, so the real commands stay readable in one place each.

.DEFAULT_GOAL := help
.PHONY: help setup up down console triage test eval-retrieval eval eval-gate eval-baseline eval-freeze eval-negative case golden ingest audit tracker ui check

help:
	@echo "First time:       make setup"
	@echo "Start / stop:     make up   make down"
	@echo "Run the product:  make console   make triage   make ingest"
	@echo "Measure / test:   make test   make eval-retrieval"
	@echo "Answer quality:   make eval   make eval-gate   make eval-baseline   make eval-negative"
	@echo "See test cases:   make golden   make case ID=g07"
	@echo "Freeze baseline:  make eval-freeze REPORT=reports/eval-<timestamp>.json"
	@echo "Tracker page:     make tracker"
	@echo "Landing page:     make ui   (tracker with the visualization embedded)"
	@echo "Before a commit:  make check   make audit"

# Product targets: see code/Makefile for what each one runs.
setup up down console triage test eval-retrieval eval eval-gate eval-baseline eval-negative golden ingest audit:
	$(MAKE) -C code $@

# Targets that take a value are forwarded explicitly, because the pattern rule
# above cannot pass variables to the code Makefile.
case:
	$(MAKE) -C code case ID=$(ID)

eval-freeze:
	$(MAKE) -C code eval-freeze REPORT=$(REPORT)

# The tracker page at http://127.0.0.1:8766. Ctrl-C stops it.
tracker:
	uv run --no-project --with markdown python tools/serve.py

# ui: the landing page. The tracker (8766) embeds the visualization (8765), so one
# address is enough; both servers run in one terminal. Each runs
# in the background and `wait` holds the recipe open. The trap turns Ctrl-C into
# a stop for the whole process group, so neither server is left running.
# It needs make up first for live runs; fake mode and the tracker do not.
ui:
	@echo "Open http://127.0.0.1:8766  (the visualization is embedded; also at :8765).  Ctrl-C stops both."
	@trap 'kill 0' INT TERM; \
	(cd code && uv run python -m app.console) & \
	uv run --no-project --with markdown python tools/serve.py & \
	wait

# check: the tracker data, the tracker server, and the privacy scan of the tree.
check:
	uv run --no-project --with markdown python tools/check.py
	uv run --no-project --with markdown --with pytest pytest tools/test_serve.py -q
	python3 tools/scan_private.py --tree
