# Everything you run, from the repository root.
#   make help
#
# The product lives in code/ (the wyld3rness harness, with FindyFixy under
# code/examples/findyfixy/) and has its own Makefile. The targets here either
# hand off to it or run the repository checks in tools/. Nothing here does work of
# its own, so the real commands stay readable in one place each.

.DEFAULT_GOAL := help
.PHONY: help adr-snapshot setup up down console triage test test-harness lint eval-retrieval eval eval-gate eval-baseline eval-freeze eval-compare eval-negative obs-secrets obs-up obs-down obs-prices case golden ingest audit verify-attestation check

help:
	@echo "First time:       make setup"
	@echo "Start / stop:     make up   make down"
	@echo "Run the product:  make console   make triage   make ingest"
	@echo "Measure / test:   make test   make test-harness   make lint   make eval-retrieval"
	@echo "Answer quality:   make eval   make eval-gate   make eval-baseline   make eval-negative"
	@echo "CI gate:          make verify-attestation   (the attestation eval-gate writes on a pass)"
	@echo "See test cases:   make golden   make case ID=g07"
	@echo "Freeze baseline:  make eval-freeze REPORT=examples/findyfixy/reports/eval-<timestamp>.json"
	@echo "Compare runs:     make eval-compare REPORT=examples/findyfixy/reports/eval-<timestamp>.json"
	@echo "Trace backend:    make obs-secrets   make obs-up   make obs-down   make obs-prices   (Langfuse on 127.0.0.1:3000)"
	@echo "Decision records:  make adr-snapshot   (redraw the diagram in each ADR)"
	@echo "Before a commit:  make check   make audit"

# Product targets: see code/Makefile for what each one runs.
setup up down console triage test test-harness lint eval-retrieval eval eval-gate eval-baseline eval-negative golden ingest audit verify-attestation:
	$(MAKE) -C code $@

# Targets that take a value are forwarded explicitly, because the pattern rule
# above cannot pass variables to the code Makefile.
case:
	$(MAKE) -C code case ID=$(ID) REPORT=$(REPORT)

eval-freeze:
	$(MAKE) -C code eval-freeze REPORT=$(REPORT)

eval-compare:
	$(MAKE) -C code eval-compare REPORT=$(REPORT) OTHER=$(OTHER)

# One architecture picture per phase for the ADRs in code/docs/adr/, drawn from the live page.
adr-snapshot:
	python3 tools/adr_snapshot.py

# check: the documentation drift check, the repository tools' tests and the
# privacy scan of the tree.
#
# uv tries its local cache first (--offline) and goes online only on a cache
# miss (exit 2), so a flaky connection never refuses a commit that has nothing
# wrong with it. Exit 1 from the tool itself is a real failure and is passed through.
UV_TOOLS = uv run --no-project
check:
	python3 tools/check_docs.py
	@$(UV_TOOLS) --offline --with pytest pytest tools -q; rc=$$?; \
	  if [ $$rc -eq 2 ]; then echo "(cache miss, retrying online)"; $(UV_TOOLS) --with pytest pytest tools -q; else exit $$rc; fi
	python3 tools/scan_private.py --tree

obs-secrets obs-up obs-down obs-prices:
	$(MAKE) -C code $@
