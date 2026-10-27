# Project tasks. Run from code/.
#
# audit: the two checks the supply-chain rules require after any change to
# uv.lock and before any commit that touches dependencies.
#   1. Known vulnerabilities in the locked set, without installing anything.
#   2. Startup hooks: a .pth file runs code every time Python starts, which is
#      how a hijacked package can persist. Only the known shims are allowed.
# On any finding, stop and read it. Do not upgrade or remove to make it pass.

.PHONY: audit hooks scan up down setup ingest console triage test eval-retrieval eval eval-gate eval-baseline eval-freeze eval-negative case golden

# Memory for the Colima VM, in GB. 6 on most days, 8 on Langfuse days:
#   make up COLIMA_MEMORY=8
# Changing it needs the VM to restart, so run `make down` first.
COLIMA_MEMORY ?= 6

# up: start the container runtime if it is off, then the services, and wait
# until they are ready. Safe to run twice. `--wait` blocks until the database
# healthcheck passes, so the next command never meets a database that is still
# starting.
up:
	@colima status >/dev/null 2>&1 || colima start --memory $(COLIMA_MEMORY)
	docker compose up -d --wait

# down: stop the services, then the VM, and check that the VM really stopped.
# Colima has no idle timeout and closing the terminal does not stop it, so it
# keeps its RAM until told otherwise. Data survives: it lives in the `pgdata`
# volume, which only `docker compose down -v` deletes. Safe to run twice.
# The compose step runs only while the VM is up. If it fails then, the make
# stops there, so a half-stopped system is never reported as clean.
down:
	@if colima status >/dev/null 2>&1; then docker compose down; fi
	colima stop
	@if colima status >/dev/null 2>&1; then echo "FAILED: Colima is still running"; exit 1; fi
	@echo "Colima is stopped."

audit:
	uv export --frozen --format requirements-txt --no-emit-project -o .audit-req.txt > /dev/null
	uvx pip-audit --disable-pip --require-hashes -r .audit-req.txt
	@echo "--- startup hooks (.pth files in .venv):"
	@bad=$$(find .venv -name '*.pth' \
	    ! -name '_virtualenv.pth' \
	    ! -name 'distutils-precedence.pth' \
	    ! -name '__editable__*.pth'); \
	if [ -n "$$bad" ]; then \
	    echo "UNEXPECTED .pth FILES, show their contents before going on:"; \
	    echo "$$bad"; exit 1; \
	fi; \
	echo "none beyond the allowed shims"

# hooks: install the privacy hooks for THIS clone (git hook settings are not
# part of a clone). They block a commit or push that carries PII or secrets.
# See tools/scan_private.py. Run once after cloning.
hooks:
	git config core.hooksPath .githooks
	@echo "privacy hooks installed (pre-commit and pre-push)"

# scan: check everything on disk and in history, on demand. Prints rule, file
# and line, never the matched text.
scan:
	uv run --no-project python ../tools/scan_private.py --tree
	uv run --no-project python ../tools/scan_private.py --history

# setup: once per clone. Installs the locked packages into .venv, pulls the models
# named in .env (answering models and the judge, each once), starts the database
# and loads the incidents. It stops early, and says why, if .env is missing or
# does not name both models. It never writes .env: that file is yours. Every step is safe to run again: uv sync and
# ollama pull skip what is current, and ingest writes 0 when nothing changed.
setup:
	@test -f .env || { echo "code/.env is missing. Copy it: cp .env.example .env"; exit 1; }
	@grep -q '^SMALL_MODEL=.' .env && grep -q '^LARGE_MODEL=.' .env || { echo "code/.env must set SMALL_MODEL and LARGE_MODEL"; exit 1; }
	uv sync
	@set -a; . ./.env; set +a; \
	  for m in "$$SMALL_MODEL" "$$LARGE_MODEL" "$${JUDGE_MODEL:-qwen2.5:7b}"; do echo "$$m"; done | sort -u | while read -r m; do ollama pull "$$m" || exit 1; done
	$(MAKE) up
	$(MAKE) ingest

# The commands below only name what already exists, so there is one short word
# for each. Run them from code/ or, through the root Makefile, from the folder above.
ingest:
	uv run python -m app.ingest

console:
	uv run python -m app.console

triage:
	uv run python -m app.graph

test:
	uv run pytest

eval-retrieval:
	uv run python -m app.eval_retrieval

# Answer-quality evaluation (app/eval_answers.py). Needs make up and Ollama. Each
# run takes many minutes. Reports land in code/reports/ (git-ignored).
#   eval          run and print the scores, no gate
#   eval-gate     the release gate: exit 1 if answers got worse vs evals/baseline.json
#   eval-baseline freeze this run as the new baseline. Review the diff before committing.
#   eval-negative the proof the gate works: run WITHOUT recall; it must fail
eval:
	uv run python -m app.eval_answers

eval-gate:
	uv run python -m app.eval_answers --gate

eval-baseline:
	uv run python -m app.eval_answers --write-baseline

eval-negative:
	uv run python -m app.eval_answers --no-recall --gate

# golden: print the 30 golden cases (alert, expected severity, reference) to read
# and check by eye. Read-only.
golden:
	uv run python -m app.golden --show

# case: print one case from the newest report: the alert, what the model and the
# judge saw, the answer and every judge decision. For checking a score by eye.
#   make case ID=g07
case:
	@test -n "$(ID)" || { echo "usage: make case ID=g07"; exit 2; }
	uv run python -m app.eval_answers --show-case $(ID)

# eval-freeze: freeze a SAVED report as the baseline (no run). The baseline is then
# exactly the numbers you reviewed. Refuses a dirty-tree run, run errors, a high
# judge failure rate and a no-recall run.
#   make eval-freeze REPORT=reports/eval-<timestamp>.json
eval-freeze:
	@test -n "$(REPORT)" || { echo "usage: make eval-freeze REPORT=reports/eval-<timestamp>.json"; exit 2; }
	uv run python -m app.eval_answers --baseline-from $(REPORT)
