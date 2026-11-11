# Project tasks. Run from code/, the workspace root: the wyld3rness harness
# (src/, tests/, infra/) and the example app it is tested on (examples/findyfixy/).
#
# audit: the two checks the supply-chain rules require after any change to
# uv.lock and before any commit that touches dependencies.
#   1. Known vulnerabilities in the locked set, without installing anything.
#   2. Startup hooks: a .pth file runs code every time Python starts, which is
#      how a hijacked package can persist. Only the known shims are allowed.
# On any finding, stop and read it. Do not upgrade or remove to make it pass.

.PHONY: audit hooks scan up down setup ingest console triage test test-harness eval-retrieval eval eval-gate eval-baseline eval-freeze eval-compare eval-negative case golden verify-attestation obs-secrets obs-up obs-down obs-prices

# Memory for the Colima VM, in GB. 6 on most days, 8 on Langfuse days:
#   make up COLIMA_MEMORY=8
# Changing it needs the VM to restart, so run `make down` first.
COLIMA_MEMORY ?= 6

# The example app's folder, its database services and its settings file.
APP_DIR := examples/findyfixy
APP_COMPOSE := $(APP_DIR)/docker-compose.yml
APP_ENV := $(APP_DIR)/.env
# Every workspace member installed: the harness and the example app.
RUN := uv run --all-packages

# up: start the container runtime if it is off, then the services, and wait
# until they are ready. Safe to run twice. `--wait` blocks until the database
# healthcheck passes, so the next command never meets a database that is still
# starting.
up:
	@colima status >/dev/null 2>&1 || colima start --memory $(COLIMA_MEMORY)
	docker compose -f $(APP_COMPOSE) up -d --wait

# down: stop the services, then the VM, and check that the VM really stopped.
# Colima has no idle timeout and closing the terminal does not stop it, so it
# keeps its RAM until told otherwise. Data survives: it lives in the `pgdata`
# volume, which only `docker compose down -v` deletes. Safe to run twice.
# The compose step runs only while the VM is up. If it fails then, the make
# stops there, so a half-stopped system is never reported as clean.
down:
	@if colima status >/dev/null 2>&1; then docker compose -f $(APP_COMPOSE) down; fi
	colima stop
	@if colima status >/dev/null 2>&1; then echo "FAILED: Colima is still running"; exit 1; fi
	@echo "Colima is stopped."

audit:
	uv export --frozen --all-packages --format requirements-txt --no-emit-workspace -o .audit-req.txt > /dev/null
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
# named in examples/findyfixy/.env (answering models and the judge, each once), starts the database
# and loads the incidents. It stops early, and says why, if .env is missing or
# does not name both models. It never writes .env: that file is yours. Every step is safe to run again: uv sync and
# ollama pull skip what is current, and ingest writes 0 when nothing changed.
setup:
	@test -f $(APP_ENV) || { echo "code/$(APP_ENV) is missing. Copy it: cp $(APP_DIR)/.env.example $(APP_ENV)"; exit 1; }
	@grep -q '^SMALL_MODEL=.' $(APP_ENV) && grep -q '^LARGE_MODEL=.' $(APP_ENV) || { echo "code/$(APP_ENV) must set SMALL_MODEL and LARGE_MODEL"; exit 1; }
	uv sync --all-packages
	@set -a; . ./$(APP_ENV); set +a; \
	  for m in "$$SMALL_MODEL" "$$LARGE_MODEL" "$${JUDGE_MODEL:-qwen2.5:7b}"; do echo "$$m"; done | sort -u | while read -r m; do ollama pull "$$m" || exit 1; done
	$(MAKE) up
	$(MAKE) ingest

# The commands below only name what already exists, so there is one short word
# for each. Run them from code/ or, through the root Makefile, from the folder above.
ingest:
	$(RUN) findyfixy-ingest

console:
	$(RUN) findyfixy-console

triage:
	$(RUN) findyfixy-triage

test:
	$(RUN) pytest

# test-harness: the harness's own suite in a throwaway environment that holds
# the harness and its dev tools only, no app. It proves the harness stands
# alone: an import of FindyFixy, or of a library only FindyFixy installs, fails
# here even if a test missed it.
test-harness:
	uv run --isolated --package wyld3rness pytest tests

eval-retrieval:
	$(RUN) findyfixy-eval-retrieval

# Answer-quality evaluation: findyfixy/evals/answers.py runs FindyFixy through the
# harness in wyld3rness/eval/. Needs make up and Ollama. Each run takes many
# minutes. Reports land in code/examples/findyfixy/reports/ (git-ignored).
#   eval          run and print the scores, no gate
#   eval-gate     the release gate: exit 1 if answers got worse vs examples/findyfixy/evals/baseline.json.
#                 A pass of the full set on a clean tree also writes
#                 examples/findyfixy/evals/attestation.json. Commit it: CI checks it. It does
#                 not run on a stale database: run make ingest first.
#   eval-baseline freeze this run as the new baseline. Review the diff before committing.
#   eval-negative the proof the gate works: run WITHOUT recall; it must fail
eval:
	$(RUN) findyfixy-eval

eval-gate:
	$(RUN) findyfixy-eval --gate

eval-baseline:
	$(RUN) findyfixy-eval --write-baseline

eval-negative:
	$(RUN) findyfixy-eval --no-recall --gate

# verify-attestation: what CI runs. Rehashes the answer-affecting files and
# exits 1 unless the attestation is a pass over exactly these files.
# Plain python3 and the standard library: no uv sync, and no file it hashes
# is imported. -I keeps the environment and the script's folder off the
# import path. The workflow runs the same command from the training root.
verify-attestation:
	python3 -I src/wyld3rness/eval/attest.py verify --root . --attestation examples/findyfixy/evals/attestation.json

# golden: print the 30 golden cases (alert, expected severity, reference) to read
# and check by eye. Read-only.
golden:
	$(RUN) findyfixy-golden --show

# case: print one case from the newest report: the alert, what the model and the
# judge saw, the answer and every judge decision. For checking a score by eye.
#   make case ID=g07
#   make case ID=g07 REPORT=examples/findyfixy/reports/eval-<timestamp>.json   (a report other than the newest)
case:
	@test -n "$(ID)" || { echo "usage: make case ID=g07 [REPORT=$(APP_DIR)/reports/eval-<timestamp>.json]"; exit 2; }
	$(RUN) findyfixy-eval --show-case $(ID) $(if $(REPORT),--report $(REPORT))

# eval-compare: compare a SAVED report with the newest one (or OTHER), case by
# case: answers, contexts, labels, every metric score. Exit 1 on any difference.
# The proof that a change which must not move a score did not. No run.
#   make eval-compare REPORT=examples/findyfixy/reports/eval-<timestamp>.json
#   make eval-compare REPORT=examples/findyfixy/reports/eval-<a>.json OTHER=examples/findyfixy/reports/eval-<b>.json
eval-compare:
	@test -n "$(REPORT)" || { echo "usage: make eval-compare REPORT=$(APP_DIR)/reports/eval-<timestamp>.json [OTHER=$(APP_DIR)/reports/eval-<timestamp>.json]"; exit 2; }
	$(RUN) findyfixy-eval --compare $(REPORT) $(if $(OTHER),--report $(OTHER))

# eval-freeze: freeze a SAVED report as the baseline (no run). The baseline is then
# exactly the numbers you reviewed. Refuses a dirty-tree run, run errors, a high
# judge failure rate and a no-recall run.
#   make eval-freeze REPORT=examples/findyfixy/reports/eval-<timestamp>.json
eval-freeze:
	@test -n "$(REPORT)" || { echo "usage: make eval-freeze REPORT=$(APP_DIR)/reports/eval-<timestamp>.json"; exit 2; }
	$(RUN) findyfixy-eval --baseline-from $(REPORT)

# obs-*: the local Langfuse v4 stack that receives traces (infra/langfuse/).
# It needs Colima at 8 GB: COLIMA_MEMORY=8 make up, or restart Colima with
# --memory 8. obs-secrets writes infra/langfuse/.env once (random secrets,
# mode 0600, never overwritten: the volumes keep the first passwords).
# obs-up refuses to start without it. obs-down stops the stack and keeps
# its data; `docker compose -f $(LANGFUSE) down -v` deletes the data too.
LANGFUSE := infra/langfuse/docker-compose.yml

obs-secrets:
	uv run python infra/langfuse/new_env.py

obs-up:
	@test -f infra/langfuse/.env || { echo "no infra/langfuse/.env: run make obs-secrets first"; exit 2; }
	docker compose -f $(LANGFUSE) --env-file infra/langfuse/.env up -d --wait
	@echo "Langfuse: http://127.0.0.1:3000  (login and password: infra/langfuse/.env)"

obs-down:
	docker compose -f $(LANGFUSE) --env-file infra/langfuse/.env down

# obs-prices: register examples/findyfixy/config/prices.toml (reference prices per token) with
# the running Langfuse, which then prices every model call it receives.
# Safe to re-run: unchanged prices are left alone.
obs-prices:
	$(RUN) findyfixy-prices
