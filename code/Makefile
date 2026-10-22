# Project tasks. Run from code/.
#
# audit: the two checks the supply-chain rules require after any change to
# uv.lock and before any commit that touches dependencies.
#   1. Known vulnerabilities in the locked set, without installing anything.
#   2. Startup hooks: a .pth file runs code every time Python starts, which is
#      how a hijacked package can persist. Only the known shims are allowed.
# On any finding, stop and read it. Do not upgrade or remove to make it pass.

.PHONY: audit hooks scan up down

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
