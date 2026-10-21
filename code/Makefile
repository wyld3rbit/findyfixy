# Project tasks. Run from code/.
#
# audit: the two checks the supply-chain rules require after any change to
# uv.lock and before any commit that touches dependencies.
#   1. Known vulnerabilities in the locked set, without installing anything.
#   2. Startup hooks: a .pth file runs code every time Python starts, which is
#      how a hijacked package can persist. Only the known shims are allowed.
# On any finding, stop and read it. Do not upgrade or remove to make it pass.

.PHONY: audit

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
