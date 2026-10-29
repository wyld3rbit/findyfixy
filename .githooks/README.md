# Git hooks

Two shell hooks that enforce the project's checks. Install them once per clone with the `hooks` target in `code/Makefile`.

- `.githooks/pre-commit` scans the staged content for PII and secrets with `tools/scan_private.py`, then runs `make check`, and `make test` when files under `code/` are staged.
- `.githooks/pre-push` scans every commit about to be pushed, including its metadata, and runs the checks again, which catches a commit made with `--no-verify`.
