# Tools

Scripts that check the repository. Run them through `make check`, which supplies the packages they need.

- `tools/check_docs.py`: fails when a README names a file or make target that does not exist.
- `tools/scan_private.py`: blocks PII and secrets in commits, pushes and the working tree. The hooks in `.githooks/` call it.
- `tools/adr_snapshot.py`: draws the architecture at the close of each phase from `code/examples/findyfixy/src/findyfixy/console/static/architecture.html`, for the ADRs (`make adr-snapshot`). Pictures drawn from an older page layout are frozen and checked by hash.
- `tools/test_check_docs.py`, `tools/test_scan_private.py` and `tools/test_adr_snapshot.py`: tests for the docs checker, the privacy scanner and the snapshot generator.
