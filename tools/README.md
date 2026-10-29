# Tools

Scripts that serve and check the repository. Run them through `make ui` and `make check`, which supply the packages they need.

- `tools/serve.py`: the local tracker server.
- `tools/lessons.py`: renders a lesson's markdown for the tracker.
- `tools/check.py`: validates `record/progress.json`, runs the docs drift check and renders every lesson.
- `tools/check_docs.py`: fails when a README or `CLAUDE.md` names a file or make target that does not exist.
- `tools/scan_private.py`: blocks PII and secrets in commits, pushes and the working tree. The hooks in `.githooks/` call it.
- `tools/adr_snapshot.py`: draws the architecture at the close of each phase from `code/docs/architecture.html`, for the ADRs (`make adr-snapshot`).
- `tools/test_check_docs.py`, `tools/test_serve.py` and `tools/test_adr_snapshot.py`: tests for the checker, the server and the snapshot generator.
