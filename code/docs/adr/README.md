# Architecture decision records

This project is a prototype that tests whether all 3 levels of AI-assisted engineering are reachable today with an open-source harness. It keeps changing. Each record here covers one phase of the build and shows the system as it stood when the phase closed.

A record has a decision, a diagnosis, guiding policies, coherent actions, a build-buy-adopt call, a staffing rule, what would change the decision, evidence and limits. The structure follows a standard engineering strategy shape.

Records are append-only. A later phase adds a record and never edits an earlier one. Records 0001 to 0005 were written on 2026-10-29, after their phases closed, from the day notes, the changelog and the decision log. From Day 6 each record is written in the commit that closes its phase.

Each diagram is drawn from the live architecture page by `make adr-snapshot`. A test fails if the page and the latest phase disagree.

| ADR | Phase | Decision | Code at close |
|---|---|---|---|
| [0001](0001-route-small-model-first.md) | Day 1 | Route on a small model first and distrust its confidence | `853532b` |
| [0002](0002-validate-outputs-and-tool-arguments.md) | Day 2 | Validate every model output and every tool argument | `5262b5b` |
| [0003](0003-incident-memory-in-postgres.md) | Day 3 | Store incident memory in Postgres with pgvector | `21b4c4a` |
| [0004](0004-hybrid-retrieval-with-reranker.md) | Day 4 | Retrieve with hybrid search and a reranker | `cd9283d` |
| [0005](0005-gate-releases-on-measured-answer-quality.md) | Day 5 | Gate releases on measured answer quality | `1e7f27a` |
