# Changelog

Notable changes to wyld3rness and its example app. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions will follow [Semantic Versioning](https://semver.org/) once releases begin. Each entry links the decision record that explains it.

## Unreleased

### Harness

- **Eval gate.** Runs an app on golden cases with known answers. Labels such as severity are matched exactly, and a judge model scores faithfulness, response relevancy, context recall and action quality. Each run is compared with a frozen baseline, with seeded bootstrap intervals, and fails on a measured regression or a judge failure rate above its cap. A negative control can never become a baseline. ([ADR 0005](docs/adr/0005-gate-releases-on-measured-answer-quality.md))
- **One interface for any app.** `SystemUnderTest` (`describe`, `answer`, `summarise`) separates the harness from the app. A test fails if the harness imports anything but the standard library, its declared dependencies and itself. ([ADR 0006](docs/adr/0006-eval-gate-as-harness-component-attested-in-ci.md))
- **Attested CI gate.** A full, clean, passing run writes an attestation: the hash of every answer-affecting file, found from the eval's import closure plus declared data, lock and compose files. CI recomputes the hashes in its own job and fails a change no passing run covers. ([ADR 0006](docs/adr/0006-eval-gate-as-harness-component-attested-in-ci.md))
- **Tracing.** Every run, eval case and command is an OpenTelemetry trace in a self-hosted Langfuse, with a span per step, eval scores on their case, and a cost on each model call from a committed reference price table. Export is batched on its own thread, so a slow or stopped backend does not slow a run. ([ADR 0007](docs/adr/0007-every-run-is-a-trace-in-a-self-hosted-backend.md))

### Example app: FindyFixy

- **Small model first.** A small model answers, and a larger one takes the case when the answer is invalid after one retry or its confidence is low. ([ADR 0001](docs/adr/0001-route-small-model-first.md))
- **Output contract and tools.** Every answer is a validated record with one retry that carries the validation error. Tool arguments are validated inside the tool, and the deploy tool runs both directly and over MCP. ([ADR 0002](docs/adr/0002-validate-outputs-and-tool-arguments.md))
- **Incident memory.** Synthetic past incidents are chunked, embedded and stored in Postgres with pgvector by an ingestion job that is safe to re-run. ([ADR 0003](docs/adr/0003-incident-memory-in-postgres.md))
- **Hybrid retrieval.** Keyword and vector search fused by reciprocal rank, then a cross-encoder reranker, returning incidents rather than chunks. ([ADR 0004](docs/adr/0004-hybrid-retrieval-with-reranker.md))
- **Console.** A local page that runs a triage or an eval and shows each step on the live architecture diagram.

### Tooling and security

- Locked dependencies with a 7-day release cooldown, `make audit` for known vulnerabilities and unexpected startup hooks, and container images pinned by digest and scanned with Grype before first use.
- A privacy scanner in the pre-commit and pre-push hooks blocks personal data and secrets in files and commit metadata.
- Ruff and mypy in the hooks and in CI. CI runs with a read-only token and no secrets, with actions pinned by commit.
- `make check` covers documentation drift, the repository tools' tests and the privacy scan.

### Known issues

- Linux installs pull PyTorch's GPU libraries although nothing uses a GPU, which slows CI. CPU-only wheels are planned with the service image.
- The model names come from an untracked `.env` file, so a model swap changes answers without failing CI. Only the judge is pinned, through the baseline.
- The stale-database check catches a changed embedding model name, not a new revision of the same model.
- The golden set is small and synthetic, and it has not yet exercised escalation.

## History

The step-by-step build log up to 2026-11-13, including the image scans and exceptions that earlier records cite, is preserved at [this commit](https://github.com/wyld3rbit/wyld3rness/blob/19f5c7a73ff58bfb98ad0afcb98374b1e8c360bd/code/CHANGELOG.md).
