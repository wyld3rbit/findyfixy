# infra

Services the harness and FindyFixy talk to, but that are part of neither. Each runs as containers from its own compose file, separate from `code/examples/findyfixy/docker-compose.yml`, so it can be stopped without touching the database.

## Langfuse: the trace backend

Langfuse v4 receives the harness's OpenTelemetry traces and shows them at `http://127.0.0.1:3000`. Six containers: the web app, a worker, Postgres (its own, not FindyFixy's), ClickHouse (traces), Valkey (queues) and MinIO (raw event files).

- `make obs-secrets`: writes `code/infra/langfuse/.env` once, with random secrets and a headless first start (org and project `findyfixy`, its API keys, the login `admin@example.com`). Mode 0600, git-ignored, never overwritten, because the data volumes keep the first passwords.
- `make obs-up`: starts the stack and waits until every container is healthy. It needs Colima at 8 GB.
- `make obs-down`: stops it and keeps its data.
- `make obs-prices`: registers `code/examples/findyfixy/config/prices.toml` with the running stack, so Langfuse prices every model call it receives from then on. Local models cost nothing per token; each gets the price of a hosted model of its size class, named in the table, as a yardstick for comparing runs. Traces stored before keep no cost. Safe to re-run.

`code/infra/langfuse/docker-compose.yml` is the official file of Langfuse v4.72.0 with a short list of changes at its top: images pinned by digest, the upstream `redis:7` replaced by Valkey (Redis 7.4 is RSALv2/SSPL only, not OSI-licensed), every published port on 127.0.0.1, telemetry off, no default secrets. Upgrade by diffing against the next release's file. `code/tests/test_langfuse_env.py` fails when the file requires a secret the generator does not write, or when a secret falls back to a default.

Image choices and their Grype scans (critical and high only) are recorded in `code/CHANGELOG.md` under Day 6, step 2. Accepted because everything is local, the data is synthetic, and only ports 3000 and 9090 on 127.0.0.1 are reachable. Rescan when a digest moves.
