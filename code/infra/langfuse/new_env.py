"""Write code/infra/langfuse/.env once: every secret the Langfuse stack needs, random.

What it does
    Generates each value that docker-compose.yml marks as required
    (`${NAME:?...}`), plus the headless first start (LANGFUSE_INIT_*): one
    organisation, one project with its API key pair, and one local user. It
    writes them to `.env` next to the compose file with mode 0600, and refuses
    to overwrite an existing file.

Why generated, and why never overwritten
    Upstream ships fixed defaults ("mysalt", "miniosecret"). A known password
    on a database is only safe while nothing else can reach it. Random values
    cost nothing and stay safe if a port is ever published by mistake.
    Overwriting would be worse than no file: Postgres, ClickHouse and MinIO
    keep the password they were first started with in their volumes, and
    ENCRYPTION_KEY encrypts stored data. New values against old volumes lock
    the stack out of its own data. To start over, delete the volumes and the
    file together (`docker compose ... down -v`, then remove `.env`).

What it prints
    The path and the login name. Never a secret: terminals and logs are
    copied around. The password and the API keys are read from the file.

What changes at scale
    A shared deployment keeps these in a secret manager and rotates them; the
    INIT_* values only seed an empty instance and are ignored after.

Run: `make obs-secrets` (from code/), or `uv run python infra/langfuse/new_env.py`.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import secrets
import sys

ENV_PATH = Path(__file__).resolve().parent / ".env"

# The local login. example.com is reserved for documentation (RFC 2606), so
# the address can never reach anyone. Langfuse needs some email-shaped name.
INIT_USER_EMAIL = "admin@example.com"
# Fixed ids, so the project's URL and the tracer's settings never change.
PROJECT_ID = "findyfixy"


def render_env(token: Callable[[int], str] = secrets.token_hex) -> str:
    """The `.env` text. Pure apart from `token`, which the tests replace.

    `token(n)` returns n random bytes as 2n hex characters. ENCRYPTION_KEY must
    be exactly 32 bytes (64 hex characters): Langfuse refuses any other length.
    """
    values = {
        "POSTGRES_PASSWORD": token(24),
        "CLICKHOUSE_PASSWORD": token(24),
        "MINIO_ROOT_PASSWORD": token(24),
        "REDIS_AUTH": token(24),
        "SALT": token(24),
        "ENCRYPTION_KEY": token(32),
        "NEXTAUTH_SECRET": token(32),
        "LANGFUSE_INIT_ORG_ID": PROJECT_ID,
        "LANGFUSE_INIT_ORG_NAME": "FindyFixy",
        "LANGFUSE_INIT_PROJECT_ID": PROJECT_ID,
        "LANGFUSE_INIT_PROJECT_NAME": "FindyFixy",
        # The pk-lf- / sk-lf- prefixes are Langfuse's own key format.
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": f"pk-lf-{token(16)}",
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": f"sk-lf-{token(16)}",
        "LANGFUSE_INIT_USER_EMAIL": INIT_USER_EMAIL,
        # No spaces in any value: the file also loads with `set -a; . .env`.
        "LANGFUSE_INIT_USER_NAME": "Admin",
        "LANGFUSE_INIT_USER_PASSWORD": token(16),
    }
    lines = ["# Written by new_env.py. Secrets: never commit, never paste into chat."]
    lines += [f"{name}={value}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


def write_env(path: Path, text: str) -> None:
    """Create `path` with mode 0600 and write `text`. Raises FileExistsError if it exists.

    `O_EXCL` makes the existence check and the creation one step, so two runs
    at once cannot both write. The mode is set at creation, so the secrets are
    never readable by others, not even for a moment.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)


def main(path: Path = ENV_PATH) -> int:
    """Write the file, or explain why not. Returns the exit code."""
    try:
        write_env(path, render_env())
    except FileExistsError:
        print(f"{path.name} already exists; kept it (its secrets match the data volumes).")
        return 0
    print(f"wrote {path.name} (mode 0600). Langfuse login: {INIT_USER_EMAIL}; password in that file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
