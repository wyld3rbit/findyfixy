"""Tests for the Langfuse secrets writer (infra/langfuse/new_env.py) against its compose file.

What these tests cover
    That every secret the compose file requires is generated, so `obs-up`
    never stops on a missing value; that the file is private and never
    overwritten; and that nothing secret is printed.

What a staff+ reader should notice
    The required names are read from docker-compose.yml itself. When an
    upgrade adds a required secret, this test fails before the stack does.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "infra" / "langfuse" / "docker-compose.yml"
spec = importlib.util.spec_from_file_location("new_env", ROOT / "infra" / "langfuse" / "new_env.py")
new_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new_env)


def parse(text: str) -> dict[str, str]:
    """NAME=value lines to a dict, skipping comments."""
    return dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#"))


def test_every_secret_the_compose_file_requires_is_generated() -> None:
    """Each `${NAME:?...}` in the compose file has a value in the generated file."""
    required = set(re.findall(r"\$\{([A-Z0-9_]+):\?", COMPOSE.read_text()))
    assert required, "the compose file should require its secrets"
    assert required <= parse(new_env.render_env()).keys()


def test_the_compose_file_has_no_default_secret_left() -> None:
    """No secret-bearing setting falls back to a value: upstream's "mysalt" and the rest are gone.

    User names (`CLICKHOUSE_USER:-clickhouse`) and the MinIO access key id
    keep their defaults: they are names, not secrets.
    """
    # A non-empty fallback on a secret-bearing name. Empty fallbacks (`:-}`),
    # such as the unused AWS keys, are not secrets.
    secret = re.compile(r"\$\{[A-Z0-9_]*(PASSWORD|SECRET|SALT|ENCRYPTION_KEY|REDIS_AUTH)[A-Z0-9_]*:-[^}]")
    lines = [line.split("#", 1)[0] for line in COMPOSE.read_text().splitlines()]
    assert [line.strip() for line in lines if secret.search(line)] == []


def test_values_are_random_and_the_encryption_key_has_the_length_langfuse_requires() -> None:
    """Two renders share no secret; ENCRYPTION_KEY is 64 hex characters."""
    a, b = parse(new_env.render_env()), parse(new_env.render_env())
    assert len(a["ENCRYPTION_KEY"]) == 64 and int(a["ENCRYPTION_KEY"], 16) >= 0
    for name in ("POSTGRES_PASSWORD", "SALT", "LANGFUSE_INIT_PROJECT_SECRET_KEY", "LANGFUSE_INIT_USER_PASSWORD"):
        assert a[name] != b[name]
    assert a["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"].startswith("pk-lf-")
    assert a["LANGFUSE_INIT_PROJECT_SECRET_KEY"].startswith("sk-lf-")


def test_the_file_is_private_and_never_overwritten(tmp_path: Path, capsys) -> None:
    """Mode 0600; a second run keeps the first file, whose secrets the volumes already hold."""
    path = tmp_path / ".env"
    assert new_env.main(path) == 0
    first = path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert new_env.main(path) == 0
    assert path.read_text() == first
    printed = capsys.readouterr().out
    for value in parse(first).values():
        if value != new_env.INIT_USER_EMAIL and len(value) > 12:
            assert value not in printed


def test_no_value_needs_quoting_so_a_shell_can_load_the_file() -> None:
    """Docker reads the file either way; `set -a; . .env` breaks on a space or a quote."""
    for name, value in parse(new_env.render_env()).items():
        assert not any(ch in value for ch in " '\"$`\\"), name
