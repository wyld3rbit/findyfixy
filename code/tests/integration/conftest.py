"""Shared setup for the integration tests: find Ollama, or skip with a reason.

Unit tests (tests/test_routing.py) use a fake model function. They prove the
routing logic and the graph wiring, and they run in milliseconds with no
network. They cannot prove that the real pieces still fit together: the
`langchain-ollama` client, Ollama's structured-output (constrained decoding)
path, and the models themselves. That is the job of the tests in this folder.

Why a separate bank, kept out of the default run
    - Speed. Real calls take seconds each; the 7B model also pays a cold load.
    - Determinism. Models are non-deterministic. A test that depends on them
      can fail for reasons that have nothing to do with a code change.
    - Environment. They need a running server and pulled models. A fresh
      clone, or CI without a GPU, will not have them.
    So they are marked `integration` and deselected by default. Run them with:
        uv run pytest -m integration

Skip, do not fail, when the environment is missing
    A test that fails because Ollama is off tells you nothing about the code.
    The `ollama` fixture below skips with a specific reason instead, so a red
    result always means "the code is wrong", not "the server is down".
"""

import os
import signal
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from app.ingest import database_url

# Project root is code/, two levels above this file. Resolving from the file,
# not from the current directory, means the tests work from any working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OllamaEnv:
    """The two model names the integration tests will call."""

    small: str
    large: str


def _base_url() -> str:
    """Where the Ollama server is. Honors OLLAMA_HOST, as the Ollama client does.

    OLLAMA_HOST is often written without a scheme ("0.0.0.0:11434"), so add one.
    """
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return host if "://" in host else f"http://{host}"


def _pulled_models() -> set[str]:
    """Names of the models the server has, from its /api/tags endpoint.

    The 2-second timeout matters: when nothing listens, a connect fails fast,
    but a half-open network can hang for a long time. Fail fast so the skip
    is quick. Raises OSError (URLError is a subclass) if the server is down.
    """
    import json

    with urllib.request.urlopen(f"{_base_url()}/api/tags", timeout=2) as resp:
        return {m["name"] for m in json.load(resp)["models"]}


@pytest.fixture(scope="session")
def ollama() -> OllamaEnv:
    """Return the configured model names, or skip every test that asks for it.

    Session scope: the server check runs once, not once per test.
    Three things must be true, and each gets its own skip reason:
      1. SMALL_MODEL and LARGE_MODEL are set (code/.env or the environment).
      2. The server answers.
      3. Both models are pulled. A missing model would otherwise surface as a
         404 deep inside a graph run, which is a confusing place to learn it.
    """
    # Explicit path: find_dotenv() walks up from the calling *file*, which is
    # fragile under pytest. override=False keeps real environment variables
    # (set by CI, say) ahead of the file.
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    small, large = os.environ.get("SMALL_MODEL"), os.environ.get("LARGE_MODEL")
    if not small or not large:
        pytest.skip("SMALL_MODEL and LARGE_MODEL are not set (see code/.env)")

    try:
        pulled = _pulled_models()
    except OSError as exc:
        pytest.skip(f"Ollama is not reachable at {_base_url()} ({exc.__class__.__name__})")

    missing = {small, large} - pulled
    if missing:
        pytest.skip(f"models not pulled: {', '.join(sorted(missing))} (run `ollama pull`)")

    return OllamaEnv(small=small, large=large)


@pytest.fixture
def deadline():
    """A context manager that raises TimeoutError if its body runs too long.

    Why not just trust the client's own timeout? `ollama_ask` builds its client
    with no timeout, and changing app code to suit a test would be the tail
    wagging the dog. A test must not be able to hang the suite, so the bound
    lives in the test layer.

    Mechanism: SIGALRM interrupts a blocking socket read in the main thread
    and runs a handler that raises. That is Unix only (macOS, Linux). On
    Windows there is no SIGALRM, so the tests skip rather than run unbounded.
    """
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("deadline needs SIGALRM (Unix only)")

    @contextmanager
    def _deadline(seconds: int):
        def on_alarm(signum, frame):
            raise TimeoutError(f"no reply within {seconds}s")

        previous = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # always cancel, or the alarm fires in a later test
            signal.signal(signal.SIGALRM, previous)

    return _deadline


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    """A Postgres connection whose search_path is a private schema, dropped at the end.

    Tests that write to `chunks` use this, so they cannot touch the real table.
    It skips, not fails, when the database is down: a red result should mean the
    code is wrong, not that `docker compose up` was not run.
    """
    try:
        connection = psycopg.connect(database_url(), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as error:
        pytest.skip(f"Postgres not reachable (run `docker compose up -d db`): {error}")
    schema = f"test_{os.getpid()}"
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.execute(f"CREATE SCHEMA {schema}")
    connection.execute(f"SET search_path TO {schema}, public")
    register_vector(connection)
    try:
        yield connection
    finally:
        connection.execute(f"DROP SCHEMA {schema} CASCADE")
        connection.close()
