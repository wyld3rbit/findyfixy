"""Configuration read from the environment, in one place.

Why: five entry points each called `load_dotenv`, read `SMALL_MODEL` and
`LARGE_MODEL` their own way (some with `os.environ[...]`, one with `.get`), and
hard-coded their own default judge model. When a setting changes, one place
should change.

Rules this module keeps:
- Nothing is read at import time. Importing stays cheap and tests stay hermetic.
- A required variable that is missing raises `MissingEnvironment`, which names
  it. The command line turns that one error into "set the environment variable";
  a KeyError from a bug elsewhere keeps its traceback.
- `load_env` never overrides a variable that is already set, so CI or a
  container can set values without editing `.env`.
- Nothing here holds a secret in code. Model names come from `.env`; the
  tracing key pair comes from the environment or from the local Langfuse
  stack's own `.env` (both git-ignored), and `TraceConfig` never prints it.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from harness.trace.tracing import TraceConfig

# The judge defaults to a different family from the answering model, so it does
# not favour its own style when it scores.
DEFAULT_JUDGE_MODEL = "qwen2.5:7b"


class MissingEnvironment(Exception):
    """A required environment variable is not set. The message is its name."""


def load_env() -> None:
    """Load `code/.env` into the environment, without overriding what is set."""
    from dotenv import load_dotenv

    load_dotenv()


def require_env(name: str) -> str:
    """The value of a required environment variable, or `MissingEnvironment`."""
    try:
        return os.environ[name]
    except KeyError:
        raise MissingEnvironment(name) from None


@dataclass(frozen=True)
class Models:
    """The model names a run uses. `small == large` means one model, no escalation."""

    small: str
    large: str
    judge: str


def read_models() -> Models:
    """The configured models. Raises `MissingEnvironment` if small or large is unset."""
    return Models(
        small=require_env("SMALL_MODEL"),
        large=require_env("LARGE_MODEL"),
        judge=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
    )


def optional_models() -> Models | None:
    """The configured models, or None when small or large is unset.

    For the console, which can run in fake mode with no models at all.
    """
    small, large = os.environ.get("SMALL_MODEL"), os.environ.get("LARGE_MODEL")
    if not (small and large):
        return None
    return Models(small, large, os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))


# The local Langfuse (code/infra/langfuse/). Only used when a key pair is found.
DEFAULT_TRACE_HOST = "http://127.0.0.1:3000"
SERVICE_NAME = "findyfixy"


_TRACING_OFF = {"off", "0", "false", "no"}
_TRACING_ON = {"", "on", "1", "true", "yes"}


def read_tracing(stack_env: Path | None = None) -> TraceConfig | None:
    """Where to send traces, or None when tracing is off.

    In order: `TRACING` set to off, 0, false or no turns it off (any value it
    does not know raises, so a typo cannot leave prompts flowing). Otherwise
    `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from the environment, with
    `LANGFUSE_HOST` (default the local stack); setting only one of the two
    raises `MissingEnvironment` naming the other, because half a config is a
    mistake, not a choice. Otherwise, and only for the local host, the
    project keys the local stack was started with, from `stack_env` (default
    `code/infra/langfuse/.env`, written by `make obs-secrets`), so nobody
    copies a secret by hand. Those keys are never sent to another host. No
    keys anywhere means off: tracing is optional. `TRACE_USER` names who ran
    it (every trace's `user.id`), default "local" until Day 7's login.
    """
    switch = os.environ.get("TRACING", "").strip().lower()
    if switch in _TRACING_OFF:
        return None
    if switch not in _TRACING_ON:
        raise ValueError(f"TRACING={switch!r}: use on or off")
    host = os.environ.get("LANGFUSE_HOST", DEFAULT_TRACE_HOST)
    public, secret = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if bool(public) != bool(secret):
        raise MissingEnvironment("LANGFUSE_SECRET_KEY" if public else "LANGFUSE_PUBLIC_KEY")
    if not public and host == DEFAULT_TRACE_HOST:
        from dotenv import dotenv_values

        from app.paths import LANGFUSE_ENV

        path = stack_env or LANGFUSE_ENV
        stack = dotenv_values(path) if path.is_file() else {}
        public = stack.get("LANGFUSE_INIT_PROJECT_PUBLIC_KEY")
        secret = stack.get("LANGFUSE_INIT_PROJECT_SECRET_KEY")
    if not (public and secret):
        return None
    user = os.environ.get("TRACE_USER", "").strip() or "local"
    return TraceConfig(host=host, public_key=public, secret_key=secret, service_name=SERVICE_NAME, user_id=user)
