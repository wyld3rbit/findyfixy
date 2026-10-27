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
- Nothing here is a secret. Model names only; secrets belong in `.env`, which is
  git-ignored.
"""

import os
from dataclasses import dataclass

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
