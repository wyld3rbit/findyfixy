"""Day 2: the realistic incident bank, loaded from `data/incidents.json`.

What it does
    Defines `Incident` (a synthetic log excerpt plus a hand-written answer
    key) and `load_incidents` (read and validate the file).

Why a second bank beside the lesson questions
    The five lesson questions are one-liners. A 1.5B model can answer them
    from general knowledge, so they say little about how it handles real
    evidence. These incidents are 20 to 40 line log excerpts with noise, red
    herrings and an upstream root cause, so the model has to find the signal.
    The lesson five stay as they are: they are fixed word for word so runs
    stay comparable over time.

The `expected` answer key
    Hand-written, never shown to the model. It lets a run be scored: did the
    triage name an acceptable component and the right severity? Several
    components can be acceptable because a fault often has a defensible
    "where" at more than one level (the service, or the thing under it).
    Day 5's golden set can reuse these.

Size limit, and why it is a test
    Ollama runs these models at a 4096-token context, even though they
    support more. A prompt that overflows is silently cut from the start:
    no error, just a worse answer. The retry prompt repeats the whole
    question, so it is about twice the size. `MAX_TEXT_CHARS` keeps both
    inside the window. `tests/test_incidents.py` fails if an incident grows
    past it.

Synthetic data only
    Documentation IP ranges, `example.*` hosts, invented service names. No
    real company, person or customer. A test rejects any other IPv4 address.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from app.paths import DATA

# Resolved from this file, never from the current directory.
DEFAULT_PATH = DATA / "incidents.json"

# About 1,000 tokens of log text. The retry prompt doubles it and adds the
# instructions and the bad reply, which still fits in 4096.
MAX_TEXT_CHARS = 3500


class Expected(BaseModel):
    """The hand-written answer key for one incident."""

    severity: Literal["sev1", "sev2", "sev3", "sev4"]
    components: list[str] = Field(min_length=1)


class Incident(BaseModel):
    """One synthetic incident: what the model sees, and the answer key."""

    id: str
    title: str
    trap: str = Field(description="What makes this one hard, for the reader.")
    expected: Expected
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


def load_incidents(path: Path = DEFAULT_PATH) -> list[Incident]:
    """Read and validate the bank. Raises if the file is missing or malformed.

    Not called at import. A bad file should stop startup loudly, not surface
    as a confusing failure in the middle of a run.
    """
    return [Incident.model_validate(item) for item in json.loads(path.read_text())]
