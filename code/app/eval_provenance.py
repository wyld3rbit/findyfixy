"""What produced these numbers: the stamp written into every evaluation report.

Why: a score without its inputs cannot be trusted or compared. The stamp records the
judge model and its fingerprint, the hashes of the judge prompts and the golden set,
the answering models, the router threshold, the git commit and whether the tree was
clean. Clock, git and Ollama are injected so tests need none of them.

Failure model: when Ollama or git cannot answer, the field is "unknown", and a
baseline refuses an unknown fingerprint (see eval_report.baseline_refusals).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from typing import Any

from app.eval_stats import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from app.graph import THRESHOLD
from app.metrics import PROMPT_VERSIONS
from app.paths import ROOT
from app.triage import RECALL_K

# ---- Provenance -------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """SHA-256 hex of a file's bytes. Pins the golden set into the report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_provenance(
    *,
    judge_model: str,
    digest: Callable[[str], str],
    golden_sha256: str,
    small_model: str,
    large_model: str,
    git: Callable[[], tuple[str, bool | None]],
    now: Callable[[], datetime],
    k: int = RECALL_K,
    threshold: float = THRESHOLD,
    seed: int = BOOTSTRAP_SEED,
    recall_enabled: bool = True,
) -> dict[str, Any]:
    """Everything needed to say what produced these numbers. Clock, git and Ollama injected.

    `recall_enabled` is False only for the negative-control run (`--no-recall`).
    Recording it means such a run can never be mistaken for a real one.
    """
    sha, dirty = git()
    return {
        "judge_model": judge_model,
        "judge_digest": digest(judge_model),
        "prompt_versions": dict(PROMPT_VERSIONS),
        "golden_sha256": golden_sha256,
        "small_model": small_model,
        "large_model": large_model,
        "threshold": threshold,
        "recall_k": k,
        "recall_enabled": recall_enabled,
        "git_sha": sha,
        "git_dirty": dirty,
        "timestamp_utc": now().astimezone(timezone.utc).isoformat(timespec="seconds"),
        "bootstrap_seed": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def ollama_digest(model: str) -> str:
    """The digest Ollama reports for a local model, or "unknown".

    The digest is what makes a silent model swap visible: the tag stays the
    same while the weights change. Any failure (server off, model absent)
    returns "unknown" rather than stopping the eval; the gate then treats two
    "unknown" digests as equal, which is a known weak spot.
    """
    try:
        import ollama

        wanted = model if ":" in model else f"{model}:latest"
        for entry in ollama.list().models:
            if entry.model == wanted or entry.model == model:
                return entry.digest or "unknown"
    except Exception:
        return "unknown"
    return "unknown"


def git_info() -> tuple[str, bool | None]:
    """Short commit SHA and whether the tree has uncommitted changes.

    Falls back to ("unknown", None) outside a git checkout or without git.
    A dirty tree means the SHA does not fully describe the code that ran.
    """

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain"))
    except Exception:
        return "unknown", None
