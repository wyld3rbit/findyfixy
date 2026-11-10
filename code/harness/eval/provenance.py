"""What produced these numbers: the stamp written into every evaluation report.

Why: a score without its inputs cannot be trusted or compared. The stamp records the
judge model and its fingerprint, the hashes of the judge prompts and the golden set,
the git commit and whether the tree was clean. The system under test adds its own
settings through `describe()` (for FindyFixy: the answering models, the router
threshold, the recall depth). Clock, git and Ollama are injected so tests need none
of them.

Why the stamp stays flat: saved reports and the committed baseline already use flat
keys (`small_model` next to `judge_model`). Nesting the app's keys would change the
schema every reader depends on. The cost of flat is a possible name clash, so a
clash is refused loudly instead of letting an app key overwrite a harness one.

Failure model: when Ollama or git cannot answer, the field is "unknown", and a
baseline refuses an unknown fingerprint (see report.baseline_refusals).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from typing import Any

from harness.eval.metrics import PROMPT_VERSIONS
from harness.eval.stats import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, require_schema

# The keys the harness writes: the one definition `build_provenance` checks its
# own output against and the `describe()` clash check reads. `describe()` may
# not reuse any of them.
CORE_KEYS: tuple[str, ...] = (
    "judge_model",
    "judge_digest",
    "prompt_versions",
    "golden_sha256",
    "git_sha",
    "git_dirty",
    "timestamp_utc",
    "bootstrap_seed",
    "bootstrap_resamples",
    "negative_control",
)

# ---- Provenance -------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """SHA-256 hex of a file's bytes. Pins the golden set into the report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_provenance(
    *,
    judge_model: str,
    digest: Callable[[str], str],
    golden_sha256: str,
    describe: Mapping[str, Any],
    git: Callable[[], tuple[str, bool | None]],
    now: Callable[[], datetime],
    seed: int = BOOTSTRAP_SEED,
    negative_control: bool = False,
) -> dict[str, Any]:
    """Everything needed to say what produced these numbers. Clock, git and Ollama injected.

    `describe` is the system under test's own settings, merged flat. A key that
    is also a harness key raises ValueError: the app would silently replace the
    judge fingerprint or the git stamp, and the gate would trust a forged stamp.

    `negative_control` is True only for a run built to fail (for FindyFixy,
    `--no-recall`). Recording it means such a run can never be mistaken for a
    real one, and it can never become a baseline.
    """
    clash = sorted(set(describe) & set(CORE_KEYS))
    if clash:
        raise ValueError(f"describe() reuses harness provenance keys: {clash}")
    sha, dirty = git()
    core = {
        "judge_model": judge_model,
        "judge_digest": digest(judge_model),
        "prompt_versions": dict(PROMPT_VERSIONS),
        "golden_sha256": golden_sha256,
        "git_sha": sha,
        "git_dirty": dirty,
        "timestamp_utc": now().astimezone(timezone.utc).isoformat(timespec="seconds"),
        "bootstrap_seed": seed,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "negative_control": negative_control,
    }
    require_schema(core, CORE_KEYS, "the provenance stamp")
    return {**core, **describe}


def ollama_digest(model: str) -> str:
    """The digest Ollama reports for a local model, or "unknown".

    The digest is what makes a silent model swap visible: the tag stays the
    same while the weights change. Any failure (server off, model absent)
    returns "unknown" rather than stopping the eval. The gate does not treat
    two "unknown" digests as equal: `report.comparability_problems` reports an
    unknown digest on either side as a problem, and a baseline is refused when
    its digest is unknown, because neither side can show which judge scored.
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


def git_info(repo: Path) -> tuple[str, bool | None]:
    """Short commit SHA of `repo` and whether its tree has uncommitted changes.

    The caller passes the directory, so the harness assumes nothing about where
    it sits inside the repository. Any directory inside the checkout works: git
    walks up to the root, and `status --porcelain` covers the whole tree.
    Falls back to ("unknown", None) outside a git checkout or without git.
    A dirty tree means the SHA does not fully describe the code that ran.
    """

    def run(*args: str) -> str:
        """One git command in `repo`. Raises on a non-zero exit or a 10 s hang."""
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()

    try:
        return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain"))
    except Exception:
        return "unknown", None
