"""Load a pinned Hugging Face model from the local cache, and reach the network only when it is not there.

What it does
    `cache_first(load)` calls `load(local_files_only=True)`. Only when that
    fails as a cache miss (the pinned revision is not in the cache: a fresh
    clone, or a cleared cache) does it call `load(local_files_only=False)`,
    which downloads it once.

Why
    Without `local_files_only`, every load of the embedder or the reranker
    asks huggingface.co whether the files changed, even though the revision
    is pinned to a commit and cannot change. On this machine each of those
    calls raised a firewall prompt during runs. A pinned revision already in
    the cache needs no network at all, so a normal run now makes no call to
    the Hub. The `HF_HUB_OFFLINE` environment variable would do the same, but
    it is read when the library is imported and would also block the first
    download a fresh clone needs; this keeps both cases in one place.

Failure modes
    - The library reports a miss as a bare `OSError` with no error number.
      An `OSError` that carries one (a permission error, a full disk, an I/O
      error) is a real fault: it propagates, and is never hidden behind a
      download. So does anything that is not an `OSError` (a wrong
      dimension, a bug).
    - The fallback download is the same call the code made before, so a
      fresh clone behaves exactly as it did.
"""

from collections.abc import Callable
from typing import TypeVar

M = TypeVar("M")


def cache_first(load: Callable[..., M]) -> M:
    """The model from the local cache, or downloaded once if the cache does not hold it."""
    try:
        return load(local_files_only=True)
    except OSError as err:
        if err.errno is not None:  # a real file-system fault, not a miss
            raise
        return load(local_files_only=False)
