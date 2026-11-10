"""Tests for app/model_cache.py: the cache first, the network only on a miss."""

import pytest

from app.model_cache import cache_first


def test_a_cached_model_loads_with_no_network_call() -> None:
    """One call, local only: a pinned revision in the cache never reaches the Hub."""
    calls: list[bool] = []
    assert cache_first(lambda local_files_only: calls.append(local_files_only) or "model") == "model"
    assert calls == [True]


def test_a_cache_miss_downloads_once() -> None:
    """A fresh clone: the local load misses, then the download runs."""
    calls: list[bool] = []

    def load(local_files_only: bool) -> str:
        """Miss locally, succeed online."""
        calls.append(local_files_only)
        if local_files_only:
            raise OSError("not in the cached files")
        return "downloaded"

    assert cache_first(load) == "downloaded"
    assert calls == [True, False]


def test_any_other_error_stops_the_load() -> None:
    """Only a miss falls back; a bug or a bad file is not retried online."""
    def load(local_files_only: bool) -> str:
        """Fail with something that is not a cache miss."""
        raise ValueError("model dimension does not match")

    with pytest.raises(ValueError):
        cache_first(load)


def test_a_file_system_fault_is_not_hidden_behind_a_download() -> None:
    """A permission error has an error number: it is a fault, not a miss, so it is raised."""
    calls: list[bool] = []

    def load(local_files_only: bool) -> str:
        """Fail locally with a real OS error."""
        calls.append(local_files_only)
        raise PermissionError(13, "Permission denied")

    with pytest.raises(PermissionError):
        cache_first(load)
    assert calls == [True]
