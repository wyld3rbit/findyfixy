"""Tests for app/settings.py: configuration read from the environment."""

import pytest

from app.settings import (
    DEFAULT_JUDGE_MODEL,
    MissingEnvironment,
    optional_models,
    read_models,
    require_env,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("SMALL_MODEL", "LARGE_MODEL", "JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_require_env_returns_the_value_or_names_the_missing_variable(monkeypatch) -> None:
    monkeypatch.setenv("SOME_SETTING", "x")
    assert require_env("SOME_SETTING") == "x"
    with pytest.raises(MissingEnvironment, match="NOT_SET_ANYWHERE"):
        require_env("NOT_SET_ANYWHERE")


def test_read_models_needs_small_and_large_and_defaults_the_judge(monkeypatch) -> None:
    monkeypatch.setenv("SMALL_MODEL", "a")
    with pytest.raises(MissingEnvironment, match="LARGE_MODEL"):
        read_models()
    monkeypatch.setenv("LARGE_MODEL", "b")
    models = read_models()
    assert (models.small, models.large, models.judge) == ("a", "b", DEFAULT_JUDGE_MODEL)


def test_the_judge_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("SMALL_MODEL", "a")
    monkeypatch.setenv("LARGE_MODEL", "b")
    monkeypatch.setenv("JUDGE_MODEL", "j")
    assert read_models().judge == "j"


def test_optional_models_is_none_unless_both_are_set(monkeypatch) -> None:
    assert optional_models() is None
    monkeypatch.setenv("SMALL_MODEL", "a")
    assert optional_models() is None
    monkeypatch.setenv("LARGE_MODEL", "b")
    assert optional_models().small == "a"


def test_an_empty_model_name_counts_as_not_set(monkeypatch) -> None:
    monkeypatch.setenv("SMALL_MODEL", "")
    monkeypatch.setenv("LARGE_MODEL", "b")
    assert optional_models() is None


def test_nothing_is_read_at_import_time() -> None:
    """Importing settings must not need .env or any variable."""
    import importlib

    import app.settings

    importlib.reload(app.settings)  # would raise if it read the environment on import
