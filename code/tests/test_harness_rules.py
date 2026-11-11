"""The harness's architecture rules, enforced as tests so they cannot decay.

A rule that lives only in a document is forgotten the first time someone is in
a hurry. These tests read the source and fail when a rule is broken.

The strongest rule is packaging, not a test: the harness is its own
distribution and no app is among its dependencies, so `make test-harness`
runs this suite with no app installed at all. The import allowlist below
says the same thing at the source level, and names the offending line.
"""

import ast
from pathlib import Path
import sys
import tomllib

import wyld3rness

HARNESS = Path(wyld3rness.__file__).resolve().parent
PYPROJECT = HARNESS.parents[1] / "pyproject.toml"
# Top-level import names of the harness's declared dependencies. Kept next to
# pyproject.toml's list by `test_the_allowlist_matches_the_declared_dependencies`.
THIRD_PARTY = {
    "langchain_ollama": "langchain-ollama",
    "ollama": "ollama",
    "opentelemetry": "opentelemetry-*",
    "pydantic": "pydantic",
}


def _imported_modules(path: Path) -> list[str]:
    """Every module a file imports, at any depth (a lazy import inside a function counts)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(a.name for a in node.names)
    return found


def _modules() -> list[Path]:
    """Every Python file in the package, sorted for a stable failure message."""
    return sorted(HARNESS.rglob("*.py"))


def _module_name(path: Path) -> str:
    """`.../wyld3rness/eval/stats.py` -> `wyld3rness.eval.stats`."""
    parts = path.relative_to(HARNESS.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_the_harness_imports_only_the_standard_library_its_dependencies_and_itself() -> None:
    """The harness is the product; apps are its consumers. The arrow points one way.

    If a harness module imported an app (or a library only an app installs),
    the harness could not be reused for a second system, and an app change
    could break the gate of every other.
    """
    allowed = set(sys.stdlib_module_names) | set(THIRD_PARTY) | {"wyld3rness"}
    offenders = [
        f"{p.relative_to(HARNESS.parent)} imports {m}"
        for p in _modules()
        for m in _imported_modules(p)
        if m.split(".")[0] not in allowed
    ]
    assert offenders == [], f"the harness may import only the stdlib, its dependencies and itself: {offenders}"


def test_the_allowlist_matches_the_declared_dependencies() -> None:
    """Adding an import means declaring the package, and the other way round."""
    declared = {
        d.split(">")[0].split("<")[0].split("[")[0].strip()
        for d in tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    }
    expected = {name for name in declared if not name.startswith("opentelemetry-")}
    assert {v for v in THIRD_PARTY.values() if v != "opentelemetry-*"} == expected
    assert any(name.startswith("opentelemetry-") for name in declared)


def test_there_are_no_import_cycles() -> None:
    """The module graph is a DAG, so import order never matters."""
    graph = {_module_name(p): {m for m in _imported_modules(p) if m.startswith("wyld3rness.")} for p in _modules()}
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, trail: list[str]) -> None:
        """Depth-first walk that fails on reaching a module already on the path."""
        if name in done or name not in graph:
            return
        assert name not in visiting, f"import cycle: {' -> '.join([*trail, name])}"
        visiting.add(name)
        for dep in sorted(graph[name]):
            visit(dep, [*trail, name])
        visiting.discard(name)
        done.add(name)

    for module in sorted(graph):
        visit(module, [])


def test_no_relative_imports() -> None:
    """Every import is absolute, so the checks here and the attestation's closure see every edge."""
    offenders = [
        f"{p.relative_to(HARNESS.parent)}:{node.lineno}"
        for p in _modules()
        for node in ast.walk(ast.parse(p.read_text()))
        if isinstance(node, ast.ImportFrom) and node.level != 0
    ]
    assert offenders == [], f"use an absolute import: {offenders}"


def test_the_harness_does_not_work_out_paths_from_its_own_location() -> None:
    """Reports, baseline, golden set and repo directory are passed in by the app.

    A path computed from `__file__` would tie the harness to this repository's layout.
    """
    offenders = [str(p.relative_to(HARNESS.parent)) for p in _modules() if "__file__" in p.read_text()]
    assert offenders == [], f"pass the path in instead: {offenders}"


# The harness's pure core. judge.py and provenance.py are left out on purpose:
# they hold the one lazy Ollama import each (the real judge call, the digest).
PURE_MODULES = ("case", "sut", "stats", "report", "metrics", "runner", "attest")
VENDOR_CLIENTS = ("langchain", "ollama")


def test_pure_modules_import_no_vendor_client() -> None:
    """The core (stats, report and the rest) can be tested with plain values."""
    for name in PURE_MODULES:
        path = HARNESS / "eval" / f"{name}.py"
        for module in _imported_modules(path):
            assert not module.split(".")[0].startswith(VENDOR_CLIENTS), f"{path.name} imports {module}"
