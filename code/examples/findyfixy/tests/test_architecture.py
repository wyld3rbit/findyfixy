"""FindyFixy's architecture rules, enforced as tests so they cannot decay.

A rule that lives only in a document is forgotten the first time someone is in
a hurry. These tests read the source and fail when a rule is broken. The
harness's own rules, including that it never imports an app, are in the
harness suite (code/tests/test_harness_rules.py).
"""

import ast
from pathlib import Path

import findyfixy.paths as paths

APP = paths.PACKAGE


def _imported_modules(path: Path) -> list[str]:
    """Every module a file imports, at any depth (a lazy import inside a function counts)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(a.name for a in node.names)
    return found


def test_project_paths_are_defined_once() -> None:
    """Only findyfixy/paths.py may work out where the project root is.

    Ten modules once did it themselves. Two of them could disagree after a move.
    """
    offenders = [
        str(p.relative_to(APP))
        for p in _modules()
        if p.name != "paths.py" and ("parents[" in p.read_text() or ".parent.parent" in p.read_text())
    ]
    assert offenders == [], f"compute the root in findyfixy/paths.py instead: {offenders}"


def test_paths_resolve_to_the_project() -> None:
    """The root constants point at real folders of this project."""
    assert (paths.ROOT / "pyproject.toml").is_file()
    assert (paths.WORKSPACE / "pyproject.toml").is_file() and (paths.WORKSPACE / "uv.lock").is_file()
    assert paths.DATA == paths.ROOT / "data" and paths.DATA.is_dir()
    assert paths.PAGE.is_file()


def _modules() -> list[Path]:
    """Every Python file in the package, sorted for a stable failure message."""
    return sorted(APP.rglob("*.py"))


def _module_name(path: Path) -> str:
    """`.../findyfixy/triage/graph.py` -> `findyfixy.triage.graph`."""
    parts = path.relative_to(APP.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _assert_no_cycles(files: list[Path], prefix: str) -> None:
    """Fail on any import cycle among `files`, counting only imports under `prefix`."""
    graph: dict[str, set[str]] = {}
    for p in files:
        graph[_module_name(p)] = {m for m in _imported_modules(p) if m.startswith(prefix)}

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


def test_there_are_no_import_cycles() -> None:
    """The module graph is a DAG. A cycle makes import order matter, and that is a trap."""
    _assert_no_cycles(_modules(), "findyfixy.")


def test_no_relative_imports() -> None:
    """Every import is absolute, so the import checks in this file see every edge.

    The cycle, dependency-rule and vendor checks match module names such as
    `findyfixy.paths`. A relative import (`from . import paths`) carries no package
    name, so it would slip past all of them. Banning it keeps those checks whole.
    """
    offenders = [
        f"{p.relative_to(APP.parent)}:{node.lineno}"
        for p in _modules()
        for node in ast.walk(ast.parse(p.read_text()))
        if isinstance(node, ast.ImportFrom) and node.level != 0
    ]
    assert offenders == [], f"use an absolute import: {offenders}"


# Entry points build the real thing. Domain code takes its dependencies as
# arguments and must not reach for the composition root or for configuration.
# Every module with a command `main`. ingest, search and the MCP server joined
# on Day 6: each traces its command, so each reads tracing settings.
ENTRY_POINTS = {
    f"findyfixy.{m}"
    for m in (
        "console.server",
        "evals.answers",
        "evals.retrieval",
        "triage.graph",
        "tools.direct_run",
        "mcp.run",
        "retrieval.ingest",
        "retrieval.search",
        "mcp.server",
        "observability.prices",
    )
}
# The composition root and the configuration it reads.
COMPOSITION = {"findyfixy.wiring", "findyfixy.settings", "findyfixy.observability.tracing"}


def _importers(target: str) -> set[str]:
    """The modules that import `target` (a dotted name) directly."""
    found = set()
    for p in _modules():
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == target:
                found.add(_module_name(p))
    return found


def test_only_entry_points_import_the_composition_root_and_settings() -> None:
    """Domain code (triage, retrieve, metrics, judge, golden...) gets its dependencies passed in.

    If a deep module imported wiring, it could quietly open a database or load a
    model, and its tests would need one too.
    """
    for target in COMPOSITION:
        extra = _importers(target) - ENTRY_POINTS - COMPOSITION
        assert extra == set(), f"{sorted(extra)} import {target} but are not entry points"


# Pure domain code: rules and data shapes with no vendor client. These modules
# can then be tested with plain values and reused with any model server.
PURE_MODULES = ("triage/record", "events", "evals/golden", "paths", "settings", "evals/adapter")
VENDOR_CLIENTS = ("langchain", "ollama", "psycopg", "pgvector", "sentence_transformers", "mcp")


def _assert_no_vendor_client(path: Path) -> None:
    """Fail if `path` imports a vendor client anywhere in the file."""
    for module in _imported_modules(path):
        # startswith: "langchain_ollama" and "langchain_core" are langchain too.
        assert not module.split(".")[0].startswith(VENDOR_CLIENTS), f"{path.name} imports {module}"


def test_pure_modules_import_no_vendor_client() -> None:
    """Pure app modules can be tested with plain values."""
    for name in PURE_MODULES:
        _assert_no_vendor_client(APP / f"{name}.py")


def test_the_console_fakes_are_used_only_by_the_console() -> None:
    """Demo and test data must never become a dependency of real code."""
    assert _importers("findyfixy.console.fakes") <= {"findyfixy.console.server"}
