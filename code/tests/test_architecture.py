"""Architecture rules, enforced as tests so they cannot decay.

A rule that lives only in a document is forgotten the first time someone is in
a hurry. These tests read the source and fail when a rule is broken.
"""

import ast
from pathlib import Path

import app.paths as paths

APP = Path(paths.__file__).resolve().parent


def test_project_paths_are_defined_once() -> None:
    """Only app/paths.py may work out where the project root is.

    Ten modules once did it themselves. Two of them could disagree after a move.
    """
    offenders = [
        p.name
        for p in sorted(APP.glob("*.py"))
        if p.name != "paths.py" and "parents[1]" in p.read_text()
    ]
    assert offenders == [], f"compute the root in app/paths.py instead: {offenders}"


def test_paths_resolve_to_the_project() -> None:
    assert (paths.ROOT / "pyproject.toml").is_file()
    assert paths.DATA == paths.ROOT / "data" and paths.DATA.is_dir()
    assert (paths.DOCS / "architecture.html").is_file()


def test_there_are_no_import_cycles_inside_app() -> None:
    """The module graph is a DAG. A cycle makes import order matter, and that is a trap."""
    graph: dict[str, set[str]] = {}
    for p in APP.glob("*.py"):
        deps = set()
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("app."):
                deps.add(node.module.split(".")[1])
        graph[p.stem] = deps

    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, trail: list[str]) -> None:
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


# Entry points build the real thing. Domain code takes its dependencies as
# arguments and must not reach for the composition root or for configuration.
ENTRY_POINTS = {"console", "eval_answers", "eval_retrieval", "graph", "direct_run", "mcp_run"}


def _importers(target: str) -> set[str]:
    found = set()
    for p in APP.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == f"app.{target}":
                found.add(p.stem)
    return found


def test_only_entry_points_import_the_composition_root_and_settings() -> None:
    """Domain code (triage, retrieve, metrics, judge, golden...) gets its dependencies passed in.

    If a deep module imported wiring, it could quietly open a database or load a
    model, and its tests would need one too.
    """
    for target in ("wiring", "settings"):
        extra = _importers(target) - ENTRY_POINTS - {"wiring", "settings"}
        assert extra == set(), f"{sorted(extra)} import app.{target} but are not entry points"


# Pure domain code: rules and data shapes with no vendor client. These modules
# can then be tested with plain values and reused with any model server.
PURE_MODULES = ("triage", "events", "metrics", "golden", "paths", "settings")
VENDOR_CLIENTS = ("langchain", "ollama", "psycopg", "pgvector", "sentence_transformers", "mcp")


def test_pure_modules_import_no_vendor_client() -> None:
    for name in PURE_MODULES:
        for node in ast.walk(ast.parse((APP / f"{name}.py").read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            for module in modules:
                assert module.split(".")[0] not in VENDOR_CLIENTS, f"{name}.py imports {module}"

