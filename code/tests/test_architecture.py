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
