"""Architecture rules, enforced as tests so they cannot decay.

A rule that lives only in a document is forgotten the first time someone is in
a hurry. These tests read the source and fail when a rule is broken.
"""

import ast
from pathlib import Path

import app.paths as paths

APP = Path(paths.__file__).resolve().parent
# The harness package, found from the project root the app already defines.
HARNESS = paths.ROOT / "harness"


def _imported_modules(path: Path) -> list[str]:
    """Every module a file imports, at any depth (a lazy import inside a function counts)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(a.name for a in node.names)
    return found


def _harness_modules() -> list[Path]:
    """Every Python file under harness/, sorted for a stable failure message."""
    return sorted(HARNESS.rglob("*.py"))


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
    """The root constants point at real folders of this project."""
    assert (paths.ROOT / "pyproject.toml").is_file()
    assert paths.DATA == paths.ROOT / "data" and paths.DATA.is_dir()
    assert (paths.DOCS / "architecture.html").is_file()


def _module_name(path: Path) -> str:
    """`harness/eval/stats.py` -> `harness.eval.stats`, `app/graph.py` -> `app.graph`."""
    parts = path.relative_to(paths.ROOT).with_suffix("").parts
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


def test_there_are_no_import_cycles_inside_app() -> None:
    """The module graph is a DAG. A cycle makes import order matter, and that is a trap."""
    _assert_no_cycles(sorted(APP.glob("*.py")), "app.")


def test_there_are_no_import_cycles_inside_the_harness() -> None:
    """The harness module graph is a DAG too, so import order never matters."""
    _assert_no_cycles(_harness_modules(), "harness.")


def test_the_harness_never_imports_the_app() -> None:
    """The harness is the product; FindyFixy is its first consumer. The arrow points one way.

    If a harness module imported `app`, the harness could not be reused for a
    second system, and a FindyFixy change could break the gate of every other.
    """
    offenders = [
        f"{p.relative_to(paths.ROOT)} imports {m}"
        for p in _harness_modules()
        for m in _imported_modules(p)
        if m == "app" or m.startswith("app.")
    ]
    assert offenders == [], f"the harness must not import the app: {offenders}"


def test_harness_and_app_use_no_relative_imports() -> None:
    """Every import is absolute, so the import checks in this file see every edge.

    The cycle, dependency-rule and vendor checks match module names such as
    `app.paths`. A relative import (`from . import paths`) carries no package
    name, so it would slip past all of them. Banning it keeps those checks whole.
    """
    offenders = [
        f"{p.relative_to(paths.ROOT)}:{node.lineno}"
        for p in [*sorted(APP.glob("*.py")), *_harness_modules()]
        for node in ast.walk(ast.parse(p.read_text()))
        if isinstance(node, ast.ImportFrom) and node.level != 0
    ]
    assert offenders == [], f"use an absolute import: {offenders}"


def test_the_harness_does_not_work_out_paths_from_its_own_location() -> None:
    """Reports, baseline, golden set and repo directory are passed in by the app.

    A path computed from `__file__` would tie the harness to this repository's layout.
    """
    offenders = [str(p.relative_to(paths.ROOT)) for p in _harness_modules() if "__file__" in p.read_text()]
    assert offenders == [], f"pass the path in instead: {offenders}"


# Entry points build the real thing. Domain code takes its dependencies as
# arguments and must not reach for the composition root or for configuration.
# Every module with a command `main`. `ingest`, `search` and `mcp_server` joined
# on Day 6: each traces its command, so each reads tracing settings.
ENTRY_POINTS = {"console", "eval_answers", "eval_retrieval", "graph", "direct_run", "mcp_run",
                "ingest", "search", "mcp_server"}


def _importers(target: str) -> set[str]:
    """The app modules that import `app.<target>` directly."""
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
    for target in ("wiring", "settings", "tracing"):
        extra = _importers(target) - ENTRY_POINTS - {"wiring", "settings", "tracing"}
        assert extra == set(), f"{sorted(extra)} import app.{target} but are not entry points"


# Pure domain code: rules and data shapes with no vendor client. These modules
# can then be tested with plain values and reused with any model server.
PURE_MODULES = ("triage", "events", "golden", "paths", "settings", "eval_adapter")
# The harness's pure core. judge.py and provenance.py are left out on purpose:
# they hold the one lazy Ollama import each (the real judge call, the digest).
HARNESS_PURE_MODULES = ("case", "sut", "stats", "report", "metrics", "runner", "attest")
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


def test_pure_harness_modules_import_no_vendor_client() -> None:
    """The harness core (stats, report and the rest) imports no vendor client."""
    for name in HARNESS_PURE_MODULES:
        _assert_no_vendor_client(HARNESS / "eval" / f"{name}.py")


def test_the_console_fakes_are_used_only_by_the_console() -> None:
    """Demo and test data must never become a dependency of real code."""
    assert _importers("console_fakes") <= {"console"}

