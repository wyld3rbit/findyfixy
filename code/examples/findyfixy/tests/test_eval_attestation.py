"""FindyFixy's attestation inputs: the eval path, the files it reads, and where CI looks.

What these tests cover
    The import closure of `ATTEST_ENTRY` over the real tree holds the eval
    path and none of the console-only modules. Every path that code on the
    eval path builds from the findyfixy/paths.py constants is inside the declared
    `ATTEST_EXTRA`, or is one of the gate's own outputs. The Makefile target
    and the CI workflow check the same file the gate writes.

What a staff+ reader should notice
    The path test is the guard for the gap the closure cannot see: a file
    read at runtime. It resolves `DATA / "x.json"` and every other join of a
    paths constant with string literals, so a new input on the eval path
    fails here until it is declared. A path built any other way (an
    environment variable, a computed name) is reported as unresolvable
    rather than skipped.
"""

import ast
from pathlib import Path
import tomllib

from findyfixy.evals.answers import ATTEST_ENTRY, ATTEST_EXTRA, ATTEST_SOURCE_ROOTS, ATTESTATION_PATH
import findyfixy.paths as paths
from findyfixy.paths import WORKSPACE
from wyld3rness.eval.attest import fingerprint, import_closure

# Modules that serve the console, the demos or the MCP server, not the eval.
# A change to any of them must not ask for a 30-case rerun.
# Paths relative to the workspace root (code/), as the attestation writes them.
APP = "examples/findyfixy/src/findyfixy/"
HARNESS = "src/wyld3rness/"
OFF_THE_EVAL_PATH = (
    *(
        APP + m
        for m in (
            "console/server.py",
            "console/eval_run.py",
            "console/fakes.py",
            "console/dev_reload.py",
            "console/stress.py",
            "tools/direct_run.py",
            "mcp/client.py",
            "mcp/run.py",
            "mcp/server.py",
        )
    ),
    HARNESS + "testing.py",
)
ON_THE_EVAL_PATH = tuple(
    APP + m
    for m in (
        "evals/answers.py",
        "evals/adapter.py",
        "triage/graph.py",
        "triage/record.py",
        "retrieval/retrieve.py",
        "retrieval/ingest.py",
        "llm.py",
    )
) + tuple(HARNESS + m for m in ("eval/runner.py", "eval/metrics.py", "eval/judge.py", "eval/attest.py"))
# Paths the gate writes and never reads, so they are not inputs.
OUTPUTS = {ATTESTATION_PATH.relative_to(WORKSPACE).as_posix()}
# Read on the eval path but deliberately not attested. `.env` is git-ignored
# (CI's checkout has none to hash) and holds secrets; what it sets that shapes
# answers, the model names, is recorded in every report's provenance instead.
NOT_ATTESTED = {"examples/findyfixy/.env"}


def closure() -> list[str]:
    """The answer-affecting modules, as paths relative to code/."""
    return [p.relative_to(WORKSPACE).as_posix() for p in import_closure(WORKSPACE, ATTEST_ENTRY, ATTEST_SOURCE_ROOTS)]


def test_the_closure_is_the_eval_path_and_not_the_console() -> None:
    """The set is computed from imports, so this pins what it must and must not hold."""
    files = closure()
    assert set(ON_THE_EVAL_PATH) <= set(files)
    assert set(OFF_THE_EVAL_PATH).isdisjoint(files), set(OFF_THE_EVAL_PATH) & set(files)


def path_constants() -> dict[str, str]:
    """Every Path constant in findyfixy/paths.py, as a path relative to code/ ("" for the root)."""
    found = {}
    for name, value in vars(paths).items():
        if name.isupper() and isinstance(value, Path) and value.is_relative_to(WORKSPACE):
            rel = value.relative_to(WORKSPACE).as_posix()
            found[name] = "" if rel == "." else rel
    return found


def _base(node: ast.expr, constants: dict[str, str]) -> str | None:
    """The relative path a `NAME` or `paths.NAME` stands for, else None."""
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "paths":
        return constants.get(node.attr)
    return None


def resolve(node: ast.expr, constants: dict[str, str]) -> str | None:
    """`DATA / "a" / "b"` -> "data/a/b". None when the left end is not a paths constant.

    Raises ValueError when it is, but a later part is not a string literal:
    such a path cannot be checked, so it must not pass unseen.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return _base(node, constants)
    left = resolve(node.left, constants)
    if left is None:
        return None
    if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
        raise ValueError(f"{ast.unparse(node)} is built from a value, not a literal")
    return f"{left}/{node.right.value}".lstrip("/")


def built_paths(path: Path, constants: dict[str, str]) -> set[str]:
    """Every outermost path join in a module that starts at a paths constant. Pure."""
    tree = ast.parse(path.read_text())
    inner = {id(n.left) for n in ast.walk(tree) if isinstance(n, ast.BinOp)}
    return {
        found
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and id(node) not in inner
        and (found := resolve(node, constants)) is not None
    }


def covered(rel: str) -> bool:
    """True when a declared file is `rel`, or a declared directory holds it."""
    return any(rel == extra or (extra.endswith("/") and rel.startswith(extra)) for extra in ATTEST_EXTRA)


def test_every_path_the_eval_path_builds_is_declared_or_an_output() -> None:
    """A runtime read is invisible to the closure, so this reads the joins instead."""
    constants = path_constants()
    built = {
        (rel, found)
        for rel in closure()
        if rel != APP + "paths.py"  # where the constants themselves are defined
        for found in built_paths(WORKSPACE / rel, constants)
    }
    assert any(found.startswith("examples/findyfixy/data/") for _, found in built)  # the scan sees real reads
    undeclared = sorted(
        (rel, found) for rel, found in built if not covered(found) and found not in OUTPUTS | NOT_ATTESTED
    )
    assert undeclared == [], f"declare these in ATTEST_EXTRA (findyfixy/evals/answers.py): {undeclared}"


def test_the_declared_inputs_fingerprint_on_this_tree() -> None:
    """Every extra exists, the data directory is expanded, and its README is not an input."""
    found = fingerprint(WORKSPACE, ATTEST_ENTRY, ATTEST_EXTRA, ATTEST_SOURCE_ROOTS)
    ex = "examples/findyfixy/"
    assert {ex + "data/golden.jsonl", ex + "data/incidents.jsonl", "uv.lock", ex + "docker-compose.yml"} <= set(
        found.files
    )
    assert ex + "data/README.md" not in found.files and len(found.digest) == 64


def test_make_and_ci_verify_the_file_the_gate_writes() -> None:
    """One attestation path: the gate's, the Makefile's and the workflow's agree."""
    rel = ATTESTATION_PATH.relative_to(WORKSPACE).as_posix()
    makefile = (WORKSPACE / "Makefile").read_text()
    workflow = (WORKSPACE.parent / ".github" / "workflows" / "check.yml").read_text()
    assert f"src/wyld3rness/eval/attest.py verify --root . --attestation {rel}" in makefile
    assert f"code/src/wyld3rness/eval/attest.py verify --root code --attestation {rel}" in workflow


def test_every_workspace_package_is_a_source_root() -> None:
    """A workspace member is an editable install, not pinned by uv.lock.

    Left out of the source roots, its code would be skipped as a locked
    library, and a change to it would pass CI without a run.
    """
    workspace = tomllib.loads((WORKSPACE / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]
    expected = {"src", *(f"{member}/src" for member in workspace["members"])}
    assert set(ATTEST_SOURCE_ROOTS) == expected
