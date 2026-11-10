"""FindyFixy's attestation inputs: the eval path, the files it reads, and where CI looks.

What these tests cover
    The import closure of `ATTEST_ENTRY` over the real tree holds the eval
    path and none of the console-only modules. Every path that code on the
    eval path builds from the app/paths.py constants is inside the declared
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

import app.paths as paths
from app.eval_answers import ATTEST_ENTRY, ATTEST_EXTRA, ATTESTATION_PATH
from app.paths import ROOT
from harness.eval.attest import fingerprint, import_closure

# Modules that serve the console, the demos or the MCP server, not the eval.
# A change to any of them must not ask for a 30-case rerun.
OFF_THE_EVAL_PATH = (
    "app/console.py", "app/console_eval.py", "app/console_fakes.py", "app/dev_reload.py",
    "app/stress.py", "app/direct_run.py", "app/mcp_client.py", "app/mcp_run.py", "app/mcp_server.py",
)
ON_THE_EVAL_PATH = (
    "app/eval_answers.py", "app/eval_adapter.py", "app/graph.py", "app/triage.py", "app/retrieve.py",
    "app/ingest.py", "app/llm.py", "harness/eval/runner.py", "harness/eval/metrics.py",
    "harness/eval/judge.py", "harness/eval/attest.py",
)
# Paths the gate writes and never reads, so they are not inputs.
OUTPUTS = {ATTESTATION_PATH.relative_to(ROOT).as_posix()}


def closure() -> list[str]:
    """The answer-affecting modules, as paths relative to code/."""
    return [p.relative_to(ROOT).as_posix() for p in import_closure(ROOT, ATTEST_ENTRY)]


def test_the_closure_is_the_eval_path_and_not_the_console() -> None:
    """The set is computed from imports, so this pins what it must and must not hold."""
    files = closure()
    assert set(ON_THE_EVAL_PATH) <= set(files)
    assert set(OFF_THE_EVAL_PATH).isdisjoint(files), set(OFF_THE_EVAL_PATH) & set(files)


def path_constants() -> dict[str, str]:
    """Every Path constant in app/paths.py, as a path relative to code/ ("" for the root)."""
    found = {}
    for name, value in vars(paths).items():
        if name.isupper() and isinstance(value, Path) and value.is_relative_to(ROOT):
            rel = value.relative_to(ROOT).as_posix()
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
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and id(node) not in inner
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
        if rel != "app/paths.py"  # where the constants themselves are defined
        for found in built_paths(ROOT / rel, constants)
    }
    assert any(found.startswith("data/") for _, found in built)  # the scan sees real reads
    undeclared = sorted((rel, found) for rel, found in built if not covered(found) and found not in OUTPUTS)
    assert undeclared == [], f"declare these in ATTEST_EXTRA (app/eval_answers.py): {undeclared}"


def test_the_declared_inputs_fingerprint_on_this_tree() -> None:
    """Every extra exists, the data directory is expanded, and its README is not an input."""
    found = fingerprint(ROOT, ATTEST_ENTRY, ATTEST_EXTRA)
    assert {"data/golden.jsonl", "data/incidents.jsonl", "uv.lock", "docker-compose.yml"} <= set(found.files)
    assert "data/README.md" not in found.files and len(found.digest) == 64


def test_make_and_ci_verify_the_file_the_gate_writes() -> None:
    """One attestation path: the gate's, the Makefile's and the workflow's agree."""
    rel = ATTESTATION_PATH.relative_to(ROOT).as_posix()
    makefile = (ROOT / "Makefile").read_text()
    workflow = (ROOT.parent / ".github" / "workflows" / "check.yml").read_text()
    assert f"harness/eval/attest.py verify --root . --attestation {rel}" in makefile
    assert f"code/harness/eval/attest.py verify --root code --attestation {rel}" in workflow
