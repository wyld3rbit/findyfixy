"""Guard: a dependency override must never cancel a package we depend on directly.

Why this exists
    `pyproject.toml` drops four packages that `mcp` requires but only imports for
    HTTP serving, OAuth and its CLI (`override-dependencies` with a never-true
    marker). An override replaces EVERY requirement for that name, including one
    we add ourselves. Tested: with the override in place, `uv add pyjwt`
    succeeds and writes the line, and then `import jwt` fails. Nothing says why.

    Day 7 (the credential broker) needs PyJWT and Day 9 (the API) needs a server
    such as uvicorn. This test turns that silent failure into an immediate one
    with the fix in the message.
"""

import re
import tomllib

from findyfixy.paths import WORKSPACE

PYPROJECT = WORKSPACE / "pyproject.toml"  # the workspace root holds [tool.uv] and the overrides


def package_names(requirements: list[str]) -> set[str]:
    """Normalised distribution names from requirement strings. Pure."""
    return {re.split(r"[\s;<>=!~\[(]", r.strip(), maxsplit=1)[0].lower().replace("_", "-") for r in requirements}


def direct_dependencies(root: dict, members: list[dict]) -> set[str]:
    """Every package any workspace project declares, in its dependencies or a group. Pure.

    The override in the root file applies to the whole workspace, so a package
    a member adds (`uv add --package findyfixy pyjwt`) is cancelled just the same.
    """
    names: set[str] = set()
    for data in [root, *members]:
        names |= package_names(data.get("project", {}).get("dependencies", []))
        for group in data.get("dependency-groups", {}).values():
            names |= package_names([g for g in group if isinstance(g, str)])
    return names


def test_no_direct_dependency_is_cancelled_by_an_override():
    data = tomllib.loads(PYPROJECT.read_text())
    overridden = package_names(data["tool"]["uv"].get("override-dependencies", []))
    members = [
        tomllib.loads((WORKSPACE / m / "pyproject.toml").read_text())
        for m in data["tool"]["uv"]["workspace"]["members"]
    ]
    assert members, "the workspace lists no members: this test would check the harness alone"
    direct = direct_dependencies(data, members)
    assert "langgraph" in direct  # FindyFixy's own dependencies are seen
    clash = sorted(overridden & direct)
    assert not clash, (
        f"{clash} are direct dependencies AND removed by override-dependencies in "
        "pyproject.toml. The install would silently leave them out. Delete the "
        "matching override line(s) first."
    )


def test_package_names_handles_the_forms_that_appear_in_pyproject():
    assert package_names(
        ["PyJWT>=2.10", "uvicorn ; sys_platform == 'never'", "langchain_ollama[x]>=1", "mcp>=1.30.0,<2"]
    ) == {"pyjwt", "uvicorn", "langchain-ollama", "mcp"}
