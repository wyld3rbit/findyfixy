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
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def package_names(requirements: list[str]) -> set[str]:
    """Normalised distribution names from requirement strings. Pure."""
    return {
        re.split(r"[\s;<>=!~\[(]", r.strip(), maxsplit=1)[0].lower().replace("_", "-")
        for r in requirements
    }


def test_no_direct_dependency_is_cancelled_by_an_override():
    data = tomllib.loads(PYPROJECT.read_text())
    overridden = package_names(data["tool"]["uv"].get("override-dependencies", []))
    direct = package_names(data["project"]["dependencies"])
    for group in data.get("dependency-groups", {}).values():
        direct |= package_names([g for g in group if isinstance(g, str)])
    clash = sorted(overridden & direct)
    assert not clash, (
        f"{clash} are direct dependencies AND removed by override-dependencies in "
        "pyproject.toml. The install would silently leave them out. Delete the "
        "matching override line(s) first."
    )


def test_package_names_handles_the_forms_that_appear_in_pyproject():
    assert package_names(
        ["PyJWT>=2.10", "uvicorn ; sys_platform == 'never'", "langchain_ollama[x]>=1",
         "mcp>=1.30.0,<2"]
    ) == {"pyjwt", "uvicorn", "langchain-ollama", "mcp"}
