"""The CI gate: a fingerprint of every file that changes answers, and the verdict of a run over them.

What it does
    A live eval run takes hours on a CPU-only CI runner, so the gate runs on
    the developer's machine. On a pass, `write_attestation` records which
    answer-affecting files the run saw (their SHA-256 hashes), the report's
    hash and the verdict. CI then runs `verify`: it recomputes the
    fingerprint from the checkout and fails when any of those files differ
    from the attested ones, or when the verdict is not a pass.

    fingerprint(root, entry_modules, extra_files) -> Fingerprint(digest, files)
    write_attestation(path, report, gate_problems, fingerprint, ...)
    verify(path, fingerprint) -> problems (empty means attested)
    python3 -I <this file> verify --root DIR --attestation REL   (the CI step)

Why the file set is computed, not hand-listed
    A hand list drifts silently: someone adds `app/rerank.py`, imports it
    from the graph, and forgets the list. CI then passes a change that was
    never evaluated. Here the set is the transitive import closure of the
    app's eval entry modules, read from the source by `ast` (nothing is
    imported), restricted to the first-party packages under the root. A new
    module joins the set the moment something on the eval path imports it.

Failure model
    - Invisible inputs. A dynamic import (`importlib.import_module(name)`) or
      a file read at runtime is not an import statement, so the closure
      cannot see it. That is why the app declares its data and lock files
      as `extra_files`, and why a whole directory can be declared (`data/`):
      a new file in it is covered without anyone remembering to list it.
    - Over-inclusion is the safe direction. Lazy imports inside functions,
      `TYPE_CHECKING` blocks and `try: import` all count, so the set can be
      a little larger than what one run executes. The cost is an extra rerun,
      never a missed change.
    - A relative import raises: it carries no package name, so following it
      would need a second resolver. tests/test_architecture.py bans them.

Trust model
    This is a process control, not a security control. CI trusts the local
    run: anyone who can push can also edit the attestation, this file or the
    workflow. What it buys is that an honest change to the eval path cannot
    merge by accident without a passing run. Branch protection and review
    are what stop a dishonest one. At scale a GPU runner replaces the
    attestation with a real run in CI.

Why this file stands alone
    CI runs it as `python3 -I wyld3rness/eval/attest.py` with no `uv sync`, so
    it imports the standard library only and nothing from the harness. It is
    the leaf of the harness: report.py and cli.py import from it, never the
    other way round. `-I` keeps the script's own folder off `sys.path`, so a
    planted `json.py` next to it cannot shadow the real module. The verify
    step reads and hashes files; it never imports or runs the code it hashes.

    The leaf also holds the rules that make a run traceable (not a control,
    a clean tree, a known commit and judge), because the baseline and the
    attestation must refuse the same runs for the same reasons.

Privacy and the public log
    The attestation is committed to a public repository. It holds only
    paths relative to the root, hashes, model names, digests, the git SHA
    and UTC timestamps: never an absolute path, a user or a host name. In CI
    the file is input from whoever pushed, so every field is type-checked
    and every line printed is stripped of control characters, so a crafted
    value cannot start a GitHub Actions workflow command (`::error::`).
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Final, NamedTuple, TextIO, TypeGuard

# 2 added `source_roots`: where first-party packages live (a `src/` layout,
# or several in a workspace). A version 1 file names none, so it is refused
# with a message instead of being read as "the root".
ATTESTATION_VERSION: Final = 2
VERDICT_PASS: Final = "pass"
# Provenance keys copied into the attestation: what identifies the run. The
# app's own model fields are added by suffix (`small_model`, `large_model`),
# because the harness cannot name them. An allowlist, not "everything": the
# file is public, and an app's describe() could one day carry a host URL.
RUN_KEYS: Final = ("judge_model", "judge_digest", "git_sha", "timestamp_utc")
MODEL_KEY_SUFFIX: Final = "_model"
# What provenance holds when git or Ollama could not answer.
UNKNOWN: Final = (None, "unknown")
# Inside a declared directory, files that document it rather than feed the run.
DIRECTORY_SKIPS: Final = frozenset({"README.md"})

# An app's own older way of marking a control run in provenance: key -> the
# value that means "this was a control". Reports written before the harness had
# `negative_control` carry the app's flag instead, and the harness cannot know
# its name. The app passes it in. Such keys are also kept out of a baseline,
# because they describe one run, like `negative_control` does.
LegacyControl = Mapping[str, Any]
NO_LEGACY_CONTROL: LegacyControl = MappingProxyType({})


class FingerprintError(ValueError):
    """The inputs cannot be fingerprinted: a missing module or file, or an unsupported import.

    Distinct from a mismatch. A mismatch says "rerun the gate"; this says the
    declared inputs or the source are broken, so no comparison was possible.
    """


class AttestationRefused(Exception):
    """An attestation must not be written for this run. The message lists every reason."""


class AttestationInvalid(ValueError):
    """The attestation file is malformed: not JSON, or a field of the wrong type."""


class AttestationMissing(AttestationInvalid):
    """There is no attestation file. Not attested yet, rather than broken."""


class Inputs(NamedTuple):
    """What an app declares as answer-affecting: entry modules, extra files and source roots.

    `entry_modules` are dotted names (`findyfixy.evals.answers`). `extra_files` are
    paths relative to the root, in POSIX form (`uv.lock`). An entry ending in
    `/` (`data/`) is a directory: every file under it counts, except hidden
    ones and a README, and the set is listed again at every check.
    `source_roots` are the folders, relative to the root, that hold first-party
    top-level packages: `src` for the harness and `examples/findyfixy/src` for
    an app in the same workspace. The closure follows imports across all of
    them, so a harness file on the app's eval path is attested too.
    """

    entry_modules: tuple[str, ...]
    extra_files: tuple[str, ...]
    source_roots: tuple[str, ...] = (".",)


class Fingerprint(NamedTuple):
    """The digest over every answer-affecting file, and each file's own hash.

    `files` maps a POSIX path relative to the root to that file's SHA-256.
    The per-file hashes are what let `verify` name the file that changed.
    """

    digest: str
    files: Mapping[str, str]


# ---- Helpers shared with report.py and cli.py ---------------------------------


def utc_now() -> datetime:
    """The current time in UTC. Injected as a default so tests can replace it."""
    return datetime.now(UTC)


def file_sha256(path: Path) -> str:
    """SHA-256 hex of a file's bytes. No newline or encoding normalisation.

    Bytes, not text: a CRLF checkout or a re-encoded file is a different
    input to the program, so it must be a different hash.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON to a temp file in the same directory, then rename over `path`.

    The rename is atomic on one filesystem, so a reader or a crash never sees
    a half-written file. The temp file is removed on failure. Keys are sorted
    so a rewrite of the same content is byte-identical and diffs stay small.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Not a `with` on the call itself: the except branch needs `handle.name` to clean up.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


# ---- Traceability: the rules the baseline and the attestation share ------------


def is_negative_control(provenance: Mapping[str, Any], legacy_control: LegacyControl = NO_LEGACY_CONTROL) -> bool:
    """True for a run built to fail the gate.

    `negative_control` is the harness flag. A `legacy_control` key holding its
    marking value is the same fact in a report written before the flag existed.
    The key must be present and equal that value: a missing key is not a control.
    """
    if provenance.get("negative_control") is True:
        return True
    return any(key in provenance and provenance[key] == value for key, value in legacy_control.items())


def traceability_problems(
    provenance: Mapping[str, Any], legacy_control: LegacyControl = NO_LEGACY_CONTROL
) -> list[str]:
    """Why a run cannot stand for its code later. Empty means traceable. Pure.

    One rule set for both the baseline and the attestation. `git_dirty` must
    be exactly False: True means uncommitted changes, and None means git
    could not tell. Both are refused, because in neither case does the SHA
    describe the code that ran.
    """
    reasons = []
    if is_negative_control(provenance, legacy_control):
        # A control exists to fail. Freezing or attesting it would make failure the standard.
        reasons.append("this run was a negative control, built to fail the gate")
    if provenance.get("git_dirty") is not False:
        reasons.append("the working tree had uncommitted changes, or git could not tell, so the code is not traceable")
    for key in ("judge_digest", "git_sha"):
        if provenance.get(key) in UNKNOWN:
            reasons.append(f"{key} is unknown, so the run could not be traced or compared later")
    return reasons


# ---- Fingerprint --------------------------------------------------------------


def _plain(rel: str, what: str) -> PurePosixPath:
    """`rel` as a relative POSIX path, or `FingerprintError`.

    Refuses an absolute path or a `..` step: the attestation is public and
    holds relative paths only, and in CI these come from a pushed file, so
    they are validated, not followed.
    """
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or not rel or not rel.isprintable():
        raise FingerprintError(f"{what} {rel!r} must be a plain path relative to the root")
    return pure


def first_party(root: Path, source_roots: Sequence[str] = (".",)) -> dict[str, Path]:
    """Top-level names that resolve to code under the source roots, each with the folder that holds it.

    Read from the tree, so a new package counts without a code change.
    Everything else (the standard library, third-party packages) is pinned
    by `uv.lock`, which the app declares as an extra file. A name found in
    two source roots is an error: which one runs would depend on the order
    of `sys.path`, and the fingerprint must not guess.
    """
    found: dict[str, Path] = {}
    for rel in source_roots:
        base = root.joinpath(*_plain(rel, "source root").parts)
        if not base.is_dir():
            raise FingerprintError(f"source root {rel} is missing")
        names = {p.parent.name for p in base.glob("*/__init__.py")} | {p.stem for p in base.glob("*.py")}
        for name in names:
            if name in found:
                raise FingerprintError(f"top-level module {name} is in two source roots")
            found[name] = base
    return found


def module_file(base: Path, name: str) -> Path | None:
    """The file that defines module `name` in the source root `base`, or None.

    A package wins over a same-named module, as it does for Python's own
    finder when both sit in one directory.
    """
    path = base.joinpath(*name.split("."))
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    return None


def _relative(root: Path, path: Path) -> str:
    """`path` relative to `root`, in POSIX form. The only path shape that is written down."""
    return path.relative_to(root).as_posix()


def _inside(root: Path, rel: str, path: Path) -> Path:
    """`path`, unless a symlink takes it out of the root. Then `FingerprintError`.

    A checkout from a fork could link `app/x.py` to a file elsewhere on the
    runner. Hashing it would put a stranger's file into a public message.
    """
    if not path.resolve().is_relative_to(root.resolve()):
        raise FingerprintError(f"{rel} resolves outside the root")
    return path


def imports_of(root: Path, path: Path) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Every absolute import in a file, at any depth, as (module, imported names).

    `import a.b` gives ("a.b", ()). `from a import b, c` gives ("a", ("b", "c")),
    because `b` may be a submodule that runs on import. A relative import
    raises `FingerprintError`. So does a syntax error: a file that does not
    parse cannot be shown to be on or off the eval path.
    """
    try:
        tree = ast.parse(path.read_bytes(), filename=_relative(root, path))
    except SyntaxError as err:
        raise FingerprintError(f"{_relative(root, path)}:{err.lineno}: does not parse: {err.msg}") from err
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module is None:
                raise FingerprintError(
                    f"{_relative(root, path)}:{node.lineno}: relative import. "
                    "The fingerprint follows absolute imports only."
                )
            yield node.module, tuple(a.name for a in node.names if a.name != "*")


def import_closure(root: Path, entry_modules: Sequence[str], source_roots: Sequence[str] = (".",)) -> list[Path]:
    """Every first-party file the entry modules can execute through imports. Sorted.

    Importing `a.b.c` runs `a/__init__.py` and `a/b/__init__.py` first, so
    each parent package is added too. A first-party import that resolves to
    no file raises: either the source is broken or the root is wrong, and
    a silent skip would shrink the fingerprint.
    """
    packages = first_party(root, source_roots)
    seen: dict[str, Path] = {}
    queue: list[Path] = []

    def add(name: str) -> None:
        """Queue module `name` and its parent packages. The caller checked it is first-party."""
        parts = name.split(".")
        if not all(part.isidentifier() for part in parts):
            # Names from source are identifiers by syntax. An entry module read
            # from a pushed attestation is not, and `a.b/../../x` must not
            # become a path out of the root.
            raise FingerprintError(f"module name {name!r} is not a dotted Python name")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix in seen:
                continue
            found = module_file(packages[parts[0]], prefix)
            if found is None:
                raise FingerprintError(f"cannot find first-party module {prefix} under the root")
            seen[prefix] = _inside(root, prefix, found)  # checked before it is parsed
            queue.append(found)

    for entry in entry_modules:
        if entry.split(".")[0] not in packages:
            raise FingerprintError(f"entry module {entry!r} is not a first-party module under the root")
        add(entry)
    while queue:
        for module, names in imports_of(root, queue.pop()):
            if module.split(".")[0] not in packages:
                continue  # the standard library or a locked package: no file to probe
            add(module)
            for name in names:
                # `from pkg import mod` imports the submodule pkg.mod;
                # `from pkg.mod import name` imports a name. Only a
                # name that resolves to a file is a module.
                if module_file(packages[module.split(".")[0]], f"{module}.{name}") is not None:
                    add(f"{module}.{name}")
    return sorted(set(seen.values()))


def _declared(root: Path, rel: str) -> dict[str, Path]:
    """The files one declared extra stands for: a file, or every file under a `dir/`.

    Refuses an absolute path or a `..` step: the attestation is public and
    holds relative paths only, and in CI the list comes from a pushed file,
    so it is validated, not followed. In a directory, hidden files (an
    editor's or the OS's, git-ignored) and `DIRECTORY_SKIPS` are left out.
    """
    pure = _plain(rel, "declared file")
    path = root.joinpath(*pure.parts)
    if rel.endswith("/"):
        if not path.is_dir():
            raise FingerprintError(f"declared directory {rel} is missing")
        found = {}
        for child in sorted(path.rglob("*")):
            parts = child.relative_to(root).parts
            if child.is_file() and child.name not in DIRECTORY_SKIPS and not any(p.startswith(".") for p in parts):
                found[_relative(root, child)] = child
        return found
    if not path.is_file():
        raise FingerprintError(f"declared file {rel} is missing")
    return {pure.as_posix(): path}


def digest_of(files: Mapping[str, str]) -> str:
    """SHA-256 over `<file sha>  <path>` lines, sorted by path. Order-independent by construction."""
    lines = "".join(f"{files[rel]}  {rel}\n" for rel in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def fingerprint(
    root: Path, entry_modules: Sequence[str], extra_files: Sequence[str], source_roots: Sequence[str] = (".",)
) -> Fingerprint:
    """The fingerprint of every answer-affecting file under `root`.

    The set is the import closure of `entry_modules`, followed through the
    packages in `source_roots`, plus `extra_files`. Raises `FingerprintError`
    when an input cannot be found or parsed.
    """
    paths = {_relative(root, p): p for p in import_closure(root, entry_modules, source_roots)}
    for rel in extra_files:
        paths.update(_declared(root, rel))
    files = {rel: file_sha256(_inside(root, rel, path)) for rel, path in sorted(paths.items())}
    return Fingerprint(digest=digest_of(files), files=MappingProxyType(files))


def differences(attested: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """Which files changed, were added or were removed since the attestation, by path."""
    changed = sorted(rel for rel in attested.keys() & current.keys() if attested[rel] != current[rel])
    added = sorted(current.keys() - attested.keys())
    removed = sorted(attested.keys() - current.keys())
    return (
        [f"changed: {rel}" for rel in changed]
        + [f"added: {rel}" for rel in added]
        + [f"removed: {rel}" for rel in removed]
    )


# ---- Writing ------------------------------------------------------------------


def refusals(
    report: Mapping[str, Any],
    gate_problems: Sequence[str],
    fingerprint: Fingerprint,
    *,
    expected_case_ids: Collection[str],
    started: Fingerprint | None = None,
) -> list[str]:
    """Every reason this run must not be attested. Empty means it may be.

    Only a full, comparable, real run that passed the gate on a clean tree
    can stand for the files. Each rule closes one way to attest code that
    never passed:
    - a failed gate (which also covers a baseline that is not comparable)
    - the traceability rules (`traceability_problems`), shared with the baseline
    - a subset: fewer cases than the golden set passes more easily
    - files that changed between the start and the end of the run
    """
    reasons = []
    if gate_problems:
        reasons.append(f"the gate did not pass ({len(gate_problems)} problem(s))")
    reasons += traceability_problems(report["provenance"])
    ran, expected = {c["id"] for c in report["cases"]}, set(expected_case_ids)
    if ran != expected:
        detail = f"{len(ran & expected)} of {len(expected)} golden cases ran"
        if ran - expected:
            detail += f", plus {len(ran - expected)} not in the golden set"
        reasons.append(f"not a full run: {detail}")
    if started is not None and started.digest != fingerprint.digest:
        reasons.append(
            "answer-affecting files changed during the run: " + "; ".join(differences(started.files, fingerprint.files))
        )
    return reasons


def run_identity(provenance: Mapping[str, Any]) -> dict[str, str]:
    """The provenance fields that identify the run: judge, models, git SHA, time.

    Only string values are kept, so the file stays a flat map of strings
    and an app key that ends in `_model` but holds a structure is left out.
    """
    keys = [*RUN_KEYS, *(k for k in provenance if k.endswith(MODEL_KEY_SUFFIX))]
    return {key: provenance[key] for key in keys if isinstance(provenance.get(key), str)}


def write_attestation(
    path: Path,
    report: Mapping[str, Any],
    gate_problems: Sequence[str],
    fingerprint: Fingerprint,
    *,
    inputs: Inputs,
    report_file: Path,
    expected_case_ids: Collection[str],
    started: Fingerprint | None = None,
    now: Callable[[], datetime] = utc_now,
) -> dict[str, Any]:
    """Write the attestation for a passing run, atomically, and return what was written.

    Raises `AttestationRefused` with every reason from `refusals`; nothing is
    written then, and an older attestation is left alone (its own files still
    say what it covers). `report_file` is the saved report: only its name and
    hash are recorded, never its path, because reports are git-ignored and
    the name is enough to find it on the machine that ran it.
    """
    reasons = refusals(report, gate_problems, fingerprint, expected_case_ids=expected_case_ids, started=started)
    if reasons:
        raise AttestationRefused("; ".join(reasons))
    payload = {
        "version": ATTESTATION_VERSION,
        "verdict": VERDICT_PASS,
        "created_utc": now().astimezone(UTC).isoformat(timespec="seconds"),
        "fingerprint": fingerprint.digest,
        "inputs": {
            "entry_modules": list(inputs.entry_modules),
            "extra_files": list(inputs.extra_files),
            "source_roots": list(inputs.source_roots),
        },
        "files": dict(fingerprint.files),
        "report": {"name": report_file.name, "sha256": file_sha256(report_file)},
        "run": run_identity(report["provenance"]),
    }
    atomic_write_json(path, payload)
    return payload


# ---- Verifying ------------------------------------------------------------------


def _is_str_map(value: object) -> TypeGuard[dict[str, str]]:
    """True for a JSON object whose keys and values are all strings."""
    return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())


def _is_str_list(value: object) -> TypeGuard[list[str]]:
    """True for a JSON array of strings."""
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def load_attestation(path: Path) -> dict[str, Any]:
    """Read and type-check every field an attestation is read for.

    Raises `AttestationMissing` when there is no file and `AttestationInvalid`
    for anything malformed. Checked here, at the boundary, because in CI the
    file comes from whoever pushed: a wrong type must become a message, not
    a traceback further down.
    """
    shown = path.as_posix()  # as the caller gave it: relative in CI
    if not path.is_file():
        raise AttestationMissing(f"no attestation at {shown}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise AttestationInvalid(f"{shown} is not valid JSON: {err}") from err
    if not isinstance(doc, dict):
        raise AttestationInvalid(f"{shown} must hold a JSON object")
    if type(doc.get("version")) is int and doc["version"] != ATTESTATION_VERSION:
        # Checked first: an older format fails on its version, not on a field it never had.
        raise AttestationInvalid(f"{shown}: version {doc['version']}, expected {ATTESTATION_VERSION}")
    inputs, report = doc.get("inputs"), doc.get("report")
    checks = (
        ("version", type(doc.get("version")) is int, "an integer"),
        ("verdict", isinstance(doc.get("verdict"), str), "a string"),
        ("created_utc", isinstance(doc.get("created_utc"), str), "a string"),
        ("fingerprint", isinstance(doc.get("fingerprint"), str), "a string"),
        ("files", _is_str_map(doc.get("files")), "an object of paths to hashes"),
        (
            "inputs",
            isinstance(inputs, dict)
            and _is_str_list(inputs.get("entry_modules"))
            and _is_str_list(inputs.get("extra_files"))
            and _is_str_list(inputs.get("source_roots")),
            "an object listing entry_modules, extra_files and source_roots",
        ),
        ("report", _is_str_map(report) and {"name", "sha256"} <= set(report), "an object with name and sha256"),
        ("run", _is_str_map(doc.get("run")), "an object of strings"),
    )
    for field, ok, expected in checks:
        if not ok:
            raise AttestationInvalid(f"{shown}: '{field}' must be {expected}")
    return doc


def attested_inputs(doc: Mapping[str, Any]) -> Inputs:
    """The inputs the attestation was made over, from a loaded document."""
    inputs = doc["inputs"]
    return Inputs(tuple(inputs["entry_modules"]), tuple(inputs["extra_files"]), tuple(inputs["source_roots"]))


def check(doc: Mapping[str, Any], fingerprint: Fingerprint) -> list[str]:
    """Problems between a loaded attestation and the current fingerprint. Empty means attested. Pure."""
    problems = []
    if doc["verdict"] != VERDICT_PASS:
        problems.append(f"the attested verdict is {doc['verdict']!r}, not {VERDICT_PASS!r}")
    diff = differences(doc["files"], fingerprint.files)
    problems += diff
    if not diff and doc["fingerprint"] != fingerprint.digest:
        # Same files, different digest: the digest field was edited by hand.
        problems.append("the recorded fingerprint does not match its own file hashes")
    return problems


def verify(path: Path, fingerprint: Fingerprint) -> list[str]:
    """Every reason the attestation at `path` does not cover `fingerprint`. Empty means pass."""
    try:
        doc = load_attestation(path)
    except AttestationInvalid as err:
        return [str(err)]
    return check(doc, fingerprint)


# ---- Command line ---------------------------------------------------------------

EXIT_ATTESTED: Final = 0
EXIT_NOT_ATTESTED: Final = 1
EXIT_ERROR: Final = 2
MAX_LINE: Final = 300


def printable(text: str, limit: int = MAX_LINE) -> str:
    """`text` safe for a CI log: one line, no control characters, `::` broken, capped. Pure.

    GitHub Actions treats a log line that starts with `::` as a command
    (`::error::`, `::stop-commands::`, `::add-mask::`). Every string printed
    here can carry values from a pushed attestation, so newlines (which
    would start a new line) and other control characters are dropped, and
    `::` is broken up wherever it appears.
    """
    clean = "".join(ch for ch in text if ch.isprintable()).replace("::", ": :")
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def _say(text: str, stream: TextIO | None = None) -> None:
    """Print one sanitised line. The command line prints through nothing else."""
    print(printable(text), file=stream or sys.stdout)


def _parser() -> argparse.ArgumentParser:
    """`verify --root DIR --attestation REL [--list]`."""
    parser = argparse.ArgumentParser(prog="attest", description="Check the eval attestation against the checkout.")
    commands = parser.add_subparsers(dest="command", required=True)
    check_cmd = commands.add_parser("verify", help="exit 0 when the attestation covers the current files")
    check_cmd.add_argument(
        "--root", type=Path, required=True, help="the workspace folder the attested paths are relative to"
    )
    check_cmd.add_argument("--attestation", required=True, help="the attestation file, relative to --root")
    check_cmd.add_argument("--list", action="store_true", help="also print every attested file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the attestation. Exit 0 attested, 1 not attested, 2 unreadable input or bad usage.

    1 means "rerun the gate": no attestation yet, a failed verdict, or files
    that differ. 2 means something is broken: a malformed attestation, a
    declared file that is gone, a module that does not parse.

    The inputs come from the attestation itself, so the app declares them
    in one place (its eval entry point) and CI needs no copy of the list. A
    change to that declaration changes the entry module's own hash, so it
    cannot slip past either.
    """
    args = _parser().parse_args(argv)
    root: Path = args.root
    if not root.is_dir():
        _say(f"attest: error: root {root} is not a directory", sys.stderr)
        return EXIT_ERROR
    rerun = "Run `make eval-gate` on a clean tree and commit the attestation it writes."
    try:
        doc = load_attestation(root / args.attestation)
        current = fingerprint(root, *attested_inputs(doc))
    except AttestationMissing as err:
        _say(f"attest: FAIL: {err}. {rerun}")
        return EXIT_NOT_ATTESTED
    except (AttestationInvalid, FingerprintError) as err:
        _say(f"attest: error: {err}", sys.stderr)
        return EXIT_ERROR
    problems = check(doc, current)
    for problem in problems:
        _say(f"attest: FAIL: {problem}")
    if problems:
        _say(f"attest: not attested. {rerun}")
        return EXIT_NOT_ATTESTED
    run = doc["run"]
    _say(
        f"attest: pass. {len(current.files)} answer-affecting files match the run of "
        f"{run.get('timestamp_utc')} at {run.get('git_sha')}, fingerprint {current.digest[:12]}"
    )
    if args.list:
        for rel in current.files:
            _say(f"attest:   {rel}")
    return EXIT_ATTESTED


if __name__ == "__main__":
    sys.exit(main())
