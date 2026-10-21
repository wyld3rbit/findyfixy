"""Block PII and secrets from entering this repo or its history.

What it does
    Scans text for things that must never be committed, and exits non-zero if it
    finds any. Three entry points, each used by a git hook or by `make scan`:

        --staged            what is about to be committed (pre-commit hook)
        --pre-push REMOTE   every commit about to be pushed, with metadata
                            (pre-push hook; reads git's ref list on stdin)
        --tree              every tracked and untracked file on disk
        --history [RANGE]   every commit, its metadata and the lines it added

The rule it enforces
    No PII and no secrets, ever, even though the remote is private. A private
    repo can be made public or cloned, and history cannot be un-published.

What it looks for
    local_path      /Users/<name>, /home/<name>, C:\\Users\\<name>
    identity        this machine's own username, home directory and git email,
                    read at run time and never stored in the repo
    private_term    any word in `.private-terms` (git-ignored), such as a name
    email           any email address except the allowed ones (GitHub no-reply,
                    the co-author trailer, example domains, git@github.com)
    secret          private key blocks, cloud and token formats, and
                    `key = "long string"` assignments
    forbidden_file  `.env`, `.pem`, `.key`, credentials files (not `.env.example`)
    author          a commit whose author or committer email is not a GitHub
                    no-reply address

It never prints what it matched
    Output is `path:line: rule`. Printing the match would copy the leak into the
    terminal, shell history and any log that captures it. The line number is
    enough to find it.

Limits (read before trusting it)
    It is a net, not proof. It matches patterns. It cannot know that a word in
    prose is a person's name unless that word is in `.private-terms`, and it
    cannot recognise a secret in a format it has no pattern for. It runs only if
    the hooks are installed (`make hooks`), and a commit made with
    `--no-verify` skips it. Run `make scan` to check everything on demand.

What changes at scale
    A team would run the same checks in CI as well, because a local hook can be
    skipped or missing. A real secret scanner (entropy checks, a verified
    pattern library) would replace the hand-written patterns here.
"""

import argparse
import getpass
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# ---- what counts as sensitive ----------------------------------------------

# A real directory under a user's home. `<name>` style placeholders do not match
# because angle brackets are not in the character class. The names below are
# generic CI or placeholder accounts, not people.
_PATH = re.compile(r"(?:/Users|/home)/([A-Za-z0-9._-]+)|[A-Za-z]:\\+Users\\+([A-Za-z0-9._-]+)")
_GENERIC_ACCOUNTS = {"runner", "user", "username", "you", "me", "name", "shared", "ubuntu"}

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_NO_REPLY_DOMAIN = "users.noreply.github.com"
_ALLOWED_EMAILS = {"git@github.com", "noreply@anthropic.com"}
_ALLOWED_DOMAINS = {_NO_REPLY_DOMAIN, "example.com", "example.net", "example.org"}

_SECRETS = [
    re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    # key = "long string": a credential-looking name assigned a quoted value.
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|passw(?:or)?d|auth)\b\s*[:=]\s*"
        r"""['"][^'"\s]{12,}['"]"""
    ),
]

_FORBIDDEN_FILE = re.compile(
    r"(?:^|/)(?:\.env(?:\..+)?|[^/]+\.(?:pem|key|p12|pfx)|id_(?:rsa|ed25519)[^/]*"
    r"|credentials[^/]*\.json)$"
)
_FORBIDDEN_OK = re.compile(r"(?:^|/)\.env\.example$")

_MAX_BYTES = 5_000_000  # a file larger than this is almost certainly not source


@dataclass(frozen=True)
class Identity:
    """This machine's own details. Read at run time, never stored in the repo."""

    username: str = ""
    home: str = ""
    email: str = ""


@dataclass(frozen=True)
class Finding:
    """One problem. `where` is a path, a commit label, or 'metadata'.

    Holds the rule and the line number, never the matched text.
    """

    where: str
    line: int
    rule: str

    def __str__(self) -> str:
        loc = f"{self.where}:{self.line}" if self.line else self.where
        return f"{loc}: {self.rule}"


def email_allowed(email: str) -> bool:
    """True for addresses that are safe to commit. Pure."""
    email = email.lower()
    return email in _ALLOWED_EMAILS or email.rsplit("@", 1)[-1] in _ALLOWED_DOMAINS


def is_no_reply(email: str) -> bool:
    """True for a GitHub no-reply address, the only identity a commit may carry."""
    return email.lower().endswith("@" + _NO_REPLY_DOMAIN)


def scan_text(text: str, identity: Identity, terms: Iterable[str] = ()) -> list[tuple[int, str]]:
    """Return (line number, rule) for each problem in `text`. Pure.

    One rule is reported at most once per line. The matched text is not returned.
    """
    sensitive_email = identity.email if identity.email and not email_allowed(identity.email) else ""
    word_terms = [
        re.compile(rf"(?i)(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])")
        for t in terms
        if t.strip()
    ]
    # A short username would match ordinary words, so it is caught through the
    # path rule instead.
    user = re.compile(rf"(?i)\b{re.escape(identity.username)}\b") if len(identity.username) >= 4 else None
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        rules: set[str] = set()
        for m in _PATH.finditer(line):
            if (m.group(1) or m.group(2)).lower() not in _GENERIC_ACCOUNTS:
                rules.add("local_path")
        if identity.home and len(identity.home) > 3 and identity.home in line:
            rules.add("identity")
        if sensitive_email and sensitive_email in line:
            rules.add("identity")
        if user and user.search(line):
            rules.add("identity")
        if any(t.search(line) for t in word_terms):
            rules.add("private_term")
        if any(not email_allowed(e) for e in _EMAIL.findall(line)):
            rules.add("email")
        if any(p.search(line) for p in _SECRETS):
            rules.add("secret")
        found += [(number, r) for r in sorted(rules)]
    return found


def forbidden_file(path: str) -> bool:
    """True for a path that should never be committed. Pure."""
    return bool(_FORBIDDEN_FILE.search(path)) and not _FORBIDDEN_OK.search(path)


def scan_metadata(author: str, committer: str, message: str, identity: Identity,
                  terms: Iterable[str] = ()) -> list[str]:
    """Rules broken by a commit's identity and message. Pure."""
    rules = []
    if not is_no_reply(author) or not is_no_reply(committer):
        rules.append("author")
    rules += sorted({r for _, r in scan_text(message, identity, terms)})
    return rules


# ---- reading from git and the disk -----------------------------------------


def git(*args: str, check: bool = True) -> str:
    """Run git and return stdout as text."""
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=check)
    return out.stdout


def runtime_identity() -> Identity:
    """This machine's username, home directory and git email."""
    try:
        username = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name is fine
        username = ""
    return Identity(
        username=username,
        home=str(Path.home()),
        email=git("config", "user.email", check=False).strip(),
    )


def repo_root() -> Path:
    return Path(git("rev-parse", "--show-toplevel").strip())


def load_terms(root: Path) -> list[str]:
    """Words from `.private-terms` (git-ignored). Missing file means none."""
    path = root / ".private-terms"
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def decode(data: bytes) -> str | None:
    """Text of a file, or None if it is binary or too large to be source."""
    if len(data) > _MAX_BYTES or b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_files(items: Iterable[tuple[str, bytes]], identity: Identity,
               terms: list[str]) -> list[Finding]:
    """Scan (path, content) pairs: the path rule, then each line of the text."""
    out: list[Finding] = []
    for path, data in items:
        if forbidden_file(path):
            out.append(Finding(path, 0, "forbidden_file"))
        text = decode(data)
        if text is not None:
            out += [Finding(path, n, r) for n, r in scan_text(text, identity, terms)]
    return out


def staged_items() -> Iterator[tuple[str, bytes]]:
    """Staged files, read from the INDEX, not the working tree.

    The working copy can differ from what is staged. What will be committed is
    the index, so that is what is scanned.
    """
    names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    for name in filter(None, names.split("\0")):
        blob = subprocess.run(["git", "show", f":{name}"], capture_output=True)
        yield name, blob.stdout


def tree_items(root: Path) -> Iterator[tuple[str, bytes]]:
    """Every tracked and untracked (not ignored) file as it is on disk."""
    tracked = git("ls-files", "-z").split("\0")
    untracked = git("ls-files", "-z", "--others", "--exclude-standard").split("\0")
    for name in filter(None, tracked + untracked):
        path = root / name
        if path.is_file():
            yield name, path.read_bytes()


def added_lines(diff: str) -> Iterator[tuple[str, int, str]]:
    """(path, line number, text) for each line a diff added. Pure."""
    path, number = "", 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            number = int(m.group(1)) - 1 if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            number += 1
            yield path, number, line[1:]


def history_findings(rev_args: list[str], identity: Identity, terms: list[str]) -> list[Finding]:
    """Scan metadata and added lines of every commit in a revision range."""
    out: list[Finding] = []
    shas = git("rev-list", *rev_args).split()
    for sha in shas:
        label = f"commit {sha[:8]}"
        meta = git("log", "-1", "--format=%ae%x00%ce%x00%B", sha).split("\0", 2)
        author, committer, message = (meta + ["", "", ""])[:3]
        out += [Finding(label, 0, f"metadata {r}") for r in
                scan_metadata(author.strip(), committer.strip(), message, identity, terms)]
        diff = git("show", "--no-color", "--format=", "-U0", sha)
        seen = set()
        for path, number, text in added_lines(diff):
            if forbidden_file(path) and path not in seen:
                seen.add(path)
                out.append(Finding(f"{label} {path}", 0, "forbidden_file"))
            out += [Finding(f"{label} {path}", number, r)
                    for n, r in scan_text(text, identity, terms)]
    return out


def pending_identity_findings() -> list[Finding]:
    """The identity the next commit would carry (pre-commit only)."""
    out = []
    for var in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        ident = git("var", var, check=False)
        m = re.search(r"<([^>]*)>", ident)
        if m and not is_no_reply(m.group(1)):
            out.append(Finding("metadata", 0, f"{var.split('_')[1].lower()} email is not a GitHub no-reply address"))
    return out


def push_ranges(stdin: str, remote: str) -> list[list[str]]:
    """Revision ranges a push would publish, from git's pre-push stdin. Pure-ish.

    Each line is `local_ref local_sha remote_ref remote_sha`. A new branch has an
    all-zero remote sha, so scan everything not already on that remote. A delete
    pushes nothing.
    """
    zero = "0" * 40
    ranges = []
    for line in stdin.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == zero:
            continue
        local, remote_sha = parts[1], parts[3]
        ranges.append([local, "--not", f"--remotes={remote}"] if remote_sha == zero
                      else [f"{remote_sha}..{local}"])
    return ranges


# ---- command line -----------------------------------------------------------


def report(findings: list[Finding]) -> int:
    """Print findings and return the exit code."""
    if not findings:
        print("scan_private: clean.")
        return 0
    for f in findings:
        print(f"scan_private: {f}", file=sys.stderr)
    print(
        f"scan_private: {len(findings)} problem(s). Nothing was printed from the matches, "
        "on purpose: open the file at the line shown. Remove the data or use a "
        "placeholder. No PII or secrets, ever.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--tree", action="store_true")
    mode.add_argument("--history", nargs="*", metavar="REV", default=None)
    mode.add_argument("--pre-push", metavar="REMOTE")
    args = ap.parse_args(argv)
    root, identity = repo_root(), runtime_identity()
    terms = load_terms(root)
    if args.staged:
        findings = scan_files(staged_items(), identity, terms) + pending_identity_findings()
    elif args.tree:
        findings = scan_files(tree_items(root), identity, terms)
    elif args.history is not None:
        findings = history_findings(args.history or ["--all"], identity, terms)
    else:
        findings = []
        for rng in push_ranges(sys.stdin.read(), args.pre_push):
            findings += history_findings(rng, identity, terms)
    return report(findings)


if __name__ == "__main__":
    sys.exit(main())
