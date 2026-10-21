"""Tests for the privacy scanner (tools/scan_private.py) and its git hooks.

What these tests cover
    That each rule fires on what it should and stays quiet on what is allowed;
    that the scanner reads what will be COMMITTED (the index), not the working
    copy; that it never prints what it matched; and that the real hook script
    stops a real `git commit` and lets a clean one through.

Why the fake data is built at run time
    This file is itself scanned by the scanner. A literal fake path or key here
    would trip it, and would also be exactly the thing the rule forbids. Each
    sample is assembled from pieces, so no sample appears in the source.

What a staff+ reader should notice
    - A scanner that never fails proves nothing. Every rule has a test that
      makes it fire, and the mutation record in CHANGELOG.md says each was broken
      once to confirm a test noticed.
    - The "does not print the match" test is the one that stops a leak from
      being copied into terminals and logs.
    - The hook test runs the committed hook script in a scratch repo. A scanner
      that is correct but not wired in protects nothing.
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "scan_private.py"
HOOKS = REPO / ".githooks"

spec = importlib.util.spec_from_file_location("scan_private", TOOL)
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

# ---- samples, assembled so this file contains none of them ------------------

SOMEONE = "some" + "one"
PATH = "/" + "Users" + "/" + SOMEONE + "/project"
WIN_PATH = "C:" + "\\" + "Users" + "\\" + SOMEONE + "\\x"
AWS = "AKIA" + "IOSFODNN7EXAMPLE"
GH = "gh" + "p_" + "a" * 36
SK = "sk" + "-" + "b" * 24
PEM = "-----" + "BEGIN " + "RSA PRIVATE KEY" + "-----"
ASSIGN = 'api_key = "' + "c" * 20 + '"'
EMAIL = SOMEONE + "@" + "corp" + ".test"
NOREPLY = "1234+" + SOMEONE + "@users.noreply.github.com"
NOBODY = sp.Identity()


def rules(text, identity=NOBODY, terms=()):
    return {r for _, r in sp.scan_text(text, identity, terms)}


# ---- the rules ---------------------------------------------------------------


@pytest.mark.parametrize("text", [PATH, WIN_PATH, f"cd {PATH} && ls"])
def test_a_local_user_path_is_flagged(text):
    assert rules(text) == {"local_path"}


@pytest.mark.parametrize(
    "text",
    ["/Users/<name>/x", "/home/<user>/x", "/home/runner/work", "a relative/path/x", "code/.venv"],
)
def test_placeholders_and_generic_paths_are_not_flagged(text):
    assert rules(text) == set()


@pytest.mark.parametrize("text", [EMAIL, f"contact: {EMAIL}."])
def test_an_ordinary_email_is_flagged(text):
    assert rules(text) == {"email"}


@pytest.mark.parametrize(
    "text",
    [NOREPLY, "Co-Authored-By: Claude <noreply@anthropic.com>", "a@example.com",
     "git@github.com:owner/repo.git", "@pytest.fixture", "uv add mcp>=1.0"],
)
def test_allowed_emails_and_lookalikes_are_not_flagged(text):
    assert rules(text) == set()


@pytest.mark.parametrize("secret", [AWS, GH, SK, PEM, ASSIGN])
def test_a_secret_shape_is_flagged(secret):
    assert rules(f"x {secret} y") == {"secret"}


@pytest.mark.parametrize(
    "text",
    ["hash = sha256:" + "a" * 64, 'name = "findyfixy"', "token_budget = 4096",
     'secret_name = "x"', "api_key = os.environ['KEY']"],
)
def test_ordinary_code_and_hashes_are_not_flagged_as_secrets(text):
    assert rules(text) == set()


def test_the_machines_own_username_home_and_email_are_flagged():
    ident = sp.Identity(username="zephyrine", home="/" + "srv/zephyrine", email=EMAIL)
    assert rules("login zephyrine ok", ident) == {"identity"}
    assert rules("at /srv/zephyrine/x", ident) == {"identity"}
    assert rules(f"mail {EMAIL}", ident) == {"email", "identity"}
    assert rules("zephyrinetown", ident) == set()          # whole word only


def test_a_short_username_is_left_to_the_path_rule_not_matched_as_a_word():
    ident = sp.Identity(username="ab", home="", email="")
    assert rules("a cab and a lab", ident) == set()


def test_a_noreply_git_email_is_not_treated_as_sensitive():
    ident = sp.Identity(email=NOREPLY)
    assert rules(f"co-author {NOREPLY}", ident) == set()


def test_private_terms_are_blocked_whole_word_and_case_insensitive():
    assert rules("hello Zephyrine!", terms=["zephyrine"]) == {"private_term"}
    assert rules("zephyrines", terms=["zephyrine"]) == set()
    assert rules("anything", terms=["", "  "]) == set()


def test_line_numbers_are_reported_and_each_rule_once_per_line():
    text = f"ok\n{PATH} {PATH}\nok\n{AWS}"
    assert sp.scan_text(text, NOBODY) == [(2, "local_path"), (4, "secret")]


@pytest.mark.parametrize(
    ("path", "expected"),
    [(".env", True), ("code/.env.local", True), ("k/server.pem", True), ("a/id_rsa", True),
     ("credentials-prod.json", True), (".env.example", False), ("code/app/env.py", False),
     ("notes/day2.md", False)],
)
def test_forbidden_files(path, expected):
    assert sp.forbidden_file(path) is expected


def test_metadata_rules():
    ok = sp.scan_metadata(NOREPLY, NOREPLY, "fine\n\nCo-Authored-By: C <noreply@anthropic.com>", NOBODY)
    assert ok == []
    assert "author" in sp.scan_metadata(EMAIL, NOREPLY, "fine", NOBODY)
    assert "author" in sp.scan_metadata(NOREPLY, EMAIL, "fine", NOBODY)
    assert "local_path" in sp.scan_metadata(NOREPLY, NOREPLY, f"saw {PATH}", NOBODY)


def test_binary_and_huge_files_are_skipped_not_crashed_on():
    assert sp.decode(b"\0\1\2") is None
    assert sp.decode(b"x" * (sp._MAX_BYTES + 1)) is None
    assert sp.decode("é".encode("latin-1")) is None
    assert sp.decode(b"plain") == "plain"


def test_added_lines_tracks_line_numbers_across_hunks():
    diff = "+++ b/f.py\n@@ -1,0 +5,2 @@\n+one\n+two\n@@ -9 +20 @@\n+three\n"
    assert list(sp.added_lines(diff)) == [("f.py", 5, "one"), ("f.py", 6, "two"), ("f.py", 20, "three")]


def test_push_ranges_for_a_new_branch_an_update_and_a_delete():
    zero, a, b = "0" * 40, "a" * 40, "b" * 40
    stdin = f"refs/heads/new {a} refs/heads/new {zero}\nrefs/heads/m {b} refs/heads/m {a}\nrefs/heads/x {zero} refs/heads/x {a}\n"
    assert sp.push_ranges(stdin, "origin") == [[a, "--not", "--remotes=origin"], [f"{a}..{b}"]]


# ---- against a real scratch git repository ----------------------------------

pytestmark_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def git(cwd, *args, env=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", NOREPLY)
    git(root, "config", "user.name", "T")
    return root


def scan(root, *args, home=None):
    env = {"PATH": __import__("os").environ["PATH"], "HOME": str(home or root.parent / "h"),
           "USER": "qq", "LOGNAME": "qq"}
    return subprocess.run(["python3", str(TOOL), *args], cwd=root, capture_output=True,
                          text=True, env=env)


def stage(root, name, text):
    (root / name).write_text(text)
    git(root, "add", name)


def test_a_clean_staged_file_passes(repo):
    stage(repo, "a.txt", "nothing private here\n")
    assert scan(repo, "--staged").returncode == 0


def test_a_staged_path_is_blocked_with_file_and_line_but_never_the_match(repo):
    stage(repo, "a.txt", f"fine\n{PATH}\n{AWS}\n")
    out = scan(repo, "--staged")
    shown = out.stdout + out.stderr
    assert out.returncode == 1
    assert "a.txt:2: local_path" in shown and "a.txt:3: secret" in shown
    assert SOMEONE not in shown and AWS not in shown and "AKIA" not in shown


def test_the_index_is_scanned_not_the_working_copy(repo):
    stage(repo, "a.txt", f"{PATH}\n")             # bad content staged
    (repo / "a.txt").write_text("clean now\n")     # working copy fixed but NOT staged
    assert scan(repo, "--staged").returncode == 1  # what will be committed is still bad
    git(repo, "add", "a.txt")
    assert scan(repo, "--staged").returncode == 0
    (repo / "a.txt").write_text(f"{PATH}\n")       # bad again, unstaged only
    assert scan(repo, "--staged").returncode == 0


def test_env_files_are_blocked_but_the_example_is_not(repo):
    stage(repo, ".env", "X=1\n")
    assert "forbidden_file" in scan(repo, "--staged").stderr
    git(repo, "rm", "-q", "--cached", ".env")
    stage(repo, ".env.example", "X=\n")
    assert scan(repo, "--staged").returncode == 0


def test_a_commit_identity_that_is_not_a_noreply_address_is_blocked(repo):
    git(repo, "config", "user.email", EMAIL)
    stage(repo, "a.txt", "fine\n")
    out = scan(repo, "--staged")
    assert out.returncode == 1 and "author" in out.stderr.lower() or "not a GitHub" in out.stderr


def test_the_private_terms_file_is_read_from_the_repo_root(repo):
    (repo / ".private-terms").write_text("# names\n\nzephyrine\n")
    stage(repo, "a.txt", "Dear Zephyrine\n")
    assert "private_term" in scan(repo, "--staged").stderr


def test_the_machines_own_home_directory_is_flagged_from_the_environment(repo, tmp_path):
    home = tmp_path / "srv" / "zephyrine"
    home.mkdir(parents=True)
    stage(repo, "a.txt", f"saved to {home}/x\n")
    assert "identity" in scan(repo, "--staged", home=home).stderr


def test_history_finds_a_bad_commit_and_a_bad_author(repo):
    (repo / "a.txt").write_text(f"{PATH}\n")
    git(repo, "add", "a.txt")
    git(repo, "-c", f"user.email={EMAIL}", "commit", "-q", "-m", "bad")
    out = scan(repo, "--history")
    assert out.returncode == 1
    assert "a.txt:1: local_path" in out.stderr and "metadata author" in out.stderr
    assert SOMEONE not in out.stdout + out.stderr


def test_a_clean_history_passes(repo):
    stage(repo, "a.txt", "fine\n")
    git(repo, "commit", "-q", "-m", "ok")
    assert scan(repo, "--history").returncode == 0


def test_the_tree_scan_reads_untracked_files_too(repo):
    (repo / "loose.txt").write_text(f"{PATH}\n")
    assert "loose.txt:1: local_path" in scan(repo, "--tree").stderr


# ---- the committed hook, run by a real `git commit` --------------------------


@pytest.mark.skipif(shutil.which("uv") is None or shutil.which("git") is None, reason="needs uv and git")
def test_the_real_pre_commit_hook_blocks_a_bad_commit_and_allows_a_clean_one(repo):
    (repo / "tools").mkdir()
    shutil.copy(TOOL, repo / "tools" / "scan_private.py")
    (repo / ".githooks").mkdir()
    for name in ("pre-commit", "pre-push"):
        shutil.copy(HOOKS / name, repo / ".githooks" / name)
        (repo / ".githooks" / name).chmod(0o755)
    git(repo, "config", "core.hooksPath", ".githooks")

    stage(repo, "bad.txt", f"{PATH}\n")
    blocked = git(repo, "commit", "-m", "should not happen")
    assert blocked.returncode != 0
    assert git(repo, "rev-list", "--count", "--all").stdout.strip() in ("", "0")
    assert SOMEONE not in blocked.stdout + blocked.stderr

    git(repo, "rm", "-q", "--cached", "bad.txt")
    (repo / "bad.txt").unlink()
    stage(repo, "good.txt", "all fine\n")
    ok = git(repo, "commit", "-m", "fine")
    assert ok.returncode == 0, ok.stderr
    assert git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"


@pytest.mark.skipif(shutil.which("uv") is None or shutil.which("git") is None, reason="needs uv and git")
def test_the_real_pre_push_hook_blocks_a_bad_push_and_allows_a_clean_one(repo, tmp_path):
    (repo / "tools").mkdir()
    shutil.copy(TOOL, repo / "tools" / "scan_private.py")
    (repo / ".githooks").mkdir()
    for name in ("pre-commit", "pre-push"):
        shutil.copy(HOOKS / name, repo / ".githooks" / name)
        (repo / ".githooks" / name).chmod(0o755)
    git(repo, "config", "core.hooksPath", ".githooks")
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))

    stage(repo, "ok.txt", "fine\n")
    assert git(repo, "commit", "-q", "-m", "clean").returncode == 0
    assert git(repo, "push", "-q", "origin", "main").returncode == 0           # clean push goes through
    published = git(remote, "rev-list", "--count", "main").stdout.strip()

    stage(repo, "bad.txt", f"{PATH}\n")
    # --no-verify skips the pre-commit hook, so only the push check stands in the way.
    assert git(repo, "commit", "-q", "--no-verify", "-m", "slipped past").returncode == 0
    blocked = git(repo, "push", "-q", "origin", "main")
    assert blocked.returncode != 0
    assert "bad.txt:1: local_path" in blocked.stderr
    assert SOMEONE not in blocked.stdout + blocked.stderr
    assert git(remote, "rev-list", "--count", "main").stdout.strip() == published  # nothing arrived
