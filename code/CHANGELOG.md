# Changelog

Newest entry first. Written so another agent can resume the work.

## Conventions (read first)

- Run everything from `code/`. Use `uv run`. Add packages with `uv add`. Never `pip install`.
- No absolute local paths, usernames or emails in any file or commit. This repo is public.
- Synthetic data only. Open-source licenses only. Pin versions and image tags.
- Models: Qwen2.5 1.5B (small) and 7B (large) on Ollama. Names come from `.env`.
- Plan before code: show the plan for each step and wait for approval.
- Document all generated code for a learner at staff+ depth: a module docstring (what and why), docstrings on every function and class, and comments that explain trade-offs, failure modes and what changes at scale. Short sentences. No local paths or personal data in docs either.

## Tracker log lines (for the learner to paste into tracker.html)

Agents do not edit `tracker.html`. These lines are ready to add to its `log` array.

- "Day 1 complete except the drill. Python pinned to 3.13. Router refactored with an injected model function, 9 routing tests that need no Ollama."
- "Real-model rerun after the refactor: 0 of 5 escalated again. Confidence came back on mixed scales (0.95, 9.0, 100.0), so self-reported confidence is a weak signal."
- "Personal data scrubbed from the course files. The Start in Claude Code button now works out its folder from the page address."

## Scrub of personal data (course folder)

- Removed the hardcoded local path from `tracker.html`, `lessons.js` and `d01.md` (now `cd code`), and the first name from `CLAUDE.md`. The tracker's start button no longer builds a path at all: it says to open a terminal in the training folder and run `claude`. Every path in the training is relative to that folder.
- `CLAUDE.md` gained a "Rules added during Day 1" section: no paths or PII, documentation depth, idiomatic code, fakes in tests, Python 3.13 pin, rules-win-over-lessons, keep this changelog current.
- Folder layout: the code folder was renamed from `repo/` to `code/` so the whole training folder can be one git repo. All path references in the lessons, tracker and `CLAUDE.md` were updated. `code/.venv` was rebuilt from `uv.lock`, because a virtual environment keeps the old folder path in its scripts.

## Git layout

- One git repo, rooted at the training folder (the one that holds `CLAUDE.md`). The code is in `code/`. There is no inner repo.
- The four earlier commits were merged in under `code/`, so `git log` keeps the history.
- One `.gitignore`, at the training root. Its patterns have no leading slash, so they match at any depth and cover `code/` too. The separate `code/.gitignore` was removed. Keep it to one file.
- Daily notes live in `notes/` at the training root, not in `code/`. Moved with `git mv`, so history follows.
- Root `README.md` tells a new reader how to start. No license file yet: the owner chooses one.
- Not done: pushing to GitHub, and choosing a license.

## Day 1: setup and first graph (in progress)

### Done

- Checked tools: uv 0.13.1, Python 3.15, Docker Compose 5.4, Ollama 0.40.6.
- Pulled `qwen2.5:1.5b` and `qwen2.5:7b`. Small model replies to a test prompt.
- `git init`, `uv init`, `uv venv`. Project name `triage-copilot`.
- Added packages: `langgraph` 1.2.16 (MIT), `langchain-ollama` 1.1.0 (MIT), `pydantic` 2.14.1 (MIT), `python-dotenv` 1.2.4 (BSD-3-Clause). Locked in `uv.lock`.
- `.gitignore` covers `.env`, `.venv/`, `__pycache__/`.

- `.env` (git-ignored) holds `SMALL_MODEL` and `LARGE_MODEL`. No `.env.example`, per the lesson.

- `app/graph.py`: router graph per the lesson spec. Ran on the 5 lesson questions: 5 of 5 stayed on the small model, mean latency 2.9s. The large path is untested.
- Removed the 0-to-1 range check on `confidence` (small model wrote 75 and crashed the run). The raw number is used as written. `main()` prints it.
- `notes/day1.md` (in the training folder, outside `code/`): results, caveats, self-check.
- Large-model path verified with one forced escalation on a real question: small 2.6s, then large 7.2s, total 9.8s, final model `qwen2.5:7b`. Recorded in `notes/day1.md`.
- `docs/graph.md`: hand-drawn Mermaid diagram of the router graph. Routing wiring verified with stubbed models (low confidence runs `large`, high confidence stops at `small`).

### Changed from the lesson

- Python pinned to 3.13 (`.python-version`, `requires-python = ">=3.13,<3.14"`). `sentence-transformers` and `ragas` do not resolve on 3.15 (checked with a dry-run resolve). uv-managed 3.13 was already present.
- `.gitignore` expanded beyond the lesson's 3 lines. `data/` is NOT ignored: the lesson's data files must be committed. `reports/` is ignored.

- `uv init` (0.13.1) made a packaged layout. Removed `src/`, the build backend, the script entry and the author block from `pyproject.toml`. Plain app layout now.
- `uv init` copied a name and email from git config into `pyproject.toml`. Removed.
- Skipped the lesson's `cd` line, which holds a full local path.

### Refactor: injected model function (staff+ idiom)

- `app/graph.py`: graph is now built by `build_graph(ask, *, small_model, large_model, threshold)`. No work at import time. The real call is `ollama_ask`. `route` returns `Literal["large", "__end__"]`, so LangGraph draws all edges. The module-level `app` variable is gone; `uv run python -m app.graph` still works.
- Departs from the lesson skeleton (module-level graph, lambda router) on purpose. Behavior is unchanged.
- Added `pytest` 9.1.1 (MIT), dev group only. `tests/test_routing.py`: 9 cases, no Ollama needed. Mutation-checked: changing `>=` to `>`, or dropping the `Literal` annotation, makes a test fail.
- `pyproject.toml`: `pythonpath = ["."]` so tests can import `app`.
- Not done yet (deferred): config object and env-driven threshold (Day 8), timeout and retry (Day 2), lint, format and type-check tooling (needs license check for ruff and pyright).

### Decisions for later days (from a read of d02 to d10)

- Day 3: pin the exact `pgvector/pgvector` tag (lesson uses floating `pg16`). Bind port 5432 to `127.0.0.1`.
- Day 5: golden set must be synthetic only, despite the lesson saying "from real experience".
- Day 9: generate the launchd or cron file at setup time and git-ignore it. Commit a template only. No absolute paths in the repo.
- Day 10: fresh clone has no `.env`. Add a committed `.env.example` and have `make setup` copy it.
- Day 6: check licenses of Redis and MinIO in the Langfuse stack before use. Not in the bill of materials.

### Next

- First git commit (waiting on the user's choice of author identity).
- Day 1 drill question.
- Day 2: structured output (`Triage` schema) and tools.
