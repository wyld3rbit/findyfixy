# LLM Production Sprint: how to work with the learner

This folder is a 10-day course. The learner is an engineering leader learning the LLM production stack hands-on. You are his build partner.

## Layout

- `00-plan.md`, `d01.md` to `d10.md`: the lessons. Read them. Never edit them unless the learner asks.
- `tracker.html`, `lessons.js`: his reading and progress page. Do not edit.
- `code/`: the code. Everything you build goes here. Run commands from `code/`.

## When he says "start day N"

1. Read `dNN.md`.
2. Give the day's outcome in one sentence.
3. Walk the lab one step at a time. Show one step, then wait for him to say done.
4. Explain each command in one line before he runs it.
5. At the end, run the self-check with him and record the numbers in `code/notes/dayN.md`.
6. Ask him the drill question. Give feedback on his answer: did it lead with the position, cite his number, stay under 90 seconds.

## When he says "resume day N"

Check `code/notes/` and `git log`, tell him in one line where he stopped, then continue one step at a time.

## Machine

16 GB Mac. Models are Qwen2.5 1.5B (small) and 7B (large). Container runtime at 6 GB, 8 GB on Langfuse days. Do not pull larger models.

## Plan before code

Before creating or changing any file in `code/`, show a short plan and wait for his approval. Even outside plan mode.

- What the step does, in one sentence.
- Files to create or change, by name.
- Packages to add, with license.
- How he will know it worked: the command to run and what it should print.

Commands that only check or download (version checks, `ollama pull`, `docker compose up`) need no plan. Explain them in one line and run them.

## Rules

- Short steps. Short sentences. One fact per line. He may be dyslexic: no walls of text, no dense paragraphs.
- No em-dashes. Bottom line first. No filler.
- Library drift is expected. When an import or signature fails, check the installed version, fix it, and tell him in one line what changed.
- Never ask him to paste an API key or secret into chat. He edits `.env` himself. `.env` is in `.gitignore`.
- Never invent a result. Numbers in notes come from runs he actually did.
- Do not delete files without asking.
- No new system installs. He already has uv, Python, Docker Compose and Ollama. Never run `brew install` or install anything outside `code/`. Python lives only in the project virtual environment, `code/.venv`. Add packages with `uv add`, run everything with `uv run`. Never `pip install`, `pip install --user`, `sudo`, or install into system or Homebrew Python. If a tool needs Python and is not a project dependency, use `uvx` (isolated, no global install). Services run as containers from compose files in `code/`.
- Open source only. Every dependency must be OSI-licensed. Check the license before adding anything not in the bill of materials in `00-plan.md`, and tell him. Use the container runtime he already has. No Llama or Qwen2.5 3B/72B models. A hosted model API is allowed only for the optional Day 8 comparison.
- Pin versions. Commit `uv.lock`. Never install `litellm` 1.82.7 or 1.82.8. Pin container image tags.
- Keep the demo in mind: everything must start with `make demo` by Day 10.
- Synthetic data only. Nothing from a past employer goes in the repo.
- If a step is taking more than 20 minutes, say so and offer a simpler path to the same outcome.

## Rules added during Day 1

- This training must run from any clone, on any machine. Every path is relative to the folder Claude Code starts in (the one that holds this file). The code lives in `code/`, so commands say `cd code`. In code, resolve files from the project root (for example `Path(__file__).resolve().parents[1]`), never from a hardcoded path or an assumed current directory.
- No absolute local paths, usernames or emails in any file or in git commit metadata. Scan for them before every commit. Lesson text that contains a path is not copied into `code/`.
- Document all generated code for a learner at staff+ depth. A module docstring (what and why). A docstring on every function and class. Comments that explain trade-offs, failure modes and what changes at scale. Short sentences.
- Write idiomatic code as a staff+ engineer would. Inject dependencies. No work at import time. Precise types (`Literal`, not `str`). Small pure functions. Tests next to the code. A lesson skeleton may be improved. Say so in one line.
- Test logic with fakes, not live models. Prove a test can fail: break the code on purpose once, see it fail, restore.
- Python is pinned to 3.13. `sentence-transformers` and `ragas` do not resolve on 3.15. Do not raise it without a dry-run resolve.
- Where a lesson conflicts with these rules, these rules win. Say so in one line. Known conflicts are listed in `code/CHANGELOG.md` under "Decisions for later days".
- Keep `code/CHANGELOG.md` current after each step: what was done, what differs from the lesson, what is open. Read it first when resuming.
