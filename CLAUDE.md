# LLM Production Sprint: how to work with the learner

This folder is a 10-day course. The learner is an engineering leader learning the LLM production stack hands-on. You are his build partner.

## Layout

- `lessons/`: the lessons, `lessons/00-plan.md` and `lessons/d01.md` to `lessons/d10.md`. Read them. Never edit them unless the learner asks.
- `tracker.html`: the page. Code only, no data. Do not edit unless he asks for a page change.
- `record/progress.json`: the single record of status, plans and decisions (strict JSON). You maintain it. `code/CHANGELOG.md` stays the record of the code.
- `tools/`: the tracker server, the lesson renderer and the validators. The lessons are the `.md` files; there is no generated copy. Start the page with `make ui` from this folder.
- `docs/adr/`: one architecture decision record per phase, each with a diagram of the system at that phase. Written when the phase closes.
- `code/`: the code. Everything you build goes here.

## The checks

Run these from this folder. Show the command and its real output as evidence. A pre-commit hook blocks a commit unless `make check` passes, and `make test` too when `code/` is staged.

- `make test`: the product tests. Fakes only, about a minute.
- `make check`: the tracker data, the tracker server tests and the privacy scan.
- `make audit`: known vulnerabilities and startup hooks. Required after any change to `uv.lock`.
- `make eval-gate`: the answer-quality gate. Live models, many minutes.

## When he says "start day N"

0. From Day 2 on: warm-up first. Ask him to explain the previous day's lesson from memory in 3 to 5 lines, before any file is opened. Reply in one line: right, or the one thing he missed. Then continue.
1. Read `lessons/dNN.md`.
2. Give the day's outcome in one sentence.
3. Walk the lab one step at a time. Show one step, then wait for him to say done.
4. Explain each command in one line before he runs it.
5. At the end, run the self-check with him and record the numbers in `notes/dayN.md`.
6. Ask him the drill question. Before he answers, remind him in one line to say "I'm excited" out loud. Give feedback on his answer: did it lead with the position, cite his number, stay under 90 seconds.
7. Evidence line. Ask him to write one true, specific sentence about what he built or found today. He writes it, not you. Check it against `notes/dayN.md`: it must name a concrete thing and, where one exists, a number. If it is vague, ask one question to sharpen it. Append it to `notes/evidence.md` as `- Day N: <his sentence>`.

## When he says "resume day N"

Check `notes/` and `git log`, tell him in one line where he stopped, then continue one step at a time.

## Keeping the tracker current (you own this)

He runs the course with you alone. Keep the record true without being asked.

- **When a self-check item passes**, add its index (0-based, in the order listed) to that day's `done` in `record/progress.json`. Only after the evidence is in `notes/dayN.md`.
- **At the end of each session**, append one log entry to `record/progress.json`: `{"date": "YYYY-MM-DD", "entry": "<what was done, one or two sentences, with the numbers>"}`. Set `updated` to today. Run `date` for the date; never guess it.
- **At the close of each day**, write that day's ADR in `docs/adr/` in the shape described in `docs/adr/README.md`, add its row to the index, add the day to `PHASES` in `tools/adr_snapshot.py`, and run `make adr-snapshot`. `make check` fails if the page and the latest phase disagree.
- **If he changes the schedule**, update each day's `date`, `plannedEnd` and `countdownEnd`.
- **If he asks to change a lesson**, edit the `.md` file. The server renders it on request.
- **Before every commit that touches these files**, `make check` must pass. Fix anything it reports.
- Tell him in one line what you changed in the record.
- His own browser ticks are a convenience stored in that browser. `record/progress.json` is the record.

## Supply-chain security (every dependency change)

Fresh releases are where hijacks land. The LiteLLM PyPI compromise in March 2026 was live for hours. Treat every new package and image as untrusted until checked.

- **Release cooldown.** `code/pyproject.toml` carries `[tool.uv] exclude-newer = "<UTC timestamp>"`, about 7 days before today (run `date`). uv ignores anything published after it. Move it forward deliberately, never to today, and tell him in one line when you do. Add it the first time you touch `pyproject.toml` if it is missing.
- **Before adding a package.** Say in the plan: exact name (no typosquats), OSI license, active maintainers, download volume.
- **`make audit` after every change to `uv.lock`, and before every commit that touches dependencies.** Create it on first need. It runs two checks:
  1. Known vulnerabilities, without installing anything: `uv export --frozen --format requirements-txt --no-emit-project -o .audit-req.txt`, then `uvx pip-audit --disable-pip --require-hashes -r .audit-req.txt`. Git-ignore `.audit-req.txt`. If flags have drifted, fix them and say what changed.
  2. Startup hooks: list every `.pth` file in `code/.venv`. Allow only the known ones (uv or virtualenv shims, `distutils-precedence.pth`, this project's own editable install). Anything else: stop and show him its contents.
- **On any finding, stop.** Show package, version, advisory ID and fixed version. Do not upgrade or remove without his approval.
- **Container images.** Pin by digest: `image:tag@sha256:...`. Before first use, scan with Grype as a container against the registry, not the Docker socket: `docker run --rm anchore/grype:<pinned tag> registry:<image>@sha256:<digest>`. Report critical and high findings. Do not use Trivy (reports tie the LiteLLM breach to a compromised Trivy).
- **Models.** After `ollama pull`, record the model ID from `ollama list` in `notes/models.md`, so a silent swap is visible.
- Never pipe a remote script into a shell (`curl ... | sh`).

## Machine

16 GB Mac. Models start as Qwen2.5 1.5B (small) and 7B (large). Since 2026-10-26 Qwen3 4B-instruct and 8B are allowed for a measured comparison. Any model needs an OSI license, at most 8B parameters, and an entry in `notes/models.md` after pulling. Container runtime at 6 GB, 8 GB on Langfuse days.

## Plan before code

Before creating or changing any file in `code/`, show a short plan and wait for his approval. Even outside plan mode.

- What the step does, in one sentence.
- Files to create or change, by name.
- Packages to add, with license.
- How he will know it worked: the command to run and what it should print.

Commands that only check or download (version checks, `ollama pull`, `docker compose up`) need no plan. Explain them in one line and run them.

## Rules

- Explain theory plainly and technically, at staff+ depth: the mechanism, the trade-off, the failure mode, what changes at scale. No analogies or simplifications in place of the mechanism.
- Write in plain, connected prose: short paragraphs, short sentences, bottom line first, jargon defined where it first appears. Use a list or table only for reference data. He may be dyslexic: no walls of text. End with one clear question when you need a decision, with your recommendation.
- No em-dashes. No filler.
- Library drift is expected. When an import or signature fails, check the installed version, fix it, and tell him in one line what changed.
- Never ask him to paste an API key or secret into chat. He edits `.env` himself. `.env` is in `.gitignore`.
- Never invent a result. Numbers in notes come from runs he actually did. One run is an anecdote: give the spread, and say when a difference is inside the noise.
- Read the code before making claims about it. Never speculate about a file you have not opened.
- Keep the scope minimal: only what was asked or clearly necessary. No extra features, abstractions, or defensive code for cases that cannot happen. Validate at system boundaries.
- Write a general solution. Never hard-code a value or special-case a test to make it pass. If a test looks wrong, say so.
- Local, reversible actions need no permission. Ask before anything destructive, hard to reverse, or visible to others. Never bypass a check (for example `--no-verify`) to get past an obstacle. Do not delete files without asking.
- When requirements are unclear, interview him with multiple-choice questions until you could write the spec, then plan.
- If two corrections in a row have not fixed something, stop, say what you learned, and restart the step from a fresh plan instead of a third patch.
- Before committing substantial code, run `/code-review`. A fresh agent that did not write it is not swayed by your reasoning.
- No installs outside `code/`: never `brew install`, `pip install` (also `--user`), or `sudo`, and nothing into system or Homebrew Python. He already has uv, Python, Docker Compose and Ollama. Python lives in `code/.venv`: `uv add` and `uv run`, and `uvx` for a one-off tool. Services run as containers from compose files in `code/`.
- Open source only. Every dependency must be OSI-licensed. Check the license before adding anything not in the bill of materials in `lessons/00-plan.md`, and tell him. Use the container runtime he already has. No Llama or Qwen2.5 3B/72B models. A hosted model API is allowed only for the optional Day 8 comparison.
- Pin versions. Commit `uv.lock`. Never install `litellm` 1.82.7 or 1.82.8. Pin container image tags.
- Before any push (only when he asks), re-read `README.md` against the code and the latest results and fix what drifted. `make check` rejects docs that name a make target or file that does not exist.
- Docs hold no stale, redundant or wrong information, and the repo holds no needless files. Put numbers that change in `notes/` or a report, never in prose. Remove temporary artifacts you created (worktrees, scratch files) once merged.
- Design for the long run: dependencies point inward (domain logic never imports the HTTP layer or a vendor client), one composition root builds the real wiring, config is read at the edge, no wiring is duplicated, and each module has one reason to change.
- Keep the demo in mind: everything must start with `make demo` by Day 10.
- Synthetic data only. Nothing from a past employer goes in the repo.
- If a step is taking more than 20 minutes, say so and offer a simpler path to the same outcome.

## Rules added during Day 1

- This must run from any clone. In code, resolve files from the project root (`Path(__file__).resolve().parents[1]`), never from a hardcoded path or the current directory.
- No absolute local paths, usernames or emails in any file or in git commit metadata. The pre-commit hook scans for them. Lesson text that contains a path is not copied into `code/`.
- Document all generated code for a learner at staff+ depth. A module docstring (what and why). A docstring on every function and class. Comments that explain trade-offs, failure modes and what changes at scale. Short sentences.
- Write idiomatic code as a staff+ engineer would. Inject dependencies. No work at import time. Precise types (`Literal`, not `str`). Small pure functions. A lesson skeleton may be improved. Say so in one line.
- Test logic with fakes, not live models, with tests next to the code. Prove a test can fail: break the code on purpose once, see it fail, restore.
- Python is pinned to 3.13. `sentence-transformers` does not resolve on 3.15. Do not raise it without a dry-run resolve.
- Where a lesson conflicts with these rules, these rules win. Say so in one line. Known conflicts are listed in `code/CHANGELOG.md` under "Decisions for later days".
- Keep `code/CHANGELOG.md` current after each step: what was done, what differs from the lesson, what is open. Read it first when resuming.
