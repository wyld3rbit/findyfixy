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

## Where we stopped (read this first when resuming)

- Day 1 is built, tested and committed. Left open on purpose: the Day 1 drill (skipped) and the Day 1 evidence line (the new end-of-day step in `CLAUDE.md`).
- Next: Day 2, structured output and tools (`d02.md`). From Day 2 on, `CLAUDE.md` says to start with the warm-up: the learner explains Day 1 from memory in 3 to 5 lines before any file is opened. Reply in one line: right, or the one thing missed.
- Things to watch on Day 2:
  - Self-reported confidence came back on mixed scales (0.95, 9.0, 100.0 from the 1.5B model; 0.95 from the 7B). Day 2's `Triage` schema and validate-retry should deal with this.
  - Both models returned only an intro sentence in `answer` for question 1, with no list of checks. Cause not diagnosed. Check whether the `Triage` schema fixes it.
  - `uv add mcp` and `langchain-mcp-adapters` are not in the bill of materials. Check licenses and tell the learner before adding.
- Run commands from `code/`. Tests: `uv run pytest` (9 passed, 4 integration deselected). Integration: `uv run pytest -m integration` (needs Ollama up; passed live once, 4 of 4). Router: `uv run python -m app.graph`.
- Preferences saved in Claude's memory and in `CLAUDE.md`: plan before code, short plain sentences, staff+ depth docs, idiomatic code, no personal data or absolute paths, no judging of drills unless asked, never suggest tracker log updates, ask before committing.

## Day 1: setup and first graph (complete except drill and evidence line)

### Done

- Checked tools: uv 0.13.1, Python 3.15, Docker Compose 5.4, Ollama 0.40.6.
- Pulled `qwen2.5:1.5b` and `qwen2.5:7b`. Small model replies to a test prompt.
- `git init`, `uv init`, `uv venv`. Project name `triage-copilot`.
- Added packages: `langgraph` 1.2.16 (MIT), `langchain-ollama` 1.1.0 (MIT), `pydantic` 2.14.1 (MIT), `python-dotenv` 1.2.4 (BSD-3-Clause). Locked in `uv.lock`.
- `.gitignore`: the lesson's 3 lines at first. Now one root file, see "Git layout".

- `.env` (git-ignored) holds `SMALL_MODEL` and `LARGE_MODEL`. No `.env.example`, per the lesson.

- `app/graph.py`: router graph per the lesson spec. Ran on the 5 lesson questions: 5 of 5 stayed on the small model, mean latency 2.9s. The large path was verified later with a forced escalation (see below).
- Removed the 0-to-1 range check on `confidence` (small model wrote 75 and crashed the run). The raw number is used as written. `main()` prints it.
- `notes/day1.md` (in the training folder, outside `code/`): results, caveats, self-check.
- Large-model path verified with one forced escalation on a real question: small 2.6s, then large 7.2s, total 9.8s, final model `qwen2.5:7b`. Recorded in `notes/day1.md`.
- `docs/graph.md`: hand-drawn Mermaid diagram of the router graph. Routing wiring verified with stubbed models (low confidence runs `large`, high confidence stops at `small`).

- Integration test bank: `code/tests/integration/` (`conftest.py`, `test_ollama.py`), 4 tests that call real Ollama models. Marked `integration` and deselected by default, so `uv run pytest` stays fast and offline (9 passed, 4 deselected). Run with `uv run pytest -m integration`. They skip, with a specific reason, if env names are unset, the server is unreachable, or a model is not pulled. They assert structure only, never answer text or confidence values. A `deadline` fixture (SIGALRM, Unix only) bounds each call.
  - Verified with Ollama OFF: default run, skip on unreachable, skip on unset model names. Verified with a stub server: skip on a missing model, and the fixture lets tests run when both models are listed. Verified the deadline fixture on its own.
  - Live run, Ollama 0.40.6 with both models pulled: 4 passed in 33.04s. Per test: small model 1.70s, large model 4.62s, forced escalation 10.83s, full five-question run 15.62s. The server for this run was started by Claude with `ollama serve` as a background task, then stopped so the learner could run their own.
  - Timings are one run. Load time for the 7B model is included in whichever test called it first, so do not read them as steady-state latency.

- Test output: added `pytest-sugar` 1.1.1 (BSD, OSI-approved) and its dependency `termcolor` 3.3.0 (MIT), dev group only. In a terminal, pytest shows a green check for a pass, a red cross for a failure, `s` for a skip, and a progress bar. In a pipe (CI, captured output) it stays plain. `--force-sugar` shows the icons anyway; `-p no:sugar` turns the plugin off. The suite is unchanged: 9 passed, 4 skipped without Ollama.

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

### Day 2, step 1: Triage schema and validate-retry

- `app/triage.py` (new): `Triage` model (`severity` Literal, `component`, `next_action`, `confidence` bounded 0 to 1), `parse_triage`, `ask_validated`, `TriageError`. One retry, with the validation error and the bad reply (truncated to 500 chars) in the prompt. Worst case is 2 calls per model.
- `app/graph.py`: `Answer` is gone. State is now `question`, `triage` (a `Triage` or None), `model_used`. `ask` now returns raw reply text, not a parsed object. `ollama_ask` passes `format=Triage.model_json_schema()` (constrained decoding) and no longer uses `with_structured_output`. A small-model failure after its retry sets `triage=None` and `route` escalates. A large-model failure raises `TriageError`.
- Differs from the lesson: the lesson swaps the schema and adds a retry. Here the retry is a pure function with an injected `ask`, and parsing moved out of the client so the error text is Pydantic's.
- Behavior change on purpose: the Day 1 "confidence = 75" case used to count as confident. It now fails validation, retries, and escalates. `tests/test_routing.py` documents it.
- Tests: `tests/test_triage.py` (new, 13 cases), `tests/test_routing.py` updated (fake `ask` returns JSON text), integration tests moved to `Triage`. Default run: 25 passed, 4 deselected.
- Mutation check: made the retry re-send the original prompt without the error. `test_retry_prompt_carries_the_error_and_the_bad_reply` failed. Restored, 25 passed.
- Live run of `uv run python -m app.graph` (2026-10-19, Ollama 0.40.6): 5 of 5 questions gave a valid `Triage` on the 1.5B model. 0 of 5 escalated. Confidence 0.7 to 0.9. Latency 0.6s to 2.2s (one run; the first includes load). Retry use was not logged, so it is unknown. Several answers were valid but vague ("investigate the auth service").
- Live escalation tests (`tests/integration/test_ollama.py`, 2 new): the small model's reply is faked as `confidence: 75`, the large model is real. Asserts call order `[small, small, large]`, that the retry prompt names the `confidence` rule, and that the final record is valid. A control with a valid `0.9` reply must stay on small. Integration bank: 6 passed in 10.8s. Default run: 25 passed, 6 deselected.
- Mutation check: changed `triage is None or ...` to `triage is not None and ...` in `route`. The live test failed (small, small, no large). Restored, all green.
- Open: `docs/graph.md` still lists the Day 1 state fields.
- Open from earlier: no `exclude-newer` cooldown and no `make audit` yet. `pip-audit` on 2026-10-19 found no known vulnerabilities and no `.pth` files.

### Day 2, step 3: deploy tool, bound directly

- `data/deploys.json` (new): 11 synthetic deploy records for 5 services. Each has `hours_ago`, not a timestamp, so there is no clock in the code.
- `app/tools.py` (new): `DeploysArgs` (service is a `Literal`, hours 1 to 168), `get_recent_deploys`, `make_deploys_tool` (path injected), `make_tools_node` (`ToolNode` with `handle_tool_errors=True`, so bad arguments go back to the model as a message).
- `tests/test_tools.py` (new, 15 cases): filter and sort, inclusive window boundary, six bad-argument cases including a path-traversal string, the shipped data file, and the ToolNode on a model-style tool call.
- Library drift: this `ToolNode` raises "Missing required config key" when invoked alone. Tests run it inside a one-node graph.
- Mutation check: removed the 168-hour cap. 2 tests failed (`too-long`, schema bound). Restored, 40 passed.
- Not done: the tool is not wired into the router graph yet, and no live model has called it. That comes with the MCP version so both paths use the same function.

### Day 2, live console, steps A, B and C (added by request, not in the lesson)

- Goal: one animated page that runs the system and shows the flow and an event log. This step is the backend; the page is step C.
- `app/events.py` (new): `Event`, `Emit`, `make_emit` with an injected clock. `ask_validated` and `build_graph` take an optional `emit` and report node, request, reply, validation, retry and route events. Default reports nothing, so old callers are unchanged. `route` stays pure; a `decide` wrapper emits the route event.
- `app/console.py` (new): stdlib `http.server`, no new package. `GET /`, `/config`, `/events` (SSE, replays the latest run), `POST /run`. Live and fake modes. Fake scenarios: `stay_small`, `retry_then_small`, `escalate`, `large_fails`. One run at a time. Binds 127.0.0.1, checks Host and Origin, needs JSON content type, caps body and question size.
- Tests: `tests/test_events.py` (5), `tests/test_console.py` (17, real socket on a free port). Default run: 62 passed, 6 deselected.
- Mutation checks: dropped the retry event (1 failed); disabled the Host check (1 failed). Both restored.
- Bugs found by the tests and fixed: a replay past the end of a finished run waited out the keepalive; early 4xx rejects left the unread body on a keep-alive socket.
- Step C: `docs/architecture.html`. Inline SVG schematic, Run and Replay controls, live event log (click a row for its JSON), result panel. Built nodes are solid; deploy tool is amber (built, not wired); Days 3 to 9 are dashed. Event data is set with `textContent` only; a test fails if the page ever uses `innerHTML` or `eval(`. Default run: 63 passed. JS syntax checked with `node --check`. Not yet checked in a browser by me.
- Live "Force escalation" (by request, after a live run stayed on small at confidence 0.75): `force_escalation` on `POST /run` sets the threshold to 1.01 for that run. Both models still run for real; only the route is overridden. `run_started` carries `forced`; the page and log say so. Live mode only. Test with a fake model; mutation (threshold left at 0.7) failed it. Default run: 64 passed.
- Hunt for a bad value (by request, after live runs never showed a retry): `hunt_max` on `POST /run` (live only, 1 to 50, page default 20) repeats real runs until a `validation` event has `ok` false, then stops. The run that saw it finishes, so the retry is part of the result. Ends with one `hunt_finished`: `found`, `exhausted`, `stopped` (via `POST /stop`) or `error` (Ollama down). Questions rotate through the fixed five, starting at the one chosen; a typed question repeats. Cannot combine with force escalation. New events: `hunt_started`, `hunt_run`, `hunt_finished`. A hunt's stream ends on `hunt_finished`, not on its inner runs. The page animates fast during the hunt, then replays the finding run at normal speed.
- Tests: 10 new in `tests/test_console.py` (stops at run 3, cap, rotation, error, stop, 422s). Mutations: ignore bad values (1 failed), lift the 50 cap (1 failed). Both restored. Default run: 74 passed.
- Page checked by replaying a recorded hunt through its own JS with a stub DOM, not in a browser.
- Open: no live hunt has been run yet, so no measured bad-value rate. Record it in `notes/day2.md` when it exists.
- Realistic incident bank (by request): `data/incidents.json` (8 synthetic log excerpts, 1.3k to 2k chars, each with a hand-written answer key the model never sees), `app/incidents.py` (typed loader, `MAX_TEXT_CHARS = 3500`). Lesson five are unchanged and still what `app.graph` runs. Console: `/config` lists both banks, hunt rotates whichever bank the chosen question is in, question cap 500 to 8000 chars, body cap 8 to 32 KB. Page: question-set selector, text preview, textarea. Page renamed to "Hunter".
- Why 3500 chars: `ollama ps` showed a 4096-token context for qwen2.5:1.5b (the model supports 32768, Ollama runs it at 4096). Overflow is cut silently from the start. The retry prompt is about double the question.
- Tests: `tests/test_incidents.py` (35: size, realism, documentation IPs only, answer key not in the text, severity mix), 2 new console tests. Mutation: a non-documentation IP in one incident failed its test (the first attempt, with BSD `sed`, changed nothing and passed; redone in Python). Default run: 111 passed.
- One live pass over the 8 incidents (2026-10-20, learner did not run it; not in notes/day2.md): 0 of 8 escalated, 0 invalid replies, confidence 0.75 to 0.95, severity matched the key 3 of 8, component 4 of 8 by a crude substring match. Wrong answers carried confidence 0.8 to 0.95. One run, sampling is random, so treat as a first look.
- Page review in a real browser (headless Chrome, screenshots at desktop width and a 390px iframe), by request. Fixed: (1) `hidden` controls stayed visible because `.field { display: grid }` beat the browser's `[hidden]` rule, so Scenario showed in Live mode and Live controls in Fake mode; (2) the animation froze in a background tab because `requestAnimationFrame` does not fire there, now the dot is skipped when hidden and the wait has a backstop; (3) the retry label sat on its arc; (4) blank band above the diagram (viewBox cropped); (5) the Result panel printed a whole incident as the question, now clipped to 160 chars; (6) route badge overflowed the diamond on "no record"; (7) log rows were mouse-only, now keyboard-openable with aria-expanded; (8) the preview ignored typed text; (9) planned boxes were low contrast; (10) Stop had no error handler; (11) Mode box truncated its label and the number input was unstyled. The diagram scrolls inside its panel on a phone. Not a bug: the apparent phone overflow in a plain headless screenshot, which came from Chrome's minimum window width. Regression test for the `[hidden]` rule; removing the rule fails it. Default run: 112 passed.
- Force retry (by request): `force_retry` on `POST /run` (live only). `inject_bad_first_reply` replaces the small model's first reply with legal JSON carrying `confidence: 75`, emits `fault_injected`, and leaves every later call alone. Validation rejects it, the real error text goes into the retry prompt, and the real model answers that. Can combine with force escalation; the server refuses it with a hunt or in fake mode (422). The page marks the swap: amber `fault_injected` log row, "reply INJECTED" badge.
- Tests: 5 new (real retry on the real model: called once, prompt has the question and the error; no escalation alone; combines with force escalation; two 422s). Mutation (injection returns the real reply) failed the retry test; restored. Default run: 117 passed.
- Controls panel compacted (by request): one row for mode, set, question and buttons; one row of options; preview and textarea side by side; the empty status lines no longer take space. Live controls now start about 330px from the top instead of about 650px. Checked in headless Chrome, including a live Force-retry run with a stand-in model.
- Renamed (by request): project `triage-copilot` is now `helpy` in `pyproject.toml` and `uv.lock`; the page was Hunter and is now Helpy (title, heading, the test that checks them). `uv lock` moved only the project's own entry. No third-party package or version changed, so no `make audit` run was needed for this. The Day 0 line above, "Project name `triage-copilot`", is history and stays. Not changed: README.md, 00-plan.md and lessons.js still say "Triage Copilot" (course files, left for the learner to decide).
- Renamed again (by request): the product is now FindyFixy (package `findyfixy`). `pyproject.toml`, `uv.lock` (project entry only, no third-party change), the page title and heading, and the test that checks them. Course files: only the product name changed in `README.md` and `00-plan.md` (one line each), then `lessons.js` was rebuilt and `tools/check.py` printed OK. No lesson content was changed. This supersedes the Helpy entry above.
- Security (2026-10-20, approved by the learner): `pip-audit` found certifi 2026.7.22 affected by PYSEC-2026-4320, fixed in 2026.10.19. It was already in the committed lock, and yesterday's audit was clean, so the advisory is new. Upgraded with `uv lock --upgrade-package certifi==2026.10.19`: only certifi changed. certifi is MPL-2.0 (OSI-approved) and widely used. Re-audit: no known vulnerabilities, no `.pth` files. The fix was about 1 day old, so this is a deliberate exception to the 7-day release cooldown. Still open: `exclude-newer` is not set in `pyproject.toml`, and there is no `make audit` target yet (I ran its two checks by hand, with the requirements file in scratch).
- Open: the deploy tool is still not wired into the graph. Token counts per call are not in the events yet (comes with the Step 4 measurement).
- Decision: stdlib server, not FastAPI. Revisit at Day 8 if the gateway brings a framework.

### Day 2, step 4: direct tool path, measured

- `app/direct_run.py` (new): bounded tool-call loop (`MAX_TURNS = 4`) with an injected model. The tool is called in-process, not through `ToolNode` (differs from the lesson on purpose: the measurement isolates transport cost, and Step 5 runs the same loop through MCP). Records input and output tokens, wall time, turns, tool calls, tool errors, turn-cap hits and a correctness check (the fixed task has one right answer: auth 2.14.1 at 3h, not 2.14.0 at 30h). One warm-up run is printed and excluded. Tokens are None, not a partial sum, if any turn lacks usage.
- `tests/test_direct_run.py` (new, 14 cases, scripted fake model, no Ollama). Mutations: cap off by one (1 failed), tool result not fed back (5 failed). Both restored. Default run: 131 passed.
- Check run against the real 1.5B model (2026-10-20, mine, not for notes/day2.md): tool called 2 of 2, correct 0 of 2, 499 input and 46 to 48 output tokens, 0.6s. The model sent `hours: 1` instead of 24. That is a legal argument, so schema validation passed, and the tool returned nothing, so the model answered "no deploys". Validation catches shape, not meaning. Worth a line in the Day 2 drill.
- Open: the learner's 10-run measurement (`uv run python -m app.direct_run`), the MCP path (Step 5), the table in `notes/day2.md`. One tool understates MCP's token overhead, which grows with the number of tools.

### Day 2, step 4, stage A: the tool loop as a LangGraph graph with ToolNode (by request)

- `app/investigate.py` (new): `build_investigator(model, tool)` compiles `agent` and `tools` (a real `ToolNode`, error-handling on) with a turn counter in the state, capped at `MAX_TURNS = 4`. `analyse` and `format_evidence` are pure. Evidence text always names the arguments, so `[]` for `hours=1` reads as "nothing in the last hour". Aligns with the lesson (ToolNode) and replaces the hand-written loop I used earlier; the departure noted in the Step 4 entry above is closed.
- Behaviour difference: at the turn cap the final requested tool is NOT executed (the old loop executed it, then stopped). It is reported with an empty result, flagged as an error.
- `app/direct_run.py`: rewired onto the investigator; `execute_tool_call` is gone (ToolNode does it). New `args` column in the table (what the model asked for), which closes Step 4b.
- Tests: `tests/test_investigate.py` (new, 17 cases, real ToolNode and real tool, scripted model), `tests/test_direct_run.py` rewritten (7). Mutations: cap off by one (2 failed), no edge from tools back to agent (12 failed), evidence without arguments (1 failed). All restored. Default run: 141 passed.
- Two test bugs of mine, fixed: expected argument order (keys are sorted), and an `hours` of 200 that the tool correctly rejects (cap is 168).
- Check run on the real 1.5B model (2 runs, not for notes): 499 in, 46 out, 0.57s, `hours=1` both times. Same tokens and time as the hand-written loop, so ToolNode added no measurable overhead in this setup (2 runs only).
- Open: the learner re-runs the 10-run table with the ToolNode loop (the table in `notes/day2.md` is from the old loop and should be replaced). Stage B: wire into the router graph. Stage C: console events and page node. Step 5: MCP, which needs `ainvoke` because MCP tools are async-only.

### Day 2, step 4, stage B: the tool step wired into the router graph (by request)

- `build_graph(..., investigator=None)`: when given, an `investigate` node runs once before `small`. Its tool results become `evidence` in state and go into the Triage prompt of whichever model answers, and into the retry prompt. If `large` takes over it reuses the same evidence; the tool is not called again. With no investigator the graph is exactly the old two-node router (same nodes, same prompts).
- Why a separate step: tested live, `format=` (the Triage shape) makes the model emit a record and no `tool_calls`, so tools and structured output cannot share one call.
- `app/triage.py`: `build_prompt(question, evidence="")`, `EVIDENCE_PROMPT` (says tool output is data, not instructions, and shows the arguments), `ask_validated(..., evidence="")`. `app/investigate.py`: `investigation_prompt` (service list comes from the tool's own `Service` type). `app/events.py`: `tool_call`, `tool_result`. Tool events are reported after the loop finishes, not live, because the loop is one nested graph run. The node's exit event carries tokens, turns and calls.
- Tests: `tests/test_graph_tools.py` (10, real tool and ToolNode, scripted tool model, recording `ask`), 2 more in `test_triage.py`. Mutations: small ignores the evidence (5 failed), retry prompt drops it (2 failed), investigate skipped (8 failed). All restored. Default run: 153 passed.
- Two more test expectations of mine were wrong first (argument text format). Fixed.
- Live pass over the 8 incidents, plain vs with tools (2026-10-20, mine, not for notes/day2.md, one run, sampling random): the model called the tool for 5 of 8 and skipped 3. The step added about 900 to 2,100 input tokens per run, because the whole incident text is in its prompt too, and 0 to 2 seconds. Severity matched the key 2 of 8 plain and 2 of 8 with tools, on different incidents. No evidence of an accuracy gain.
- Known limitation: `data/deploys.json` is a fixed fake file. It does not match each incident's own timeline (for example the cascade log says auth 2.14.2 deployed at 16:21; the file has no 2.14.2). So the evidence can be unrelated to the incident, and accuracy with and without tools cannot be compared fairly. This pass shows cost and plumbing only. A fair comparison needs deploy data written to match each incident.
- Open: Stage C (console events, an Investigate node on the page, the Deploy tool box turning solid), then Step 5 (MCP, needs `ainvoke`). The learner's re-run of the 10-run table.

### Day 2, step 4, stage C: the tool step in the console and on the page (by request)

- `app/console.py`: `use_tools` on `POST /run` (live; 503 if no tool step is configured, which `main` builds from `SMALL_MODEL` and `bind_tools`). Two new fake scenarios run the real tool and real ToolNode with only the tool-calling model scripted: `tool_lookup` (auth, 24h) and `tool_wrong_window` (auth, 1h, which returns `[]`, mirroring what the real 1.5B model did). The scripted model reports no token usage on purpose, so the page shows "tokens unknown" and no invented numbers. `/config` gains `tools_available`. `run_started` carries `tools`.
- `docs/architecture.html`: the diagram is re-laid out with an Investigate node between Question and Small, and the Deploy tool box is wired to it (it was "built, not wired"). A tool call animates Investigate to the tool and back. A "Tool step" checkbox in live mode (default on when available). With tools off the page shows a dashed bypass arc, "tool step off", and dims the idle boxes. An empty result is amber, not green: legal but suspicious, since the window may be wrong. The Result panel shows the tool step's calls, turns and tokens. Legend: the "built, not wired" item is gone.
- Tests: 8 new console tests (scripted scenarios, 503, live with and without the box, config, page markers). Mutation: console drops the investigator (4 failed). Default run: 161 passed.
- Checked in headless Chrome: tool lookup, wrong window (amber "0 deploys"), and no-tools bypass. Not checked: a live run with the real model in the browser.
- Open: Step 5 (MCP; needs `ainvoke` and two packages, one not in the bill of materials), the learner's re-run of the 10-run table, and deploy data written per incident so accuracy with and without tools can be compared.

### Day 2, step 5a: MCP dependencies, release cooldown, `make audit` (approved by the learner)

- Added `mcp` 1.30.0 and `langchain-mcp-adapters` 0.3.2, both MIT. Checked before adding: exact names from their GitHub projects (`modelcontextprotocol/python-sdk`, `langchain-ai/langchain-mcp-adapters`); maintainers (Model Context Protocol, a Series of LF Projects, with four named maintainers; one named author for the adapter, in the LangChain organisation); downloads last month about 235 million and about 6.2 million; releases 2026-09-07 and 2026-08-06. `mcp` is pinned `<2` because the adapter requires it (mcp 2.3.0 exists). The adapter is not in the bill of materials in `00-plan.md`; the learner approved it.
- Lock: 50 to 66 packages, no existing version changed. All new packages are OSI-licensed (MIT, BSD-3-Clause, Apache-2.0, MIT-0, PSF).
- Installed on this Mac: 11 new packages, not 16. `override-dependencies` with a never-true marker drops `uvicorn`, `python-multipart`, `pyjwt` and `click`, which `mcp` lists as hard requirements but only imports for HTTP serving, OAuth and its CLI. That also leaves out `cryptography`, `cffi` and `pycparser` (3.8 MB for `cryptography`), about 4.4 MB in all. `pywin32` is Windows-only. The lock still records them; the install does not. Checked first by reading the `mcp` and adapter wheels' import chains (nothing we use loads them) and by importing `mcp`, `mcp.client.stdio`, `mcp.server.fastmcp` and the adapter in a scratch env. Undo: delete the override lines. An ImportError for one of those four means an MCP feature we do not use.
- Release cooldown set: `exclude-newer = "2026-10-17T00:00:00Z"`. A strict 7-day window (2026-10-13) fails today: the Day 1 lock holds seven packages first uploaded on 2026-10-15 and 16, including `langgraph` 1.2.16, which the project pins. The date moves forward deliberately, never to today. `exclude-newer-package = { certifi = "2026-10-21T00:00:00Z" }` keeps the certifi fix (PYSEC-2026-4320) that the cutoff would otherwise downgrade.
- `code/Makefile` (new) with `make audit`: `pip-audit` on the exported lock, plus a `.pth` allowlist check (`_virtualenv.pth`, `distutils-precedence.pth`, `__editable__*.pth`). Result: no known vulnerabilities, no unexpected `.pth`. Proved it can fail: a planted `.pth` made it exit 1. `.audit-req.txt` is git-ignored. This closes two earlier open items (no `exclude-newer`, no `make audit`).
- Default run: 161 passed.
- Open: 5b (an MCP server wrapping the same tool, the adapter client, async graph run), 5c (the 10-run measurement through MCP and the comparison table).

### Day 2, step 5b: the MCP path, persistent-session mode (approved; per-call mode still to do)

- `app/mcp_server.py` (new): FastMCP server over stdio exposing `get_recent_deploys`. It calls the same function as the direct path. Its argument descriptions and bounds are read from `DeploysArgs`, so the model sees the same tool either way; a test compares the two schemas. Log level WARNING so per-request logging does not add noise to measured runs.
- `app/mcp_client.py` (new): `persistent_tools()` starts the server as a child process (same Python, run from the project root, small default environment), holds one MCP session open, yields the adapter's LangChain tools, and stops the process on exit even on error.
- `app/mcp_run.py` (new): `uv run python -m app.mcp_run session [runs]`. Same task, loop, columns and summary as `direct_run`, run with `ainvoke`. Startup (spawn, initialize, list tools) is timed once and printed separately.
- `app/investigate.py`: `build_investigator` and `make_tools_node` accept one tool or a list. New pure `content_text`: MCP returns content blocks, and the evidence must be their text, not the Python repr of the list (which would have put `[{'type': 'text', ...}]` into the Triage prompt). `app/direct_run.py`: `stats_from` is shared by both paths.
- Two checks done before building tests: `ChatOllama` sends a block-list tool result to the model as the text plus a leading newline, so MCP's result shape adds almost nothing to the prompt; and an MCP tool error arrives with `status="error"`, so the tool-error column is right.
- Adapter facts read from its source: `get_tools()` opens a new session per tool call (over stdio, a new server process each time), and its tools are async-only, so the graph must run with `ainvoke`. This mode avoids the first cost on purpose.
- Tests: `tests/test_mcp.py` (12; real stdio server, scripted tool-calling model, no Ollama): schema parity, round trip equals the direct function, invalid arguments give an error, evidence is clean text, no server process left behind, including when the caller raises. Mutations: schema drift (1 failed), evidence uses the raw block list (5 failed), a stray server process (2 failed). Two test mistakes of mine fixed first: a malformed `pgrep` pattern that counted zero. Default run: 173 passed.
- Check run on the real 1.5B model, 2 runs (mine, not for notes): 499 input tokens, identical to the direct path; 0.58 to 0.61s against 0.57s direct; server startup 0.46s once; model sent `hours=1` again.
- Still to do: per-call session mode (the adapter's default), 5c (the learner's 10-run measurements for all modes and the table).

### Hot reload for the console (by request, approved)

- `uv run python -m app.console --reload` (and `--port N`). `app/dev_reload.py` (new): a stdlib supervisor runs the console as a child process and polls the files every 0.5s. A change to `app/**/*.py`, `data/*.json` or `.env` restarts the child, after a debounce so a burst of saves restarts once. A change to `docs/*.html` does not restart anything (the page is read from disk on each request); the page reloads itself. No new package: `watchfiles` or `uvicorn --reload` would add dependencies.
- The console exposes `/version` (boot token plus a fingerprint of the watched files) and `dev_reload` in `/config` only in dev mode. The page polls it once a second and reloads when it changes. If a run is in progress it waits until the run ends. If the server is mid-restart the poll just tries again.
- A broken edit does not crash-loop: the supervisor prints that the server exited and waits for the next change.
- SIGTERM and SIGHUP are turned into the same clean stop as Ctrl-C, so closing the terminal does not leave an orphaned server holding the port. (Found by an end-to-end run; without the handler the child was orphaned.)
- Tests: `tests/test_dev_reload.py` (19 with the console hooks and argument parsing; fakes for the process, clock and files). Mutations: no debounce (1 failed), respawn on crash (2 failed), Ctrl-C leaves the child (1 failed). End to end, with real processes in a scratch copy on port 8790: 14 of 14 checks (page edit does not restart; Python edit is live; broken edit gives one message and no loop; fixed edit recovers; SIGTERM and SIGINT stop both processes and free the port). With the signal handler removed, 11 of 14. The page's reload logic was checked in a stub: no reload when unchanged, reload when idle, deferred during a run and done after, no crash when the server is down. Default run: 190 passed.
- Mistakes of mine found along the way: an early version restarted the server on page edits too (the plan said only the page should reload), and several check scripts of mine had bugs (a background job that ignores Ctrl-C, a response check that only read the first 40 characters). Fixed or corrected.
- Limits: a restart kills a run in flight; a new dependency still needs `uv sync` first; `pyproject.toml` and `uv.lock` are not watched; if the supervisor itself is killed with SIGKILL its child is orphaned. The page's own reload was not watched in a real browser, only in a stub.

- Quiet dev log (by request): the console no longer logs `GET /version`, which the page polls once a second in dev mode and which buried `POST /run` and `GET /events`. Only that route is quiet. Test, with a mutation that fails it. End to end again: 14 of 14, zero `/version` lines in the supervisor log, other requests still logged. Default run: 191 passed.
- Found while answering "is escalation really using Ollama": the event log records the model name we ASKED for, not the one Ollama reports serving. Ollama's response carries a `model` field (checked: asking for `qwen2.5:7b` returns `model='qwen2.5:7b'`), and `/api/ps` lists what is loaded. The Ollama server log lines in a screenshot (`[GIN] ... POST /api/chat`, `slot ...`) show timings and token counts, not a model name, so they correlate with a call but do not prove which model ran. Not built: recording `served_by` from the response in the event log.

- Which model actually answered (by request): `ollama_ask` now returns a `Reply`, a `str` subclass that carries `served_by`, the model name Ollama reports in its response. Everything that treats a reply as text is unchanged. `model_reply` events carry `served_by` and `mismatch` (true only when a name was reported and differs from the one asked for; unknown is neither a match nor a mismatch, and fakes and injected replies report none). The tool step reports `served_by` too. The page shows it in the log, flags a mismatch in red on the node, and adds a "Served by" row to the Result panel. Checked live: asking for `qwen2.5:1.5b` and `qwen2.5:7b` returned the same names, no mismatch. Mutations: report the asked name (2 failed), never flag a mismatch (1 failed), treat unknown as a mismatch (2 failed). Default run: 202 passed.

### The picture now follows reality (by request: "are you sure the animations are correct?")

- Measured first, by replaying a recorded live run through the page's own animation code at real speed: the picture ran up to 5.6s behind a 4.6s run. It showed the small model starting 2.0s after the real call began (and after Ollama had finished with it), the large model starting 5.2s late, and the run finishing 4.6s late. So Ollama log lines appeared while the page still showed Question, Investigate or the tool. The user's observation was right.
- Two causes. (1) Fixed pacing: a 350ms pause and a 450ms move on every step, longer than real gaps, so the backlog grew. (2) The tool step reported its events in a burst after its loop finished, so its two real model calls were never shown as they happened.
- Fix A, `app/graph.py`: the investigate node runs the loop with `stream_mode="updates"` and emits as each step finishes. New per-turn `model_request` and `model_reply` events for the investigator's model (with `served_by`, `mismatch`, tokens), `tool_call` before the tool runs, `tool_result` after. No model call is announced that does not happen (a capped loop). `build_graph(..., investigate_model=...)` names the investigator's model for events, since it is injected already bound.
- Fix B, `docs/architecture.html`: live runs use real-time pacing (120ms moves, no pauses) and a catch-up mode: if more than one event is waiting, skip moves and pauses until caught up. Fake runs keep the readable pace, because their events arrive in bursts. The slow replay of a hunt's finding run keeps the readable pace too. The Investigate node now reacts to its own model calls. "Served by" in the Result panel uses the investigator's reply events.
- Measured again after the fix, on a fresh live run: every event shown within 0.14s of when it happened (large model start: real 3.39s, shown 3.39s; run finished: real 11.74s, shown 11.75s).
- Regression check, `tests/test_page_lag.py` with `tests/js/replay.js` (skips without Node): replays a run-shaped timeline and a burst of 8 parallel tool calls through the real animation code, and fails if any event is shown more than 0.5s late. Mutations: no catch-up (the burst test failed, after I added it: my first timeline did not exercise that path), live runs at the slow pace (2 failed). Tests for the live tool step: order of events, the tool call reported before the tool executes and the result after (mutation: collect the loop first, then report, failed), per-turn tokens, served model, no phantom model call at the cap. Default run: 211 passed (about 8s slower for the replay).
- Mistakes of mine along the way: an old scratch harness whose stub lacked `style`, and a first lag timeline too gentle to need catch-up.
- Not changed: the log still carries real timestamps, and is the truth for timing.

### Stress settings, Live as the default, and a favicon (by request)

- `app/stress.py` (new): `make_ollama_ask(stress)` builds the triage `ask` with one deliberate weakness, and every reply is still a real model reply: `no_format` (no JSON constraint), `vague_prompt` (the "0 to 1, not a percentage" sentence removed), `percent_log` (a line inside the report asks for a percentage, an untrusted-text injection), `small_context` (a 256 token window; Ollama cuts the start of the prompt without an error, and changing the window reloads the model). Temperature was measured and does nothing, so it is not offered. An unknown name raises. `triage.CONFIDENCE_RULE` and the evidence and retry markers are named constants so the stresses and the prompt cannot drift apart; a test pins that the removed sentence is the real one. The stress applies to the triage calls, not the tool step's model.
- Console: `stress` on `POST /run` (live only; 422 with fake mode or an unknown name), `run_started.stress`, `/config.stresses`. Can combine with Hunt, Force retry and Force escalation. Page: a Stress dropdown, a red "Stress" row in the Result panel, and the stress named in the log.
- Live is now the default mode. If the model names are not in `.env` the page falls back to Fake and says why.
- Favicon: an inline magnifier-with-a-check icon, so the browser makes no `/favicon.ico` request. `/favicon.ico` answers 204 and is not logged, for browsers that ask anyway.
- Measured on the real 1.5B model, 8 incidents each, through the real validate-and-retry path (2026-10-20, mine, not for notes/day2.md): invalid on the first try: none 0/8, no_format 8/8, vague_prompt 8/8, percent_log 5/8, small_context 7/8. The retry fixed 0 of those 28. Caveat that matters: each stress is applied to the retry call too, so the cause is still there. This measures a retry against a persistent cause, not a retry after a random one-off slip. The injected `Force retry` (a one-off bad first reply, then a real retry) is the other half and was not rerun here.
- Tests: `tests/test_stress.py` (14), 8 console tests (stress wiring, 422s, config, Live default, icon, favicon). Mutations: vague prompt removes nothing (2 failed), small context forgets the window (1), a typo runs unstressed (4), the console ignores the stress (2), no favicon route (1). Default run: 234 passed.
- Not committed yet: the live-pacing fix, `served_by` tool-step events, and this. Not yet run in a browser: a live run with a stress selected.

### Next

- Day 1 drill and evidence line, if wanted.
- Day 2: run it live, then tools (direct vs MCP) and the measurement table.
