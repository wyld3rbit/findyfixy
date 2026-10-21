# Day 2 notes: structured output and tools

## Direct vs MCP: the same tool, three ways

Date: 2026-10-20. Model: `qwen2.5:1.5b`. One warm-up run (excluded), then 10 measured runs per path.
Task (fixed): "Which deploys to the auth service happened in the last 24 hours? Use the get_recent_deploys tool, then answer in one sentence."
Right answer from `data/deploys.json`: auth 2.14.1 (3h ago) only. Auth 2.14.0 was 30h ago, outside the window.
All three paths use the same model, the same prompt and the same LangGraph tool loop (`agent` and `ToolNode`, turn cap 4). Only where the tool lives changes.

Commands (from `code/`):

- Direct: `uv run python -m app.direct_run`. The tool is a plain function in the same process.
- MCP, one persistent session: `uv run python -m app.mcp_run session`. One server process, started once, over stdio.
- MCP, per-call sessions: `uv run python -m app.mcp_run per-call`. The adapter's default: every tool call starts a new server process.

| Path | Input tokens | Output tokens (mean) | Median secs | Mean secs | Tool calls | Correct |
|---|---|---|---|---|---|---|
| Direct | 499 | 46.3 | 0.57 | 0.573 | 10 of 10 | 0 of 10 |
| MCP, persistent session | 499 | 47.1 | 0.57 | 0.585 | 10 of 10 | 0 of 10 |
| MCP, new session per call | 499 | 46.2 | 1.14 | 1.138 | 10 of 10 | 0 of 10 |

Server startup (spawn, initialize, list tools), once: 0.46s (persistent session), 0.58s (per-call run, the initial tool listing).

Per-run times (secs), warm-up excluded:

- Direct: 0.59, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.58, 0.57, 0.57.
- MCP persistent: 0.57, 0.59, 0.57, 0.65, 0.58, 0.57, 0.57, 0.57, 0.61, 0.57.
- MCP per-call: 1.14, 1.14, 1.14, 1.15, 1.14, 1.14, 1.13, 1.14, 1.13, 1.13.

## The overhead MCP added in this setup

- Tokens: 0 input tokens. Every run on every path sent 499. Output tokens differ by under 1 on average (sampling noise).
- Time, persistent session: about 0.00s at the median (0.57 vs 0.57). The mean is 0.012s higher, from two runs at 0.65s and 0.61s that also generated more tokens (53 and 49).
- Time, new session per call: +0.57s per run, one call per run (1.14 vs 0.57). That is about the same as the server startup measured on its own (0.46s to 0.58s). So the cost is the process start, paid again on every call.

## What the numbers show

- The schema the model sees is the same on both paths (a test compares them), so the token count does not move. With one tool, MCP adds no input tokens. The lesson says MCP token cost grows with the number of tools exposed, so one tool understates it.
- A persistent session costs almost nothing in time. The default (a new session per call) doubled the time per run here. For a tool called on every request, that is a large fixed cost.
- The model called the tool in 30 of 30 runs and gave the right answer in 0 of 30. In every run its arguments were `hours=1,service=auth`, a 1 hour window for a 24 hour question. The arguments are legal (hours 1 is inside the schema's 1 to 168), so validation passed, and the tool correctly returned nothing. Schema validation checks shape, not meaning.
- The runs are near identical: this model repeated itself almost exactly. Ten runs show one behaviour, not a spread, so they say nothing about variance.

## Self-check: do the Triage outputs parse?

Run in the console (2026-10-20): Live mode, Tool step off, Stress none, Hunt with a maximum of 10 runs. Result as reported by the learner: no bad value found in 10 runs.

- That is 10 of 10 replies valid. A hunt stops at the first reply that fails validation on any attempt, so none of the 10 failed even once and none needed the retry.
- Question set and which questions were used: not recorded.
- Earlier, on the 5 lesson questions (2026-10-19, `python -m app.graph`): 5 of 5 valid, 0 of 5 escalated.
- Why it holds up: Ollama's constrained decoding keeps the shape right, and the prompt keeps the scale right. A bad value shows up only when the setup is weakened. With the stress settings in the console (no JSON constraint, no scale rule, a percentage request in the report, a 256 token window) the same model failed validation on the first try in 5 to 8 of 8 incidents each. Those were assistant runs, not recorded here.

## Drill and evidence

- Drill ("Your team wants to put every internal tool behind MCP. What is your position?"): answered. It led with a position, and the evidence-first and revisit-the-decision points were strong. It gave no number, and its claim that MCP is expensive was not what the measurements showed (0 tokens and about 0.00s with a persistent session, +0.57s with a new session per call). It was not timed. Marked complete at the learner's request.
- Evidence line: recorded in `notes/evidence.md`.
