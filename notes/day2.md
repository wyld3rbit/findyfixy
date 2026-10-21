# Day 2 notes: structured output and tools

## Direct path: tool call with the small model

Run command: `uv run python -m app.direct_run` (from `code/`), 2026-10-20.
Model: `qwen2.5:1.5b`. One warm-up run (excluded), then 10 measured runs.
Task (fixed): "Which deploys to the auth service happened in the last 24 hours? Use the get_recent_deploys tool, then answer in one sentence."
Right answer from `data/deploys.json`: auth 2.14.1 (3h ago) only. Auth 2.14.0 was 30h ago, outside the window.

| Run | Input tokens | Output tokens | Seconds | Turns | Tool calls | Tool errors | Correct |
|---|---|---|---|---|---|---|---|
| warm-up | 499 | 46 | 0.84 | 2 | 1 | 0 | no (excluded) |
| 1 | 499 | 46 | 0.57 | 2 | 1 | 0 | no |
| 2 | 499 | 46 | 0.56 | 2 | 1 | 0 | no |
| 3 | 499 | 46 | 0.58 | 2 | 1 | 0 | no |
| 4 | 499 | 46 | 0.57 | 2 | 1 | 0 | no |
| 5 | 499 | 46 | 0.57 | 2 | 1 | 0 | no |
| 6 | 499 | 46 | 0.57 | 2 | 1 | 0 | no |
| 7 | 499 | 46 | 0.56 | 2 | 1 | 0 | no |
| 8 | 499 | 47 | 0.58 | 2 | 1 | 0 | no |
| 9 | 499 | 46 | 0.56 | 2 | 1 | 0 | no |
| 10 | 499 | 46 | 0.56 | 2 | 1 | 0 | no |

- Called the tool: 10 of 10. Correct answer: 0 of 10. Hit the turn cap: 0.
- Median 0.57s. Mean tokens in and out: 499 and 46.1.
- This is the direct (in-process) baseline. The MCP column is still to measure (Step 5).

## What the numbers show

- The model always used the tool, with no tool errors, and never gave the right answer.
- The runs are almost identical: 499 input tokens every time, 46 output tokens in 9 of 10, 0.56s to 0.58s. Ten runs show no spread, so they say nothing about sampling variance. They show one behaviour repeated.
- In an earlier check run (2 runs, not in the table) the tool call carried `hours: 1`, not 24. A window of 1 hour holds no auth deploy, so the tool returned nothing and the model answered "no deploys". That fits this table, but this run did not log the arguments, so it is not confirmed for these 10.
- The arguments were legal: `hours` 1 is inside the schema's 1 to 168. Schema validation checks shape, not meaning. Nothing in the loop can tell "legal but wrong" from "right".
- Input tokens (499) include the tool's name, description and JSON Schema, sent on every turn. With one tool this is small. MCP overhead grows with the number of tools, so a one-tool setup will understate it.

## Not yet measured

- MCP path: tokens and time for the same task (Step 5), and the overhead against the 499 and 0.57s above.
- Self-check: 10 of 10 live outputs parse into `Triage`.
- Tool-call arguments per run (`hours` in particular).
