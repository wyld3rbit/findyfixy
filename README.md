# LLM Production Sprint

A 10-day, hands-on course in the LLM production stack. You build one product end to end: **FindyFixy**, an incident-triage agent.

You learn by building it with Claude Code as your build partner. Each day ends with a measured number and a drill question.

## What you build

An on-call engineer pastes an incident. In seconds the agent returns severity, the likely component, the first action, and the 3 most similar past incidents. Safe actions run. Risky ones wait for a human to approve.

Built to production standards, and measured on your own machine:

- Small-model-first routing that escalates to a larger model only when needed.
- Hybrid-search memory with recall measured on a labelled set.
- An evaluation gate in CI.
- Tracing, cost control, and injection tests.
- One command to start the whole stack.

All of it is open source and runs locally. See `00-plan.md` for the full plan and the bill of materials.

## What you need

- A Mac with 16 GB of memory. The models and container sizes are chosen for it.
- [uv](https://docs.astral.sh/uv/) for Python. The project pins Python 3.13, and uv manages it.
- Docker Compose with a container runtime (the Docker app or Colima).
- [Ollama](https://ollama.com/) for local models. Day 1 pulls about 6 GB.
- [Claude Code](https://claude.com/claude-code), to build with.

## How to start

1. Clone this repository and open a terminal in its folder.
2. Run `claude`.
3. Say: `start day 1`.

To read the lessons in a browser, open `tracker.html`. It also tracks your progress.

## What is in here

```
CLAUDE.md          how Claude Code works with you on this course
00-plan.md         the plan, the arc, the bill of materials
d01.md ... d10.md  one lesson per day
tracker.html       reading and progress page (uses lessons.js)
notes/             your numbers for each day (notes/day1.md, ...)
code/              the product you build. Run its commands from here.
  CHANGELOG.md     what is done, what differs from the lessons, what is open
```

## Notes

- Secrets go in `code/.env`, which is git-ignored. Never commit it.
- All data in the course is synthetic.
- Every path in the lessons is relative to this folder.

## License

To be added by the repository owner.
