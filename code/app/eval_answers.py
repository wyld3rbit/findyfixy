"""Day 5: score FindyFixy's answers and gate a release on the scores.

What it does
    Builds the live FindyFixy adapter (app/eval_adapter.py) over the real
    database, retriever and models, and hands it to the harness command line
    (harness/eval/cli.py), which runs the cases, writes the report, and gates
    it against the committed baseline (`evals/baseline.json`).

Run it:  uv run python -m app.eval_answers [--gate | --write-baseline | --no-recall]
Every make target (`make eval`, `make eval-gate`, `make eval-baseline`,
`make eval-negative`, `make case`, `make eval-freeze`) calls this module. The
console's live eval run (app/console_eval.py) opens the same session, so the
command line and the viewer can never run different wiring.

Why so thin
    This is the composition root of the evaluation: the one place that reads
    configuration and builds real clients. The policy (metrics, gate,
    baseline) lives in the harness, and the FindyFixy logic (holdout, severity,
    twins) in the adapter. A change to either does not touch the wiring, and
    the wiring can change without a test of the policy noticing.

Failure model
    - A missing SMALL_MODEL or LARGE_MODEL becomes the harness `ConfigError`,
      so the command prints "set the environment variable ..." and exits 2.
    - The database, the retriever and the embedder are opened for the run and
      closed when it ends, even when it fails, by the session's context. (The
      console instead lends the session its own long-lived copy.)

What changes at scale
    Sequential by default: one Ollama on one Apple GPU, with the models loaded
    in shared 16 GB memory. `--workers` runs cases in parallel when the model
    server is started with OLLAMA_NUM_PARALLEL to match.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sys
from types import MappingProxyType

from app.eval_adapter import FindyFixy, Invokable, to_case
from app.events import Emit, no_emit
from app.golden import DEFAULT_PATH as GOLDEN_PATH, load_golden
from app.graph import Recall
from app.paths import EVALS, REPORTS, ROOT
from app.settings import MissingEnvironment, load_env, read_models
from app.wiring import SharedRetrieval
from harness.eval import cli
from harness.eval.cli import ConfigError, Session, SessionFactory

BASELINE_PATH = EVALS / "baseline.json"
PROG = "python -m app.eval_answers"
NO_RECALL_HELP = "negative control: run without the recall step. Never saved as a baseline."
# How FindyFixy marked a control run before the harness had `negative_control`:
# reports written before 2026-11-09 carry `recall_enabled: False` instead. The
# harness refuses such a report as a baseline and keeps the key out of one, which
# is also what lets the committed baseline be refrozen byte for byte.
LEGACY_CONTROL = MappingProxyType({"recall_enabled": False})


@contextmanager
def open_live_session(
    negative_control: bool, *, emit: Emit = no_emit, retrieval: SharedRetrieval | None = None
) -> Iterator[Session]:
    """Open the real database, retriever and embedder, and yield the live adapter.

    Environment: SMALL_MODEL and LARGE_MODEL (required), JUDGE_MODEL (default
    qwen2.5:7b). Everything heavy is imported here, so importing this module
    stays cheap. The retriever is warmed once before the harness stamps the
    time, so model loading does not land in case 1's latency. The Ollama models
    still load on first use, so case 1 may be slower.

    `negative_control` builds the graph without the recall step (`--no-recall`).
    Scores must fall and the gate must fail. The harness records the flag in
    provenance, and such a run cannot be saved as a baseline.

    `emit` receives the graph's own run events (app/events.py) for every case.
    The command line passes none. The console passes its stream, so the app
    lane of the page moves with each case from real events.

    `retrieval` is the process's shared database, embedder and retriever
    (app/wiring.py). The console passes the one its triage runs already use,
    so an eval run loads no second copy, and it stays open after the session.
    The command line passes none: the session builds its own and closes it at
    the end, even when the run fails. Either way the shared object's lock keeps
    retrieval and embedding to one thread at a time, so `--workers` is safe.
    """
    from app.graph import build_graph
    from app.llm import ollama_ask

    load_env()
    try:
        models = read_models()
    except MissingEnvironment as err:
        raise ConfigError(f"set the environment variable {err}") from err
    small, large = models.small, models.large
    goldens = load_golden(GOLDEN_PATH)
    owned = retrieval is None
    shared = SharedRetrieval() if retrieval is None else retrieval
    try:
        shared.candidates(goldens[0].question, 1)  # warm-up, not measured

        def make_graph(recall: Recall) -> Invokable:
            """The production graph with recall bound to one case, or none for the control."""
            return build_graph(
                ollama_ask,
                small_model=small,
                large_model=large,
                emit=emit,
                recall=None if negative_control else recall,
            )

        sut = FindyFixy(make_graph=make_graph, candidates=shared.candidates, small_model=small, large_model=large)
        yield Session(sut=sut, cases=[to_case(g) for g in goldens], judge_model=models.judge, embed=shared.embed)
    finally:
        if owned:
            shared.close()


def main(
    argv: Sequence[str] | None = None,
    *,
    open_session: SessionFactory = open_live_session,
    reports_dir: Path = REPORTS,
    baseline_path: Path = BASELINE_PATH,
) -> int:
    """Run the harness command line on the live FindyFixy wiring. Returns the exit code.

    The session factory and the paths are injected so tests drive this with a
    fake adapter and a temp directory.
    """
    return cli.main(
        argv,
        prog=PROG,
        run=cli.live_runner(open_session, golden_path=GOLDEN_PATH, repo_dir=ROOT),
        reports_dir=reports_dir,
        baseline_path=baseline_path,
        display_root=ROOT,
        control_flag="--no-recall",
        control_help=NO_RECALL_HELP,
        legacy_control=LEGACY_CONTROL,
    )


if __name__ == "__main__":
    sys.exit(main())
