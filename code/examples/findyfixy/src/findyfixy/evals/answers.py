"""Day 5: score FindyFixy's answers and gate a release on the scores.

What it does
    Builds the live FindyFixy adapter (findyfixy/evals/adapter.py) over the real
    database, retriever and models, and hands it to the harness command line
    (wyld3rness/eval/cli.py), which runs the cases, writes the report, and gates
    it against the committed baseline (`evals/baseline.json`).

Run it:  uv run findyfixy-eval [--gate | --write-baseline | --no-recall]
Every make target (`make eval`, `make eval-gate`, `make eval-baseline`,
`make eval-negative`, `make case`, `make eval-freeze`) calls this module. The
console's live eval run (findyfixy/console/eval_run.py) opens the same session, so the
command line and the viewer can never run different wiring.

Why so thin
    This is the composition root of the evaluation: the one place that reads
    configuration and builds real clients. The policy (metrics, gate,
    baseline) lives in the harness, and the FindyFixy logic (holdout, severity,
    twins) in the adapter. A change to either does not touch the wiring, and
    the wiring can change without a test of the policy noticing.

What the attestation covers
    `make eval-gate` writes `evals/attestation.json` on a pass (see
    wyld3rness/eval/attest.py). The harness computes the code half of the
    answer-affecting set: the import closure of `ATTEST_ENTRY`. This module
    declares the other half, `ATTEST_EXTRA`: files read at runtime, which no
    import statement reveals. The console, the stress test and the MCP
    modules are outside the closure, so changing them needs no new run.

    One input is not a file: the database. The run reads its passages from
    Postgres, loaded by `make ingest` from `data/incidents.jsonl`. A table
    loaded from an older corpus would pass with today's file hashes, so
    `check_database` runs ingest's own diff before the run and again before
    the attestation is written. If a re-ingest would change anything, the
    gate does not run and says "the database is stale: run make ingest".

    Why `.env.example` is not declared: the run never reads it. The model
    names come from the developer's own `.env`, which CI cannot see and which
    is never committed. So the attestation records the models and the judge
    digest the run actually used, and the gate's comparability check
    (judge digest, prompts, golden set) guards what a name cannot.

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

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from types import MappingProxyType

from findyfixy.evals.adapter import FindyFixy, Invokable, to_case
from findyfixy.events import Emit, no_emit
from findyfixy.evals.golden import DEFAULT_PATH as GOLDEN_PATH, load_golden
from findyfixy.triage.graph import Recall
from findyfixy.paths import EVALS, REPORTS, ROOT, WORKSPACE
from findyfixy.settings import MissingEnvironment, load_env, read_models
from findyfixy.observability.adapter import RunTracer
from findyfixy.observability.tracing import open_tracing
from findyfixy.wiring import SharedRetrieval, stage_spans
from wyld3rness.eval import cli
from wyld3rness.eval.attest import Inputs
from wyld3rness.eval.cli import AttestationSpec, ConfigError, Session, SessionFactory
from wyld3rness.trace.scores import LangfuseScores
from wyld3rness.trace.tracing import Tracing

BASELINE_PATH = EVALS / "baseline.json"
ATTESTATION_PATH = EVALS / "attestation.json"
# The eval path: this module's import closure reaches the adapter, the graph,
# triage, retrieval, ingest (its chunking and embedding model decide what the
# table holds) and the harness itself.
ATTEST_ENTRY: tuple[str, ...] = ("findyfixy.evals.answers",)
# Where first-party packages live, relative to the workspace root (code/): the
# harness and this app. The closure follows imports across both, so a harness
# change on the eval path needs a new run just like an app change.
ATTEST_SOURCE_ROOTS: tuple[str, ...] = ("src", "examples/findyfixy/src")
# What the eval path reads that no import shows, relative to the workspace root (code/).
#
# The whole data directory, not a hand-picked list: a file added there is
# covered without anyone remembering it, at the cost of a rerun when an
# unused file (the retrieval eval's queries) changes. A test fails when code
# on the eval path builds a path from findyfixy/paths.py that falls outside this set.
#
# The residual gap: a path that is not built from those constants (an
# environment variable, a string, a model downloaded by name into a cache) is
# invisible to that test. The embedding and reranker weights are pinned by
# name and revision in the code, which is fingerprinted, not by their bytes.
ATTEST_EXTRA: tuple[str, ...] = (
    "examples/findyfixy/data/",  # the golden set, the corpus `make ingest` loads (and its twin pairs), the deploy tool's data
    "examples/findyfixy/evals/baseline.json",  # what the gate compared against
    "uv.lock",  # library versions change answers (tokenizers, rerankers, clients)
    "examples/findyfixy/docker-compose.yml",  # pins the Postgres and pgvector image the retrieval runs on
)
PROG = "findyfixy-eval"
NO_RECALL_HELP = "negative control: run without the recall step. Never saved as a baseline."
# How FindyFixy marked a control run before the harness had `negative_control`:
# reports written before 2026-11-09 carry `recall_enabled: False` instead. The
# harness refuses such a report as a baseline and keeps the key out of one, which
# is also what lets the committed baseline be refrozen byte for byte.
LEGACY_CONTROL = MappingProxyType({"recall_enabled": False})


def check_database() -> list[str]:
    """Why the live database does not hold the current corpus, or nothing.

    One short read-only connection, no model loaded: ingest's own diff of
    `data/incidents.jsonl` against the stored hashes and model names. A
    database that cannot be reached is a reason too, with the fix.
    """
    import psycopg

    from findyfixy.retrieval.ingest import database_staleness
    from findyfixy.wiring import connect

    try:
        with connect() as conn:
            return database_staleness(conn)
    except psycopg.Error as err:
        return [f"cannot check that the database is current ({type(err).__name__}): run make up, then make ingest"]


@contextmanager
def open_live_session(
    negative_control: bool,
    *,
    emit: Emit = no_emit,
    retrieval: SharedRetrieval | None = None,
    tracing: Tracing | None = None,
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

    `emit` receives the graph's own run events (findyfixy/events.py) for every case.
    The command line passes none. The console passes its stream, so the app
    lane of the page moves with each case from real events.

    `retrieval` is the process's shared database, embedder and retriever
    (findyfixy/wiring.py). The console passes the one its triage runs already use,
    so an eval run loads no second copy, and it stays open after the session.
    The command line passes none: the session builds its own and closes it at
    the end, even when the run fails. Either way the shared object's lock keeps
    retrieval and embedding to one thread at a time, so `--workers` is safe.

    `tracing` works the same way: the console passes the provider it already
    runs. The command line passes none, and the session starts one when
    tracing is configured (`open_tracing`) and closes it at the end, which
    sends what is queued. With tracing, every case's graph run is a trace
    under the harness's case span, and the scores land on it.
    """
    from findyfixy.triage.graph import build_graph
    from findyfixy.llm import ollama_ask

    load_env()
    try:
        models = read_models()
    except MissingEnvironment as err:
        raise ConfigError(f"set the environment variable {err}") from err
    small, large = models.small, models.large
    goldens = load_golden(GOLDEN_PATH)
    owned = retrieval is None
    own_tracing = ExitStack()
    shared: SharedRetrieval | None = None
    scores: LangfuseScores | None = None
    try:
        if tracing is None:
            tracing = own_tracing.enter_context(open_tracing())
        run_tracer = RunTracer(tracing.tracer()) if tracing is not None else None
        if run_tracer is not None:
            emit = run_tracer.wrap(emit)
        # The score backend is chosen here, in the composition root; the
        # harness only sees the ScoreSink seam.
        scores = LangfuseScores(tracing.config) if tracing is not None else None
        shared = SharedRetrieval(stage=stage_spans(tracing)) if retrieval is None else retrieval
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

        sut = FindyFixy(make_graph=make_graph, candidates=shared.candidates, small_model=small,
                        large_model=large, run_tracer=run_tracer)
        yield Session(sut=sut, cases=[to_case(g) for g in goldens], judge_model=models.judge,
                      embed=shared.embed, tracing=tracing, scores=scores)
    finally:
        try:
            if owned and shared is not None:
                shared.close()
            if scores is not None:
                scores.close()
                if scores.dropped:
                    print(f"warning: {scores.dropped} score(s) not sent to the trace backend", file=sys.stderr)
        finally:
            own_tracing.close()  # last, and even if a close above raised: queued spans still go out


def main(
    argv: Sequence[str] | None = None,
    *,
    open_session: SessionFactory = open_live_session,
    reports_dir: Path = REPORTS,
    baseline_path: Path = BASELINE_PATH,
    attestation_path: Path = ATTESTATION_PATH,
    database_check: Callable[[], Sequence[str]] = check_database,
) -> int:
    """Run the harness command line on the live FindyFixy wiring. Returns the exit code.

    The session factory and the paths are injected so tests drive this with a
    fake adapter and a temp directory. The attestation is fingerprinted over
    the real tree (`WORKSPACE`) either way; only where it is written moves.
    `database_check` is the attestation's precondition; tests pass a fake.
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
        attestation=AttestationSpec(
            path=attestation_path,
            root=WORKSPACE,
            inputs=Inputs(ATTEST_ENTRY, ATTEST_EXTRA, ATTEST_SOURCE_ROOTS),
            case_ids=lambda: [g.id for g in load_golden(GOLDEN_PATH)],
            preconditions=database_check,
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
