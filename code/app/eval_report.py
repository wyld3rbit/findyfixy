"""Reports on disk, the baseline, the release gate, and the viewers for saved reports.

What lives here: atomic report and baseline files, the rules for when a baseline may
be written, `gate` (every reason a run must not ship), `format_table`, and the
viewer that prints one case end to end (`show_case`, `format_case`) or freezes a
saved report as the baseline (`freeze_baseline`).

Why separate from the runner: this is policy and presentation, not execution. It
reads and writes files but never calls a model. The runner (eval_answers) depends on
it, not the other way round.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from app.eval_provenance import file_sha256
from app.eval_stats import GROUPS, METRIC_NAMES, SCORE_NAMES
from app.golden import DEFAULT_PATH as GOLDEN_PATH, GoldenCase, load_golden
from app.paths import EVALS, REPORTS

REPORTS_DIR = REPORTS
BASELINE_PATH = EVALS / "baseline.json"

# Provenance keys that must match for a baseline to be comparable.
COMPARABLE_KEYS: tuple[str, ...] = ("judge_digest", "prompt_versions", "golden_sha256")
# Provenance keys copied into the baseline file.
BASELINE_PROVENANCE_KEYS: tuple[str, ...] = (
    "judge_model",
    "judge_digest",
    "prompt_versions",
    "golden_sha256",
    "small_model",
    "large_model",
    "threshold",
    "recall_k",
    "git_sha",
    "timestamp_utc",
)



class BaselineMissing(Exception):
    """The baseline file does not exist. Distinct from a failed gate."""


class BaselineRefused(Exception):
    """A baseline must not be written from this run. The message says why."""




# ---- Files ------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON to a temp file in the same directory, then rename over `path`.

    The rename is atomic on one filesystem, so a reader or a crash never sees
    a half-written report or baseline. The temp file is removed on failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def report_path(now: datetime, directory: Path = REPORTS_DIR) -> Path:
    """`reports/eval-<UTC timestamp>.json`. Second resolution is enough for a sequential run."""
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"eval-{stamp}.json"


def write_report(report: Mapping[str, Any], now: datetime, directory: Path = REPORTS_DIR) -> Path:
    """Write the report atomically and return its path."""
    path = report_path(now, directory)
    _atomic_write_json(path, report)
    return path


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    """Read the baseline. Raises `BaselineMissing` with the fix, never a bare crash."""
    if not path.exists():
        raise BaselineMissing(
            f"no baseline at {path.name}. Run `python -m app.eval_answers --write-baseline` "
            "once, review the numbers, and commit the file."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---- Baseline and gate ------------------------------------------------------


def baseline_refusals(report: Mapping[str, Any], max_failure_rate: float) -> list[str]:
    """Reasons not to write a baseline from this run. Empty means fine."""
    overall = report["aggregates"]["overall"]
    reasons = []
    if report["provenance"].get("recall_enabled") is False:
        reasons.append("recall was disabled for this run (negative control)")
    if report["provenance"].get("git_dirty") is True:
        reasons.append("the working tree had uncommitted changes, so the code is not traceable")
    for key in ("judge_digest", "git_sha"):
        if report["provenance"].get(key) in (None, "unknown"):
            reasons.append(f"{key} is unknown, so this baseline could not be traced or compared later")
    if overall["n_run_errors"]:
        reasons.append(f"{overall['n_run_errors']} case(s) had a run error")
    for name in METRIC_NAMES:
        rate = overall["scores"][name]["failure_rate"]
        if rate is not None and rate > max_failure_rate:
            reasons.append(f"{name} failure rate {rate:.2f} is above {max_failure_rate:.2f}")
    return reasons


def write_baseline(
    report: Mapping[str, Any],
    path: Path,
    *,
    floor_margin: float = 0.05,
    max_failure_rate: float = 0.15,
) -> None:
    """Freeze this run as the baseline. Never called automatically.

    Floors are the overall mean minus a margin, clamped at 0. They are
    absolute minimums: a release below them fails even if it matches the
    baseline. Refuses a run with run errors or a failure rate above the cap,
    because the numbers would not be trustworthy. A score with no mean gets
    no floor.

    Why the margin is a flat 0.05 and not tied to the confidence interval: the
    interval describes how 30 cases happen to sample, which is wide (about
    0.15). A floor guards against a rerun of the same code, and at temperature
    0 the judge-based means moved by less than 0.003 between reruns (measured
    2026-10-26). The interval-based test is the regression rule, not this one.
    """
    reasons = baseline_refusals(report, max_failure_rate)
    if reasons:
        raise BaselineRefused("; ".join(reasons))
    overall = report["aggregates"]["overall"]["scores"]
    provenance = report["provenance"]
    payload = {
        "provenance": {k: provenance[k] for k in BASELINE_PROVENANCE_KEYS if k in provenance},
        "scores": {n: {"mean": overall[n]["mean"], "ci": overall[n]["ci"]} for n in SCORE_NAMES},
        "floors": {
            n: round(max(0.0, overall[n]["mean"] - floor_margin), 4)
            for n in SCORE_NAMES
            if overall[n]["mean"] is not None
        },
        "max_failure_rate": max_failure_rate,
    }
    _atomic_write_json(path, payload)


def comparability_problems(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Which provenance fields differ, as messages. Empty means comparable."""
    problems = []
    for key in COMPARABLE_KEYS:
        now, then = report["provenance"].get(key), baseline["provenance"].get(key)
        if key == "judge_digest" and (now in (None, "unknown") or then in (None, "unknown")):
            # Two unknowns are not equal: neither side shows which judge scored.
            problems.append("judge_digest is unknown, so the judge cannot be shown to be the same")
        elif now != then:
            problems.append(f"{key} differs from the baseline")
    if problems:
        problems.append(
            "baseline is not comparable: re-baseline deliberately "
            "(--write-baseline) after reviewing the change; means were not compared"
        )
    return problems


def gate(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Return every reason to block the release. Empty list means pass.

    (a) a mean below its absolute floor
    (b) regression: mean below the baseline mean AND the baseline mean above
        the current 95% CI upper bound (so noise alone does not fail it)
    (c) a metric failure rate above `max_failure_rate`
    (d) any run error
    (e) the baseline was made with a different judge, prompts or golden set
    When (e) fires, (a) and (b) are skipped: the comparison is meaningless.
    """
    problems: list[str] = []
    overall = report["aggregates"]["overall"]
    scores = overall["scores"]

    if overall["n_run_errors"]:
        errors = [f"{c['id']}: {c['run_error']}" for c in report["cases"] if c["run_error"]]
        problems.append(f"{len(errors)} run error(s): " + "; ".join(errors))

    cap = baseline["max_failure_rate"]
    for name in METRIC_NAMES:
        rate = scores[name]["failure_rate"]
        if rate is not None and rate > cap:
            problems.append(f"{name}: failure rate {rate:.2f} is above the cap {cap:.2f}")

    incomparable = comparability_problems(report, baseline)
    if incomparable:
        return problems + incomparable

    for name, floor in baseline["floors"].items():
        mean = scores[name]["mean"]
        if mean is None:
            problems.append(f"{name}: no scored cases, cannot check the floor {floor:.2f}")
        elif mean < floor:
            problems.append(f"{name}: mean {mean:.3f} is below the floor {floor:.2f}")

    for name, then in baseline["scores"].items():
        mean, ci, base = scores[name]["mean"], scores[name]["ci"], then["mean"]
        if mean is None or base is None or ci is None:
            continue  # no interval at n < 2: a regression cannot be shown
        if mean < base and base > ci[1]:
            problems.append(
                f"{name}: regression. mean {mean:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], "
                f"baseline {base:.3f} is above the interval"
            )
    return problems


# ---- Output -----------------------------------------------------------------


def _cell(stat: Mapping[str, Any]) -> str:
    """`0.83 [0.70,0.93] f2`, or `n/a` with no scored cases. CI may be None."""
    if stat["mean"] is None:
        return f"n/a f{stat['n_failed']}"
    ci = stat["ci"]
    span = f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else " [n/a]"
    return f"{stat['mean']:.2f}{span} f{stat['n_failed']}"


# ---- Saved reports: load, show one case, freeze a baseline ------------------

# A report is a few hundred KB. A file far larger than that is not one of ours,
# and reading it whole would be an easy way to exhaust memory.
MAX_REPORT_BYTES = 20_000_000


class ReportInvalid(ValueError):
    """The file is not a report this module wrote."""


def latest_report(directory: Path = REPORTS_DIR) -> Path:
    """The newest report file by name (names carry a UTC timestamp)."""
    found = sorted(directory.glob("eval-*.json"))
    if not found:
        raise ReportInvalid(f"no reports in {directory}")
    return found[-1]


def load_report(path: Path) -> dict[str, Any]:
    """Read a saved report. Raises `ReportInvalid` for anything that is not one."""
    try:
        if path.stat().st_size > MAX_REPORT_BYTES:
            raise ReportInvalid(f"{path.name} is too large to be a report")
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise ReportInvalid(f"cannot read {path.name}: {err}") from err
    if not isinstance(data, dict) or not {"provenance", "aggregates", "cases"} <= data.keys():
        raise ReportInvalid(f"{path.name} is missing provenance, aggregates or cases")
    return data


def _verdict_rows(trace: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    """One line per judge decision: [x] supported, [ ] not, then the text and reason."""
    return [
        f"    [{'x' if row.get('supported') else ' '}] {row.get(key, '')}"
        + (f"  -- {row['reason']}" if row.get("reason") else "")
        for row in trace
    ]


def format_case(record: Mapping[str, Any], golden: GoldenCase | None) -> str:
    """One case end to end, as plain text, for a person to check by eye.

    Everything the model and the judge saw, and every decision the judge made.
    Tolerates a run error and reports saved before traces existed.
    """
    verdict = {True: "MATCH", False: "MISS", None: "-"}[record.get("severity_match")]
    question = record.get("question") or (golden.question if golden else "?")
    reference = record.get("reference") or (golden.reference if golden else "")
    expected = record.get("expected_severity") or (golden.expected_severity if golden else "?")
    head = (
        f"{record['id']}  {record['group']}  source {record['source_id']}  "
        f"expected {expected}  "
        f"answered {record.get('severity')}  {verdict}  "
        f"model {record.get('model_used')}  {record.get('latency_s', 0):.1f}s"
    )
    twin = record.get("twin_id")
    lines = [head, f"ALERT      {question}"]
    if reference:
        lines.append("REFERENCE  " + reference.replace("\n", "\n           "))
    recalled = ", ".join(
        f"{i} (twin)" if i == twin else i for i in record.get("recalled_ids", [])
    )
    lines.append(f"RECALLED   {recalled or 'nothing'}")
    if record.get("run_error"):
        lines.append(f"RUN ERROR  {record['run_error']}")
        return "\n".join(lines)
    lines.append("SEEN BY THE MODEL AND THE JUDGE")
    lines.extend(f"    {c}" for c in record.get("contexts", []) or ["(nothing was recalled)"])
    lines.append(f"ANSWER     {record.get('answer')}")
    for name, key in (
        ("faithfulness", "claim"),
        ("response_relevancy", "question"),
        ("context_recall", "sentence"),
    ):
        m = record.get("metrics", {}).get(name, {})
        if not m:
            lines.append(f"{name}: not recorded")
            continue
        if m.get("failed") or m.get("score") is None:
            lines.append(f"{name}: FAILED ({m.get('error') or 'no reason recorded'})")
            continue
        lines.append(f"{name}: {m['score']:.3f}")
        trace = m.get("trace", [])
        if name == "response_relevancy":
            lines.extend(f"    {row['similarity']:.2f}  {row['question']}" for row in trace)
        else:
            lines.extend(_verdict_rows(trace, key))
    return "\n".join(lines)


def show_case(case_id: str, report_path: Path | None, reports_dir: Path) -> int:
    """Print one case from a saved report. Returns the exit code."""
    try:
        report = load_report(report_path or latest_report(reports_dir))
    except ReportInvalid as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    record = next((r for r in report["cases"] if r.get("id") == case_id), None)
    if record is None:
        known = ", ".join(r.get("id", "?") for r in report["cases"])
        print(f"error: no case {case_id!r} in that report. Cases: {known}", file=sys.stderr)
        return 2
    golden = None
    if "question" not in record:
        # An older report holds only the id, so the question and reference come
        # from the golden file as it is now. Say so if that file has changed.
        try:
            golden = {c.id: c for c in load_golden()}.get(case_id)
            if report["provenance"].get("golden_sha256") != file_sha256(GOLDEN_PATH):
                print("warning: the golden set changed after this report was made; "
                      "the alert and reference below may not be what was scored", file=sys.stderr)
        except OSError:
            print("warning: the golden file is missing; alert and reference are not shown", file=sys.stderr)
    print(format_case(record, golden))
    return 0


def freeze_baseline(report_path: Path, baseline_path: Path) -> int:
    """Freeze a saved report as the baseline, with the same refusals as a live run.

    The baseline is then exactly the numbers that were reviewed, not a fresh
    run that happens to follow them. Returns the exit code.
    """
    try:
        report = load_report(report_path)
        write_baseline(report, baseline_path)
    except ReportInvalid as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except BaselineRefused as err:
        print(f"baseline refused: {err}", file=sys.stderr)
        return 1
    print(f"baseline written: {baseline_path.name} (from {report_path.name})")
    return 0


def format_table(report: Mapping[str, Any]) -> str:
    """Markdown table: one row per group, mean with 95% CI and failed count (f).

    A cell is `mean [low,high] fN`: N metric calls failed and are not in the mean.
    """
    header = ["group", "n", "severity_match", *METRIC_NAMES, "twin_recall", "median_s", "run_err"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for group in ("overall", *GROUPS):
        agg = report["aggregates"][group]
        recall = agg["twin_recall_rate"]
        latency = agg["median_latency_s"]
        row = [
            group,
            str(agg["n"]),
            *(_cell(agg["scores"][name]) for name in SCORE_NAMES),
            "n/a" if recall is None else f"{recall:.2f}",
            "n/a" if latency is None else f"{latency:.1f}",
            str(agg["n_run_errors"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
