"""Reports on disk, the baseline, the release gate, and the viewers for saved reports.

What lives here: atomic report and baseline files, the rules for when a baseline may
be written, `gate` (every reason a run must not ship), `format_table`, and the
viewer that prints one case end to end (`show_case`, `format_case`), compares two
saved reports case by case (`compare_reports`) or freezes a saved report as the
baseline (`freeze_baseline`).

Why separate from the runner: this is policy and presentation, not execution. It
reads and writes files but never calls a model. The CLI depends on it, not the
other way round.

Nothing here knows which app produced a report. Label scores, groups and the
app's extra aggregates are read from the report itself, so FindyFixy's
`severity_match`, `twin`/`no_twin` and `twin_recall_rate` pass through unnamed.
Every path is passed in by the caller: the harness has no idea where an app keeps
its reports or its baseline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

# The traceability rules and the control marker live in attest.py, the leaf,
# so a baseline and an attestation refuse the same runs for the same reasons.
from wyld3rness.eval.attest import (
    NO_LEGACY_CONTROL,
    UNKNOWN,
    LegacyControl,
    atomic_write_json,
    traceability_problems,
)
from wyld3rness.eval.stats import (
    CORE_AGGREGATE_KEYS,
    CORE_RECORD_KEYS,
    METRIC_NAMES,
    groups_in,
    match_name,
    ordered_score_names,
)

# Provenance keys that must match for a baseline to be comparable.
COMPARABLE_KEYS: tuple[str, ...] = ("judge_digest", "prompt_versions", "golden_sha256")
# Provenance keys NOT copied into the baseline file. Every other key is, so the
# app's own `describe()` keys (models, thresholds) reach the baseline without
# the harness naming them. These are left out because they describe how one
# run was made, not what the baseline stands for: a dirty flag, the bootstrap
# settings, and the control flag (a control run is refused anyway). An app's
# legacy control keys (see attest.LegacyControl) are left out as well.
BASELINE_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {"git_dirty", "bootstrap_seed", "bootstrap_resamples", "negative_control"}
)

class BaselineMissing(Exception):
    """The baseline file does not exist. Distinct from a failed gate."""


class BaselineRefused(Exception):
    """A baseline must not be written from this run. The message says why."""

# ---- Files ------------------------------------------------------------------
#
# The atomic writer lives in attest.py, the one harness module that must run
# alone with the standard library in CI. Reports, baselines and attestations
# then share one writer instead of two copies that could drift.


def report_path(now: datetime, directory: Path) -> Path:
    """`reports/eval-<UTC timestamp>.json`. Second resolution is enough for a sequential run."""
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"eval-{stamp}.json"


def write_report(report: Mapping[str, Any], now: datetime, directory: Path) -> Path:
    """Write the report atomically and return its path."""
    path = report_path(now, directory)
    atomic_write_json(path, report)
    return path


def load_baseline(path: Path, command: str | None = None) -> dict[str, Any]:
    """Read the baseline. Raises `BaselineMissing` with the fix, never a bare crash.

    `command` is the app's eval command, so the message names the exact fix.
    """
    if not path.exists():
        run = f"`{command} --write-baseline`" if command else "the eval with `--write-baseline`"
        raise BaselineMissing(
            f"no baseline at {path.name}. Run {run} "
            "once, review the numbers, and commit the file."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---- Baseline and gate ------------------------------------------------------


def baseline_refusals(
    report: Mapping[str, Any],
    max_failure_rate: float,
    legacy_control: LegacyControl = NO_LEGACY_CONTROL,
) -> list[str]:
    """Reasons not to write a baseline from this run. Empty means fine.

    The traceability rules are the attestation's (attest.traceability_problems):
    a control, a dirty or unknown tree, an unknown judge digest or SHA.
    """
    overall = report["aggregates"]["overall"]
    reasons = traceability_problems(report["provenance"], legacy_control)
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
    legacy_control: LegacyControl = NO_LEGACY_CONTROL,
) -> None:
    """Freeze this run as the baseline. Never called automatically.

    `legacy_control` lets an app's older control flag refuse the run and keeps
    that flag out of the baseline (see `LegacyControl`).

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
    reasons = baseline_refusals(report, max_failure_rate, legacy_control)
    if reasons:
        raise BaselineRefused("; ".join(reasons))
    overall = report["aggregates"]["overall"]["scores"]
    names = ordered_score_names(overall)
    provenance = report["provenance"]
    excluded = BASELINE_EXCLUDED_KEYS | set(legacy_control)
    payload = {
        "provenance": {k: v for k, v in provenance.items() if k not in excluded},
        "scores": {n: {"mean": overall[n]["mean"], "ci": overall[n]["ci"]} for n in names},
        "floors": {
            n: round(max(0.0, overall[n]["mean"] - floor_margin), 4)
            for n in names
            if overall[n]["mean"] is not None
        },
        "max_failure_rate": max_failure_rate,
    }
    atomic_write_json(path, payload)


def comparability_problems(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    """Which provenance fields differ, as messages. Empty means comparable."""
    problems = []
    for key in COMPARABLE_KEYS:
        now, then = report["provenance"].get(key), baseline["provenance"].get(key)
        if key == "judge_digest" and (now in UNKNOWN or then in UNKNOWN):
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


def _number(value: Any) -> str:
    """An extra aggregate: two decimals, `n/a` for None, as-is for anything else."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


# ---- Saved reports: load, show one case, freeze a baseline ------------------

# A report is a few hundred KB. A file far larger than that is not one of ours,
# and reading it whole would be an easy way to exhaust memory.
MAX_REPORT_BYTES = 20_000_000


class ReportInvalid(ValueError):
    """The file is not a report this module wrote."""


def latest_report(directory: Path) -> Path:
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
    cases = data["cases"]
    if not isinstance(cases, list) or not all(isinstance(c, dict) and isinstance(c.get("id"), str) for c in cases):
        raise ReportInvalid(f"{path.name} has a case that is not a record with an id")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ReportInvalid(f"{path.name} has duplicate case ids")
    return data


def _verdict_rows(trace: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    """One line per judge decision: [x] supported, [ ] not, then the text and reason."""
    return [
        f"    [{'x' if row.get('supported') else ' '}] {row.get(key, '')}"
        + (f"  -- {row['reason']}" if row.get("reason") else "")
        for row in trace
    ]


# The metrics in display order, with the trace key that holds each judged item.
_TRACE_KEYS: tuple[tuple[str, str], ...] = (
    ("faithfulness", "claim"),
    ("response_relevancy", "question"),
    ("context_recall", "sentence"),
    ("action_quality", "action"),
)


def record_labels(record: Mapping[str, Any]) -> list[str]:
    """The labels a record was scored on, as the record stores them in `labels`.

    Legacy fallback only: records written before 2026-11-09 have no `labels`
    key, so the names are read from key shapes. A label `L` is known by
    `expected_L`, or by `L` next to `L_match`, because the oldest saved
    reports have no `expected_L`. New records never take this path, so an app
    meta key that happens to end in `_match` is not mistaken for a label.
    """
    if "labels" in record:
        return list(record["labels"])
    expected = {k.removeprefix("expected_") for k in record if k.startswith("expected_")}
    matched = {k.removesuffix("_match") for k in record if k.endswith("_match")}
    return sorted(expected | {label for label in matched if label in record})


def _meta_value(value: Any) -> str:
    """A meta value as one short line: lists joined by commas, None as a dash."""
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "(none)"
    return str(value)


def format_case(record: Mapping[str, Any]) -> str:
    """One case end to end, as plain text, for a person to check by eye.

    Everything the model and the judge saw, and every decision the judge made.
    Labels and app meta are printed generically (`name: value`), so the viewer
    serves any app. Tolerates a run error, and reports saved before records
    carried the question and reference ("?" is printed for what is missing).
    """
    labels = record_labels(record)
    label_keys = {k for label in labels for k in (label, match_name(label), f"expected_{label}")}
    lines = [f"{record['id']}  {record.get('group', '?')}  {record.get('latency_s', 0):.1f}s"]
    lines.append(f"ALERT      {record.get('question') or '?'}")
    reference = record.get("reference") or "?"
    lines.append("REFERENCE  " + reference.replace("\n", "\n           "))
    for label in labels:
        verdict = {True: "MATCH", False: "MISS", None: "-"}[record.get(match_name(label))]
        lines.append(
            f"LABEL      {label}: expected {record.get(f'expected_{label}') or '?'}, "
            f"answered {record.get(label)}  {verdict}"
        )
    meta = [k for k in record if k not in CORE_RECORD_KEYS and k not in label_keys]
    for n, key in enumerate(meta):
        lines.append(f"{'META' if n == 0 else '':<11}{key}: {_meta_value(record[key])}")
    if record.get("run_error"):
        lines.append(f"RUN ERROR  {record['run_error']}")
        return "\n".join(lines)
    lines.append("SEEN BY THE MODEL AND THE JUDGE")
    lines.extend(f"    {c}" for c in record.get("contexts", []) or ["(no context)"])
    lines.append(f"ANSWER     {record.get('answer')}")
    for name, key in _TRACE_KEYS:
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
        elif name == "action_quality":
            lines.extend(f"    {row['rating']}  -- {row.get('reason', '')}" for row in trace)
        else:
            lines.extend(_verdict_rows(trace, key))
    return "\n".join(lines)


def show_case(case_id: str, report_path: Path | None, reports_dir: Path) -> int:
    """Print one case from a saved report. Returns the exit code.

    A very old report holds only the case id, not the question or reference.
    The harness cannot look them up: the golden file belongs to the app. It
    says so on stderr instead of showing text that may not be what was scored.
    """
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
    if "question" not in record:
        print("warning: this report predates self-contained records; "
              "the alert and reference were not saved in it", file=sys.stderr)
    print(format_case(record))
    return 0


def case_differences(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Every per-case difference between two reports that a refactor must not cause. Pure.

    Compared per case id: which cases ran and in what order, the question and
    reference, the run error, the answer text, the contexts the judge saw,
    each label with its match and expected value, and each metric's score and
    failed flag. Both reports must have unique case ids (`load_report`). Not compared: latency and timing (they always move), the
    provenance (a new run has a new timestamp and commit), and the judge's
    reasons (free text; the score is what the gate reads). Empty means the
    runs agree on everything the gate and a reader of a case can see.
    """
    old_cases = {r["id"]: r for r in old["cases"]}
    new_cases = {r["id"]: r for r in new["cases"]}
    problems = [f"{cid}: only in the first report" for cid in old_cases if cid not in new_cases]
    problems += [f"{cid}: only in the second report" for cid in new_cases if cid not in old_cases]
    shared = [cid for cid in old_cases if cid in new_cases]
    if shared != [cid for cid in new_cases if cid in old_cases]:
        problems.append("the cases ran in a different order")
    for cid in shared:
        a, b = old_cases[cid], new_cases[cid]
        # The inputs too: a changed question or reference that happens to land
        # on the same scores is still not the same run.
        fields = ["question", "reference", "run_error", "answer", "contexts"]
        for label in sorted(set(record_labels(a)) | set(record_labels(b))):
            fields += [label, match_name(label), f"expected_{label}"]
        for field in fields:
            if a.get(field) != b.get(field):
                problems.append(f"{cid}: {field} differs")
        for metric in sorted(set(a.get("metrics", {})) | set(b.get("metrics", {}))):
            ma, mb = a.get("metrics", {}).get(metric, {}), b.get("metrics", {}).get(metric, {})
            for key in ("score", "failed"):
                if ma.get(key) != mb.get(key):
                    problems.append(f"{cid}: {metric} {key} {ma.get(key)!r} -> {mb.get(key)!r}")
    return problems


def compare_reports(first: Path, second: Path) -> int:
    """Print whether two saved reports agree case by case. Returns the exit code.

    0 when they agree on every case (see `case_differences`), 1 with one line
    per difference, 2 when a file is not a report or both paths are the same
    file: a report always matches itself, so that "identical" would prove
    nothing. This is how a change that must not move a score (a faster
    runner, tracing) proves it did not.
    """
    if first.resolve() == second.resolve():
        print(f"error: both reports are {first.name}; name the other run with --report",
              file=sys.stderr)
        return 2
    try:
        a, b = load_report(first), load_report(second)
    except ReportInvalid as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    problems = case_differences(a, b)
    for problem in problems:
        print(f"DIFFERS: {problem}")
    if problems:
        print(f"compare: {len(problems)} difference(s) between {first.name} and {second.name}")
        return 1
    print(f"compare: identical, {len(a['cases'])} cases ({first.name} and {second.name}): "
          "answers, contexts, labels and every metric score")
    return 0


def freeze_baseline(
    report_path: Path, baseline_path: Path, legacy_control: LegacyControl = NO_LEGACY_CONTROL
) -> int:
    """Freeze a saved report as the baseline, with the same refusals as a live run.

    The baseline is then exactly the numbers that were reviewed, not a fresh
    run that happens to follow them. Returns the exit code.
    """
    try:
        report = load_report(report_path)
        write_baseline(report, baseline_path, legacy_control=legacy_control)
    except ReportInvalid as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except BaselineRefused as err:
        print(f"baseline refused: {err}", file=sys.stderr)
        return 1
    print(f"baseline written: {baseline_path.name} (from {report_path.name})")
    return 0


def extra_columns(report: Mapping[str, Any]) -> list[str]:
    """The app's own aggregate keys (from `summarise`), sorted. Shown after the scores."""
    overall = report["aggregates"]["overall"]
    return sorted(k for k in overall if k not in CORE_AGGREGATE_KEYS)


def format_table(report: Mapping[str, Any]) -> str:
    """Markdown table: one row per group, mean with 95% CI and failed count (f).

    A cell is `mean [low,high] fN`: N metric calls failed and are not in the mean.
    Rows are "overall", then each group in the order its first case appears.
    Columns are the scores, then the app's extra aggregates under their own key
    names, then median latency and run errors.
    """
    aggregates = report["aggregates"]
    names = ordered_score_names(aggregates["overall"]["scores"])
    extras = extra_columns(report)
    header = ["group", "n", *names, *extras, "median_s", "run_err"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    groups = [g for g in groups_in(report["cases"]) if g in aggregates]
    for group in ("overall", *groups):
        agg = aggregates[group]
        latency = agg["median_latency_s"]
        row = [
            group,
            str(agg["n"]),
            *(_cell(agg["scores"][name]) for name in names),
            *(_number(agg.get(key)) for key in extras),
            "n/a" if latency is None else f"{latency:.1f}",
            str(agg["n_run_errors"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

