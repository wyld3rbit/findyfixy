"""Day 5: the golden set. 30 holdout cases for scoring the triage agent.

What it does
    Builds `data/golden.jsonl`: 30 cases, each a pasted incident report
    (`question`), the correct triage (`reference`) and the correct severity.

Why a fixed golden set
    A score is only comparable across runs if the questions never change.
    The set is generated from the corpus with fixed rules, so the file is
    reproducible and a test can pin it. A hand-picked set would drift, and
    nobody could say how it was chosen.

Two groups, because recall helps them differently
    twin (20): one case per seeded pair. The source incident is removed from
        recall at eval time (holdout) and its twin stays. A good retriever
        can still find the same incident told in the other voice. This group
        measures what memory adds.
    no_twin (10): incidents with no twin and no pair. Recall can only return
        a similar incident, never the same one. This group is the control.
        If memory helps twins but not these, the gain is near-duplicate
        lookup, not understanding.

How a case is made
    question: the first timeline line only, the alert a responder is paged
        with. The summary and the rest of the timeline are left out: they are
        written after the fact and state the cause and the fix, so the agent
        would be handed the answer.
    reference: root cause plus remediation. This is what the judge compares
        the agent's answer to.
    expected_severity: the incident's own severity.

Limits, so nobody over-reads a score
    Data is templated, so scores will look better than on real incidents.
    The alert line is short, so recall has little to match on. A low twin
    recall rate is a real property of the system, not a flaw in the set.
    The first timeline line is almost always a symptom, not a diagnosis, but a
    few mention a cause or a service by name. Absolute scores are a ceiling.
    Compare runs, not against the outside world.
    Holdout removes the source id only. A twin is deliberately NOT removed.
    Other incidents of the same scenario (same pattern, different facts)
    also stay in recall, as they would in production.

Separation
    This file reads `twin_of`. Retrieval code must never import it.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel

from app.corpus import Incident
from app.paths import DATA
from app.triage import Severity

INCIDENTS_PATH = DATA / "incidents.jsonl"
DEFAULT_PATH = DATA / "golden.jsonl"

TWIN_CASES = 20
NO_TWIN_CASES = 10
# Fixed so the no_twin pick never changes. Change it and the file changes.
SEED = 5
# The triage prompt caps input at 3500 characters. Stay clear of the cap.
MAX_QUESTION_CHARS = 3000

# One definition of severity for the whole app (app.triage). Redefining it here
# would let the golden set and the triage record drift apart.
VALID_SEVERITIES: tuple[str, ...] = get_args(Severity)


class GoldenCase(BaseModel):
    """One holdout case. `twin_id` is the pair partner recall may return."""

    id: str
    source_id: str
    group: Literal["twin", "no_twin"]
    twin_id: str | None
    question: str
    reference: str
    expected_severity: Severity


def read_incidents(path: Path = INCIDENTS_PATH) -> list[Incident]:
    """Read the corpus file into `Incident` rows. Not at import: fail loudly at use."""
    with path.open(encoding="utf-8") as handle:
        return [Incident(**json.loads(line)) for line in handle if line.strip()]


def format_question(incident: Incident) -> str:
    """The alert a responder is paged with: the first timeline line, nothing else.

    The summary and later timeline lines are written with hindsight and name
    the cause or the fix. Raises ValueError if the timeline is empty.
    """
    lines = [line.strip() for line in incident.timeline.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{incident.id} has an empty timeline")
    return lines[0]


def format_reference(incident: Incident) -> str:
    """The expert triage: why it happened, then what to do."""
    return f"Root cause: {incident.root_cause}\nRemediation: {incident.remediation}"


def seeded_pairs(incidents: Sequence[Incident]) -> list[tuple[str, str]]:
    """Return (original, twin) id pairs, sorted by original id.

    The `twin_of` field sits on the later row and points at the original.
    """
    pairs = [(i.twin_of, i.id) for i in incidents if i.twin_of]
    return sorted(pairs)


def pick_no_twin(incidents: Sequence[Incident], count: int, seed: int = SEED) -> list[Incident]:
    """Pick `count` unpaired incidents, spread over services and severities.

    Greedy and seeded. Shuffle the candidates with a fixed seed, then take
    the one that adds the most unseen service and unseen severity, repeating
    until `count`. Ties go to the earlier shuffled row, so the result is
    stable. At scale this is a cheap stand-in for stratified sampling.
    """
    paired = {x for pair in seeded_pairs(incidents) for x in pair}
    pool = [i for i in sorted(incidents, key=lambda i: i.id) if i.id not in paired]
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} unpaired incidents, need {count}")
    random.Random(seed).shuffle(pool)

    chosen: list[Incident] = []
    services: set[str] = set()
    severities: set[str] = set()
    for _ in range(count):
        best = max(
            range(len(pool)),
            key=lambda n: ((pool[n].service not in services) + (pool[n].severity not in severities), -n),
        )
        pick = pool.pop(best)
        services.add(pick.service)
        severities.add(pick.severity)
        chosen.append(pick)
    return sorted(chosen, key=lambda i: i.id)


def build_golden(incidents: Sequence[Incident]) -> list[GoldenCase]:
    """Build the 30 cases: 20 twin first (g01..g20), then 10 no_twin (g21..g30).

    Pure and deterministic. For twin pair k (sorted by original id), an even
    k uses the original as the source and an odd k uses the twin row, so the
    holdout is tested in both directions: with the earlier voice held out and
    with the later one. Raises ValueError when the corpus has too few pairs
    or too few unpaired incidents.
    """
    by_id = {i.id: i for i in incidents}
    pairs = seeded_pairs(incidents)
    if len(pairs) < TWIN_CASES:
        raise ValueError(f"only {len(pairs)} seeded pairs, need {TWIN_CASES}")

    entries: list[tuple[Incident, Literal["twin", "no_twin"], str | None]] = []
    for k, (original, twin) in enumerate(pairs[:TWIN_CASES]):
        source, partner = (original, twin) if k % 2 == 0 else (twin, original)
        entries.append((by_id[source], "twin", partner))
    entries += [(i, "no_twin", None) for i in pick_no_twin(incidents, NO_TWIN_CASES)]

    return [
        GoldenCase(
            id=f"g{n:02d}",
            source_id=source.id,
            group=group,
            twin_id=twin_id,
            question=format_question(source),
            reference=format_reference(source),
            expected_severity=source.severity,
        )
        for n, (source, group, twin_id) in enumerate(entries, start=1)
    ]


def write_golden(cases: Sequence[GoldenCase], path: Path = DEFAULT_PATH) -> None:
    """Write one JSON object per line. Sorted keys and a fixed layout keep bytes stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(), sort_keys=True) + "\n")


def load_golden(path: Path = DEFAULT_PATH) -> list[GoldenCase]:
    """Read the file back. The eval runner uses this."""
    with path.open(encoding="utf-8") as handle:
        return [GoldenCase.model_validate_json(line) for line in handle if line.strip()]


def format_cases(cases: Sequence[GoldenCase]) -> str:
    """The cases as plain text a person can read and check, one block per case.

    Read-only. It exists so a human can verify the set by eye: that each
    question is only the alert, and each reference is the real answer.
    """
    blocks = []
    for c in cases:
        link = f"twin {c.twin_id}" if c.twin_id else "no twin"
        blocks.append(
            f"{c.id}  {c.group}  source {c.source_id}  ({link})  expected {c.expected_severity}\n"
            f"  QUESTION  {c.question}\n"
            f"  REFERENCE {c.reference.replace(chr(10), chr(10) + '            ')}"
        )
    return "\n\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> None:
    """Rebuild `data/golden.jsonl`, or with `--show` print the committed one."""
    args = list(argv) if argv is not None else sys.argv[1:]
    if args == ["--show"]:
        print(format_cases(load_golden()))
        return
    if args:
        raise SystemExit("usage: python -m app.golden [--show]")
    cases = build_golden(read_incidents())
    write_golden(cases)
    twins = sum(c.group == "twin" for c in cases)
    print(f"wrote {len(cases)} golden cases: {twins} twin, {len(cases) - twins} no_twin")


if __name__ == "__main__":
    main()
