"""Day 4, step 1: the labelled queries that measure retrieval.

What it does
    Builds `data/retrieval_eval.jsonl`: 40 queries, each paired with the
    incident IDs that a good retriever must return.

Why retrieval needs its own labelled set
    If the right incident never reaches the prompt, no prompt recovers it.
    Without queries that have known answers, "retrieval quality" is opinion.
    Recall@5 and MRR (step 3) are computed against this file.

Two kinds of query, because the two retrievers fail differently
    meaning (30): a plain-words paraphrase of a seeded pair. The correct
        answer is both twins, because they are the same incident told in two
        voices. Rare tokens are avoided on purpose, so keyword search has
        little to hold on to and vector search should win.
    token (10): a hostname such as `scheduler-3de` plus a few words. A
        dense model smooths a string like that into its neighbours. BM25
        matches it exactly. The correct answer is every incident whose text
        contains the token, computed from the data, never typed by hand.

How labels stay honest
    Meaning labels come from `twin_of` in the corpus, not from typed IDs. A
    query is written against one anchor incident and its partner is looked
    up. The gateway query has two anchors, because the corpus holds two
    identical gateway pairs (same pattern, no number to tell them apart). A
    query that labelled only one pair would punish a correct hit on the other.

Limits
    The data is templated, so absolute scores will look flattering. Only the
    gap between retrievers means anything. The 30 meaning queries are written
    by hand (by the assistant), so they reflect one writer's phrasing.

Separation
    This file reads `twin_of`. Retrieval code must never import it. A
    retriever that could read its own answer key would make recall meaningless.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from app.paths import DATA

INCIDENTS_PATH = DATA / "incidents.jsonl"
EVAL_PATH = DATA / "retrieval_eval.jsonl"

# Text fields a hostname can appear in. The same fields the chunks are made of.
TEXT_FIELDS = ("summary", "timeline", "root_cause", "remediation")

# A hostname is a service-like name, a dash, then three hex characters with at
# least one digit: `scheduler-3de`, `billing-750`, `voice-verification-658`.
# The digit rule keeps ordinary words such as `re-add` from matching.
HOSTNAME = re.compile(r"\b[a-z]+(?:-[a-z]+)*-(?=[0-9a-f]*\d)[0-9a-f]{3}\b")

TOKEN_QUERY_COUNT = 10
# A token found in many incidents gives a large answer set, so recall@5 would
# be easy. One to three keeps the exact-match test sharp.
MAX_TOKEN_INCIDENTS = 3

Kind = Literal["meaning", "token"]


@dataclass(frozen=True)
class EvalQuery:
    """One labelled query. `relevant` is the set of IDs that count as correct."""

    id: str
    kind: Kind
    query: str
    relevant: tuple[str, ...]


# Each entry is (anchor incident IDs, query). The anchor's partner is added from
# the corpus. First 19: the summary's facts in other words. Last 11: the same
# pairs again from the cause, in the language of the root cause. A query never
# contains a hostname, so only meaning can find the answer.
MEANING: tuple[tuple[tuple[str, ...], str], ...] = (
    (("INC-0009",), "A bad limit shipped in voice-verification configuration version 4.14.6 and about a third of requests were refused."),
    (("INC-0021", "INC-0126"), "Gateway ran out of capacity at the daily traffic peak because the minimum number of replicas was too low."),
    (("INC-0027",), "Most notifications clients could not connect after the TLS library was upgraded and dropped an older protocol version."),
    (("INC-0028",), "Billing queries hung for fourteen minutes while a database migration held a lock on a table."),
    (("INC-0057",), "The billing queue backed up to about sixty-seven thousand messages because consumers stalled."),
    (("INC-0066",), "Logins were rejected on scheduler because one machine's clock was forty-five seconds off, past the token validity window."),
    (("INC-0073",), "Auth became slow after a scheduled certificate rotation, with latency at forty-one times its usual level."),
    (("INC-0075",), "One shard of auth took four fifths of the traffic and began timing out."),
    (("INC-0094",), "After a vendor began answering with 429, voice-verification clients retried until load grew forty-five fold."),
    (("INC-0096",), "Billing used up all sixteen of its database connections and callers received errors."),
    (("INC-0105",), "After a DNS change, voice-verification kept sending traffic to a retired backend for 161 minutes."),
    (("INC-0123",), "Writes to scheduler failed for 157 minutes after a disk filled up."),
    (("INC-0131",), "Notifications could not call a dependency for 85 minutes because an API credential had expired a month earlier."),
    (("INC-0145",), "A bad limit in scheduler configuration version 3.1.1 made it reject about a third of requests."),
    (("INC-0155",), "On search, cache entries expired together and nearly half of requests fell through to the database."),
    (("INC-0163",), "Scheduler callers got 503 errors because all sixty-seven database connections were in use."),
    (("INC-0178",), "Token validation failed on ledger because a server clock had drifted nineteen seconds."),
    (("INC-0179",), "Memory on scheduler grew thirteen megabytes an hour from version 2.4.0 until processes were killed."),
    (("INC-0191",), "After a rate-limit reload, auth refused traffic with 429 because the cap was nine requests per second."),
    (("INC-0073",), "Peers did not have the intermediate authority that signed the new certificate and trust was never tested before cutover, on auth."),
    (("INC-0094",), "No backoff and no retry budget meant every failure made more load, voice-verification calling a vendor."),
    (("INC-0057",), "A malformed message crashed the consumer on each retry and there was no dead-letter path, on billing."),
    (("INC-0028",), "Adding a column with a default rewrote the whole table under an exclusive lock, on billing."),
    (("INC-0155",), "A bulk refresh gave every key the same time to live, so they all expired at once, on search."),
    (("INC-0179",), "A cache added in 2.4.0 was keyed per request and never evicted anything, on scheduler."),
    (("INC-0191",), "An empty config path at deploy made the reload quietly fall back to default limits, on auth."),
    (("INC-0131",), "The credential lived ninety days and nobody owned the rotation reminder, on notifications."),
    (("INC-0075",), "Sharding by customer id alone let one huge customer concentrate load on a single shard, on auth."),
    (("INC-0027",), "The new library default raised the minimum protocol version and nobody knew which clients were old, on notifications."),
    (("INC-0105",), "A one day TTL plus a client library that cached lookups past the TTL, on voice-verification."),
)


def read_rows(path: Path = INCIDENTS_PATH) -> list[dict]:
    """Read the corpus as plain dicts. Not at import: a bad file should fail loudly at use."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def partners(rows: Sequence[Mapping]) -> dict[str, str]:
    """Map each seeded incident to its twin, in both directions."""
    pairs: dict[str, str] = {}
    for row in rows:
        if row.get("twin_of"):
            pairs[row["id"]] = row["twin_of"]
            pairs[row["twin_of"]] = row["id"]
    return pairs


def hostnames_by_incident(rows: Sequence[Mapping]) -> dict[str, set[str]]:
    """Find, for each hostname, the incidents whose text contains it."""
    found: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for field in TEXT_FIELDS:
            for name in HOSTNAME.findall(row[field]):
                found[name].add(row["id"])
    return found


def meaning_queries(rows: Sequence[Mapping]) -> list[EvalQuery]:
    """The 30 paraphrase queries. Each answer is the anchors and their twins."""
    twin = partners(rows)
    known = {row["id"] for row in rows}
    queries = []
    for index, (anchors, text) in enumerate(MEANING, start=1):
        if any(anchor not in twin for anchor in anchors):
            raise ValueError(f"meaning query {index}: an anchor is not a seeded incident")
        wanted = set(anchors) | {twin[anchor] for anchor in anchors}
        if not wanted <= known:
            raise ValueError(f"meaning query {index}: a label is not in the corpus")
        queries.append(EvalQuery(f"m{index:02d}", "meaning", text, tuple(sorted(wanted))))
    return queries


def token_queries(rows: Sequence[Mapping], count: int = TOKEN_QUERY_COUNT) -> list[EvalQuery]:
    """The exact-token queries. Hostnames are picked at an even stride, so the
    choice is fixed and spread over the sorted list, not hand-picked."""
    usable = sorted(
        (name, ids) for name, ids in hostnames_by_incident(rows).items() if len(ids) <= MAX_TOKEN_INCIDENTS
    )
    if len(usable) < count:
        raise ValueError(f"only {len(usable)} usable hostnames, need {count}")
    stride = len(usable) // count
    chosen = usable[::stride][:count]
    return [
        EvalQuery(f"t{index:02d}", "token", f"errors and alerts on {name}", tuple(sorted(ids)))
        for index, (name, ids) in enumerate(chosen, start=1)
    ]


def build_queries(rows: Sequence[Mapping]) -> list[EvalQuery]:
    """All 40 queries: meaning first, then exact-token."""
    return meaning_queries(rows) + token_queries(rows)


def write_queries(queries: Sequence[EvalQuery], path: Path = EVAL_PATH) -> None:
    """Write one JSON object per line. `relevant` becomes a list."""
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(asdict(query)) + "\n")


def load_queries(path: Path = EVAL_PATH) -> list[EvalQuery]:
    """Read the file back. Step 3 uses this."""
    with path.open(encoding="utf-8") as handle:
        return [
            EvalQuery(r["id"], r["kind"], r["query"], tuple(r["relevant"]))
            for r in map(json.loads, (line for line in handle if line.strip()))
        ]


def main() -> None:
    """Rebuild the file and print the counts the self-check expects."""
    queries = build_queries(read_rows())
    write_queries(queries)
    tokens = sum(q.kind == "token" for q in queries)
    print(f"{len(queries)} queries: {tokens} exact-token, {len(queries) - tokens} meaning")


if __name__ == "__main__":
    main()
