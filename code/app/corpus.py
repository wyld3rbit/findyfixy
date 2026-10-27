"""Synthetic incident corpus for the memory lessons (Day 3 onward).

What: builds 200 invented postmortems across 8 services and writes them as
JSON Lines. Every row carries `synthetic: true`. Nothing here comes from a real
system or a past employer. Hostnames, versions and numbers are made up.

Why a generator and not a model call: the file has to be reproducible. The same
seed gives byte-identical output, so a test can pin it and a later retrieval
experiment can be re-run and compared. A model would give a different corpus on
every run and cost time on a 16 GB machine.

How it is built:
- 20 scenarios (certificate rotation, pool exhaustion, clock skew, ...). Each
  has two "voices": the same incident told in different words.
- An incident is a scenario plus a set of facts (service, host, numbers, date).
- 180 originals are drawn across the scenarios, 9 per scenario.
- 20 of them get a twin: the SAME facts told in the other voice, written a few
  days later. These are the seeded near-duplicates. `twin_of` is the answer key.
  The ingest job must ignore it, or retrieval would be graded on a leaked label.

Known limits, so nobody over-reads a result:
- Templates are far more regular than real postmortems. Retrieval will look
  better here than on real data. Day 4 measures it, and the number is a ceiling.
- Nine incidents share each scenario. They rhyme, as real incidents do, but
  they are NOT twins: different facts. Only `twin_of` marks a true duplicate.
- Timeline stamps and the stated duration are drawn separately, so the last
  step can be stamped minutes after the start while the text says "after 138
  minutes". Retrieval does not read clocks. Do not use this corpus to test
  timeline reasoning.
- Some service and scenario pairs are a stretch. That is acceptable for
  retrieval practice. It would not be for a training set about root causes.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, get_args
from app.paths import DATA

CorpusService = Literal[
    "auth",
    "billing",
    "gateway",
    "voice-verification",
    "notifications",
    "search",
    "ledger",
    "scheduler",
]
Severity = Literal["sev1", "sev2", "sev3", "sev4"]

# `get_args` keeps the tuple and the Literal in one place. Adding a service in
# the Literal is the only edit needed.
SERVICES: tuple[CorpusService, ...] = get_args(CorpusService)

_FIRST_DAY = date(2025, 11, 1)
_DAY_SPAN = 350


@dataclass(frozen=True)
class Incident:
    """One postmortem as it is stored and later chunked.

    Field order is the order of keys in the JSON file. `twin_of` is the answer
    key for the seeded near-duplicates and is None for every original.
    """

    id: str
    service: CorpusService
    severity: Severity
    date: str
    summary: str
    timeline: str
    root_cause: str
    remediation: str
    synthetic: bool = True
    twin_of: str | None = None


@dataclass(frozen=True)
class Voice:
    """One way of telling an incident: the wording, with `{placeholders}`.

    Placeholders: service, host, n, mins, ver. Both voices of a scenario must
    use the same facts, or a twin would not really be the same incident.
    """

    summary: str
    steps: tuple[str, str, str]
    root_cause: str
    remediation: str


@dataclass(frozen=True)
class Scenario:
    """A failure pattern told in two voices.

    `offsets` are minutes after the start time for the three timeline steps.
    `services` limits where the pattern is plausible.
    """

    key: str
    severities: tuple[Severity, ...]
    offsets: tuple[int, int, int]
    voice_a: Voice
    voice_b: Voice
    services: tuple[CorpusService, ...] = SERVICES


@dataclass(frozen=True)
class Facts:
    """The specifics of one incident. A twin reuses them, apart from the date."""

    scenario: int
    service: CorpusService
    severity: Severity
    host: str
    n: int
    mins: int
    ver: str
    start_minute: int
    day: date

    def params(self) -> dict[str, str | int]:
        """Values for the voice templates."""
        return {
            "service": self.service,
            "host": self.host,
            "n": self.n,
            "mins": self.mins,
            "ver": self.ver,
        }


def _v(summary: str, s1: str, s2: str, s3: str, root: str, fix: str) -> Voice:
    """Short constructor so the scenario table below stays readable."""
    return Voice(summary, (s1, s2, s3), root, fix)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "cert-rotation", ("sev1", "sev2"), (0, 7, 22),
        _v(
            "p95 latency on {service} rose to {n} times normal after a certificate rotation.",
            "Alert: {service} p95 above SLO on {host}.",
            "Handshake errors seen: peers reject the new leaf certificate.",
            "Rolled back to the previous certificate; latency recovered after {mins} minutes.",
            "The rotated certificate came from an intermediate that peers did not trust. The rotation job had no pre-flight check against the peers' trust bundle.",
            "Add a pre-flight handshake test against every peer before the new certificate goes live. Distribute the intermediate to all trust bundles first.",
        ),
        _v(
            "After the scheduled certificate rotation, {service} became slow: p95 latency reached {n}x its usual level.",
            "On-call paged for slow responses from {host}.",
            "Logs showed TLS failures with peers refusing the freshly issued certificate.",
            "The old certificate was restored and the service settled within {mins} minutes.",
            "Peers were missing the intermediate authority that signed the new certificate, and nothing in the rotation tooling tested trust before cutover.",
            "Test the new certificate with real peers before cutover, and push the intermediate into trust stores ahead of any rotation.",
        ),
        services=("auth", "gateway", "voice-verification", "notifications"),
    ),
    Scenario(
        "pool-exhaustion", ("sev1", "sev2"), (0, 5, 18),
        _v(
            "{service} requests timed out; the database connection pool on {host} hit its limit of {n}.",
            "Alert: 503s from {service}, error rate above {n} percent.",
            "Pool metrics show all connections in use and requests waiting.",
            "Restarted {host} and raised the pool ceiling; errors cleared after {mins} minutes.",
            "A new query path held connections during a slow downstream call. Pool size was never sized for that hold time.",
            "Release connections before downstream calls, add a pool-wait timeout alert, and size the pool from measured hold time.",
        ),
        _v(
            "Callers of {service} got 503 errors because every database connection on {host} was taken, out of {n} allowed.",
            "Pager fired on a {n} percent error rate for {service}.",
            "Dashboards showed the connection pool full with a growing wait queue.",
            "A restart plus a bigger pool ended the errors after {mins} minutes.",
            "Connections were held open while waiting on a slow downstream service, and the pool had not been sized for that.",
            "Do not hold connections across downstream calls. Alert on pool wait time. Choose pool size from measured hold time.",
        ),
        services=("billing", "ledger", "auth", "scheduler"),
    ),
    Scenario(
        "clock-skew", ("sev2",), (0, 9, 25),
        _v(
            "Token validation failures on {service}: the clock on {host} drifted {n} seconds.",
            "Alert: rising 401s on {service}.",
            "{host} clock found {n} seconds ahead of peers.",
            "Resynced time and drained {host}; 401s stopped after {mins} minutes.",
            "The time daemon on {host} lost its upstream source and the node free-ran. Token validity windows are shorter than the drift.",
            "Alert on clock offset above 1 second. Run at least three time sources. Evict nodes with bad offset automatically.",
        ),
        _v(
            "Users saw login rejections from {service} because the clock on {host} was {n} seconds off.",
            "A spike of 401 responses from {service} triggered the page.",
            "Checking nodes showed {host} running {n} seconds fast.",
            "Time was corrected and the node was drained; the rejections ended within {mins} minutes.",
            "NTP lost its upstream on {host}, so the machine drifted past the token validity window.",
            "Monitor clock offset and page above 1 second. Configure three or more time sources. Auto-remove drifting nodes.",
        ),
        services=("auth", "gateway", "ledger", "scheduler"),
    ),
    Scenario(
        "cache-stampede", ("sev2", "sev3"), (0, 4, 16),
        _v(
            "Cache keys on {service} expired together and {n} percent of requests fell through to the database.",
            "Alert: database load behind {service} at its limit.",
            "Hit rate dropped as one TTL batch expired at once.",
            "Warmed the cache and added jitter; load normal after {mins} minutes.",
            "All keys were written with the same TTL during a bulk refresh, so they expired in the same second.",
            "Add random jitter to TTLs, coalesce concurrent misses for the same key, and warm the cache before bulk expiry.",
        ),
        _v(
            "A synchronized cache expiry sent {n} percent of {service} traffic straight to the database.",
            "Database saturation alarm for the {service} backend.",
            "Cache hit rate collapsed after a batch of keys expired together.",
            "The cache was refilled by hand and TTL jitter shipped; things normalised in {mins} minutes.",
            "A bulk refresh gave every key an identical TTL, so they all expired at the same moment.",
            "Jitter every TTL, collapse parallel misses into one fetch, and prewarm before a known expiry.",
        ),
        services=("gateway", "search", "auth", "notifications"),
    ),
    Scenario(
        "bad-config-push", ("sev1", "sev2"), (0, 6, 14),
        _v(
            "Config version {ver} set a wrong limit on {service}; {n} percent of traffic was rejected.",
            "Alert: rejects on {service} after config push.",
            "Diff of config {ver} showed the limit set to 10 instead of 1000.",
            "Reverted config; traffic recovered after {mins} minutes.",
            "A typo passed review because the config had no schema bounds. The push went to all hosts at once.",
            "Validate config against bounds in CI and roll config out in stages, with automatic rollback on error rate.",
        ),
        _v(
            "Config release {ver} contained a bad limit, so {service} turned away {n} percent of requests.",
            "Reject rate on {service} jumped right after the config push.",
            "Comparing {ver} with the previous config showed 10 where 1000 was intended.",
            "We reverted and the service recovered in {mins} minutes.",
            "A one-digit typo went through review since nothing enforced allowed ranges, and the change hit every host together.",
            "Enforce value ranges in CI and ship config in stages, rolling back on error-rate alarms.",
        ),
    ),
    Scenario(
        "disk-full", ("sev2",), (0, 8, 20),
        _v(
            "{host} ran out of disk and {service} writes failed for {mins} minutes.",
            "Alert: write errors on {host}.",
            "Disk at 100 percent; debug logs filled the volume.",
            "Deleted old logs and rotated; writes resumed. Disk had held {n} gigabytes of logs.",
            "Debug logging was left on after an earlier investigation. Log rotation had no size cap.",
            "Cap log size and retention, alert at 80 percent disk, and expire temporary debug flags automatically.",
        ),
        _v(
            "Writes to {service} failed for {mins} minutes when the disk on {host} filled up.",
            "On-call got write-error alerts for {host}.",
            "The volume was at 100 percent, mostly debug logs.",
            "Old logs were cleared and rotation fixed; writes returned. The logs had grown to {n} gigabytes.",
            "A debug logging flag from a past investigation was never switched off, and rotation had no size limit.",
            "Set log size and retention limits, alert at 80 percent usage, and make debug flags expire on their own.",
        ),
        services=("ledger", "notifications", "search", "scheduler"),
    ),
    Scenario(
        "dns-stale", ("sev2", "sev3"), (0, 11, 30),
        _v(
            "{service} kept sending traffic to a retired backend for {mins} minutes after a DNS change; {n} percent of calls failed.",
            "Alert: errors from {service} to one upstream.",
            "Resolvers on {host} still returned the old address.",
            "Flushed the cache on {host}; errors stopped.",
            "The record's TTL was 24 hours and the client library cached lookups beyond the TTL.",
            "Lower the TTL before planned DNS moves and configure clients to honor TTL. Drain old backends before retiring them.",
        ),
        _v(
            "After a DNS update, {service} continued to hit a decommissioned backend for {mins} minutes; {n} percent of calls failed.",
            "Errors toward a single upstream were flagged for {service}.",
            "{host} was still resolving the old address.",
            "A cache flush on {host} cleared the problem.",
            "A 24 hour TTL combined with a client library that cached lookups past the TTL.",
            "Reduce TTL ahead of DNS moves, make clients honor it, and drain a backend before it is retired.",
        ),
        services=("gateway", "auth", "voice-verification", "billing"),
    ),
    Scenario(
        "retry-storm", ("sev2",), (0, 3, 15),
        _v(
            "Client retries on {service} multiplied load {n} times after a vendor began returning 429.",
            "Alert: outbound request rate from {service} {n}x normal.",
            "Vendor returned 429; every client retried immediately.",
            "Enabled backoff with jitter; load fell after {mins} minutes.",
            "Retry logic had no backoff and no retry budget, so each failure produced more load.",
            "Use exponential backoff with jitter, cap retries per request and per client, and honor Retry-After.",
        ),
        _v(
            "When a vendor started answering 429, {service} retried so aggressively that load grew {n} fold.",
            "Outbound traffic from {service} hit {n} times its norm.",
            "Every caller retried at once on each 429.",
            "Backoff with jitter was turned on and load dropped within {mins} minutes.",
            "Retries happened immediately with no budget, so failures created more traffic.",
            "Exponential backoff with jitter, a retry cap per request and per client, and respect for Retry-After.",
        ),
        services=("billing", "notifications", "voice-verification", "ledger"),
    ),
    Scenario(
        "memory-leak", ("sev3", "sev2"), (0, 40, 95),
        _v(
            "{host} was OOM-killed repeatedly; memory on {service} grew {n} MB per hour since version {ver}.",
            "Alert: restarts of {host} every few hours.",
            "Heap growth traced to an unbounded in-memory map.",
            "Rolled back to the previous version; restarts stopped after {mins} minutes.",
            "Version {ver} added a per-request cache with no eviction.",
            "Bound every cache with a size limit and eviction policy, and add a memory-growth alert on canary hosts.",
        ),
        _v(
            "Since version {ver}, memory on {service} climbed {n} MB each hour and {host} kept being killed for running out.",
            "Pager noted {host} restarting every few hours.",
            "Heap analysis pointed at a map that never shrank.",
            "The previous version was redeployed and the restarts ended within {mins} minutes.",
            "A cache added in {ver} was keyed per request and had no eviction.",
            "Give each cache a size bound and eviction, and alarm on memory growth in canary hosts.",
        ),
        services=("search", "gateway", "notifications", "scheduler"),
    ),
    Scenario(
        "migration-lock", ("sev1",), (0, 3, 12),
        _v(
            "A schema migration locked a table on {service} for {mins} minutes.",
            "Alert: queries on {service} blocked.",
            "Migration held an exclusive lock while rewriting {n} million rows.",
            "Cancelled the migration; queries resumed.",
            "The migration added a column with a default and rewrote the whole table under an exclusive lock.",
            "Use online migration patterns: add the column nullable, backfill in batches, then set the default. Test migrations on production-sized data.",
        ),
        _v(
            "For {mins} minutes, {service} could not run queries because a migration was locking a table.",
            "Blocked queries on {service} triggered the page.",
            "The migration was rewriting {n} million rows with an exclusive lock held.",
            "It was cancelled and queries flowed again.",
            "Adding a column with a default rewrote the entire table while holding an exclusive lock.",
            "Migrate online: nullable column first, batched backfill, default last. Rehearse on production-sized data.",
        ),
        services=("ledger", "billing", "auth", "scheduler"),
    ),
    Scenario(
        "queue-backlog", ("sev3", "sev2"), (0, 12, 35),
        _v(
            "Queue depth on {service} reached {n} thousand messages because consumers stalled.",
            "Alert: oldest message on {service} queue is {n} minutes old.",
            "Consumers on {host} blocked on a poison message.",
            "Moved the poison message to a dead-letter queue; backlog drained in {mins} minutes.",
            "One malformed message crashed the consumer on every retry, and there was no dead-letter route.",
            "Add a dead-letter queue with a retry limit, and alert on oldest message age, not just depth.",
        ),
        _v(
            "Consumers of {service} stopped making progress and the queue grew to {n} thousand messages.",
            "Oldest-message age on {service} passed {n} minutes.",
            "The consumer on {host} kept failing on one bad message.",
            "That message went to a dead-letter queue and the backlog cleared in {mins} minutes.",
            "A malformed message crashed the consumer on each retry. No dead-letter path existed.",
            "Create a dead-letter queue with a retry cap and alert on oldest message age as well as depth.",
        ),
        services=("notifications", "scheduler", "billing", "search"),
    ),
    Scenario(
        "rate-limit-reload", ("sev2",), (0, 5, 13),
        _v(
            "A rate-limit reload on {service} applied a limit of {n} requests per second, far below demand.",
            "Alert: 429 responses from {service}.",
            "Reload on {host} applied the default table instead of the tuned one.",
            "Reloaded the tuned limits; 429s ended after {mins} minutes.",
            "The reload read a config path that was empty during deployment and silently fell back to defaults.",
            "Fail the reload when the config is empty, and alert when active limits differ from the intended values.",
        ),
        _v(
            "{service} began rejecting traffic with 429s after a rate-limit reload set the cap at {n} requests per second.",
            "A surge of 429 responses from {service} paged the team.",
            "{host} had loaded default limits in place of the tuned set.",
            "Re-applying the tuned limits stopped the rejections within {mins} minutes.",
            "An empty config path during deployment made the reload quietly fall back to defaults.",
            "Make an empty config a reload failure, and alert when live limits differ from intended ones.",
        ),
        services=("gateway", "auth", "search", "billing"),
    ),
    Scenario(
        "vendor-outage", ("sev2", "sev1"), (0, 6, 24),
        _v(
            "A third-party provider outage degraded {service} for {mins} minutes; {n} percent of calls failed.",
            "Alert: provider error rate at {n} percent for {service}.",
            "Provider status page confirmed an incident.",
            "Switched {service} to the secondary provider; errors fell.",
            "{service} depended on a single provider with no automatic failover.",
            "Add a secondary provider with automatic failover, and circuit-break after a set error rate.",
        ),
        _v(
            "{n} percent of {service} calls failed for {mins} minutes during an outage at an outside provider.",
            "Provider errors reached {n} percent on {service}.",
            "The provider's status page acknowledged the outage.",
            "Traffic was moved by hand to the backup provider and errors dropped.",
            "{service} relied on one provider and had no automatic failover.",
            "Set up a backup provider with automatic failover and a circuit breaker at a defined error rate.",
        ),
        services=("voice-verification", "notifications", "billing", "auth"),
    ),
    Scenario(
        "autoscaler-floor", ("sev3", "sev2"), (0, 10, 28),
        _v(
            "{service} saturated at the start of the daily peak; the autoscaler floor was too low.",
            "Alert: CPU on {service} above {n} percent.",
            "Autoscaler scaled up one step at a time and lagged the ramp.",
            "Raised the minimum replicas by hand; latency recovered after {mins} minutes.",
            "The minimum replica count was set for last year's traffic and scale-up was slower than the morning ramp.",
            "Review minimum replicas against current peak each quarter, and pre-scale ahead of the known daily ramp.",
        ),
        _v(
            "At the daily peak, {service} ran out of capacity because the autoscaler's minimum was too small.",
            "CPU on {service} passed {n} percent.",
            "The autoscaler added capacity step by step but fell behind the traffic ramp.",
            "Manually raising the replica floor restored latency in {mins} minutes.",
            "The replica floor reflected old traffic levels, and scaling was slower than the morning ramp.",
            "Check the replica floor against current peaks every quarter and scale ahead of the known morning ramp.",
        ),
        services=("search", "gateway", "auth", "notifications"),
    ),
    Scenario(
        "leader-flap", ("sev2",), (0, 8, 21),
        _v(
            "Leader election on {service} flapped {n} times in {mins} minutes.",
            "Alert: repeated leader changes on {service}.",
            "Lease renewals on {host} missed their deadline under GC pauses.",
            "Raised the lease timeout and fixed the heap size; leadership stable.",
            "The lease timeout was shorter than worst-case garbage collection pauses on {host}.",
            "Set lease timeouts above worst-case pause times, and alert when leadership changes more than twice an hour.",
        ),
        _v(
            "The leader of {service} changed {n} times within {mins} minutes.",
            "Many leadership changes were reported for {service}.",
            "{host} was missing lease renewals during long garbage collection pauses.",
            "A longer lease timeout and a heap fix settled things.",
            "Lease timeout was shorter than the longest garbage collection pause seen on {host}.",
            "Make lease timeouts longer than worst-case pauses and alert on more than two leader changes per hour.",
        ),
        services=("scheduler", "ledger", "auth", "billing"),
    ),
    Scenario(
        "hot-shard", ("sev2", "sev3"), (0, 7, 26),
        _v(
            "One shard of {service} took {n} percent of traffic and timed out.",
            "Alert: latency on one shard of {service} at {host}.",
            "A single customer key accounted for most requests.",
            "Split the hot key across shards; latency recovered after {mins} minutes.",
            "Sharding used the customer id alone, so one very large customer concentrated load on one shard.",
            "Add a salt to hot keys, monitor per-shard load, and cap per-key rate.",
        ),
        _v(
            "A single shard of {service} received {n} percent of the load and began timing out.",
            "Latency alarm on one shard of {service}, served from {host}.",
            "One customer's key produced most of the traffic.",
            "The hot key was spread over several shards and latency recovered within {mins} minutes.",
            "Shard choice depended on the customer id only, so one huge customer overloaded a shard.",
            "Salt hot keys, watch per-shard load, and limit request rate per key.",
        ),
        services=("ledger", "search", "auth", "notifications"),
    ),
    Scenario(
        "expired-secret", ("sev1", "sev2"), (0, 2, 17),
        _v(
            "An expired API credential caused {service} to fail every call to a dependency for {mins} minutes; the credential was {n} days past its renewal date.",
            "Alert: 401s from dependency on {service}.",
            "The credential used by {host} expired at midnight UTC.",
            "Issued and deployed a new credential; calls succeeded.",
            "The credential had a 90 day life and nobody owned the rotation reminder.",
            "Automate credential rotation, alert 14 days before expiry, and name an owner for every credential.",
        ),
        _v(
            "For {mins} minutes {service} could not call a dependency because an API credential had expired; renewal was {n} days overdue.",
            "Dependency calls from {service} returned 401.",
            "{host} was using a credential that lapsed at midnight UTC.",
            "A fresh credential was deployed and calls recovered.",
            "A 90 day credential had no owner for its renewal.",
            "Rotate credentials automatically, alert two weeks before expiry, and assign an owner to each credential.",
        ),
        services=("auth", "billing", "notifications", "gateway"),
    ),
    Scenario(
        "deploy-regression", ("sev2", "sev3"), (0, 9, 27),
        _v(
            "Version {ver} of {service} introduced a regression: p95 rose {n} percent.",
            "Alert: latency regression on {service} after deploy of {ver}.",
            "Profiling showed an N+1 query on the new endpoint.",
            "Rolled back {ver}; latency normal after {mins} minutes.",
            "A new code path issued one query per item. Load tests did not cover that endpoint.",
            "Batch the queries, add the endpoint to load tests, and gate deploys on a latency check in canary.",
        ),
        _v(
            "Release {ver} slowed {service}; p95 latency went up {n} percent.",
            "Latency alarm fired on {service} following the {ver} deploy.",
            "Profiles pointed to one query per item on the new endpoint.",
            "{ver} was rolled back and latency returned in {mins} minutes.",
            "The new code issued a query for every item, and no load test exercised the endpoint.",
            "Batch the queries, include the endpoint in load tests, and make canary latency a deploy gate.",
        ),
    ),
    Scenario(
        "tls-version", ("sev2",), (0, 6, 19),
        _v(
            "A TLS library upgrade on {service} dropped support for a protocol version used by {n} percent of clients.",
            "Alert: handshake failures on {service}.",
            "Failures came from older clients on TLS 1.0 after the upgrade.",
            "Re-enabled the legacy setting; handshakes succeeded after {mins} minutes.",
            "The upgrade changed the default minimum protocol version and no client census existed.",
            "Track client protocol versions before changing defaults, and announce a deprecation window.",
        ),
        _v(
            "After a TLS library upgrade, {n} percent of clients could no longer connect to {service}.",
            "Handshake failure alarm for {service}.",
            "Older clients speaking TLS 1.0 were being refused post-upgrade.",
            "The legacy setting was switched back on and connections resumed in {mins} minutes.",
            "The new library default raised the minimum protocol version and nobody knew which clients used old versions.",
            "Measure client protocol versions before changing defaults and give a published deprecation window.",
        ),
        services=("gateway", "voice-verification", "notifications", "billing"),
    ),
    Scenario(
        "noisy-neighbor", ("sev3",), (0, 14, 33),
        _v(
            "A batch job on {host} starved {service} of CPU for {mins} minutes.",
            "Alert: latency on {service} with low request volume.",
            "A co-located batch job used {n} percent of CPU on {host}.",
            "Moved the batch job to its own pool; latency recovered.",
            "Batch and serving workloads shared nodes with no CPU limits on the batch job.",
            "Set CPU limits and priority classes, and keep batch work on a separate node pool.",
        ),
        _v(
            "{service} ran slowly for {mins} minutes because a batch job on {host} consumed the CPU.",
            "Latency rose on {service} even though traffic was low.",
            "A batch job on the same node was using {n} percent of CPU.",
            "The batch job was relocated to a separate pool and latency recovered.",
            "Batch and serving pods shared a node and the batch job had no CPU limit.",
            "Apply CPU limits and priority classes, and run batch jobs on a separate node pool.",
        ),
        services=("search", "scheduler", "ledger", "notifications"),
    ),
)


@dataclass(frozen=True)
class _Draft:
    """An incident before ids exist. Ids come after sorting by date."""

    order: int
    facts: Facts
    voice_a: bool
    twin_of: int | None = None
    sort_key: tuple[date, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_key", (self.facts.day, self.order))


def clock(minute: int) -> str:
    """Format minutes since midnight as HH:MM, wrapping past 24 hours."""
    hours, mins = divmod(minute % 1440, 60)
    return f"{hours:02d}:{mins:02d}"


def draw_facts(rng: random.Random, scenario_index: int) -> Facts:
    """Draw the specifics of one incident for a scenario.

    All randomness comes from `rng`, so a seeded generator gives the same facts.
    """
    scenario = SCENARIOS[scenario_index]
    service = rng.choice(scenario.services)
    return Facts(
        scenario=scenario_index,
        service=service,
        severity=rng.choice(scenario.severities),
        host=f"{service}-{rng.randrange(16**3):03x}",
        n=rng.randint(3, 95),
        mins=rng.randint(8, 180),
        ver=f"{rng.randint(1, 6)}.{rng.randint(0, 19)}.{rng.randint(0, 9)}",
        start_minute=rng.randrange(1440),
        day=_FIRST_DAY + timedelta(days=rng.randrange(_DAY_SPAN)),
    )


def render(facts: Facts, *, voice_a: bool) -> tuple[str, str, str, str]:
    """Tell one incident in one voice.

    Returns summary, timeline, root cause and remediation. `str.format_map`
    raises KeyError on an unknown placeholder, so a typo in a template fails the
    build instead of shipping a literal `{hots}` into the corpus.
    """
    scenario = SCENARIOS[facts.scenario]
    voice = scenario.voice_a if voice_a else scenario.voice_b
    params = facts.params()
    timeline = "\n".join(
        f"{clock(facts.start_minute + offset)} {step.format_map(params)}"
        for offset, step in zip(scenario.offsets, voice.steps, strict=True)
    )
    return (
        voice.summary.format_map(params),
        timeline,
        voice.root_cause.format_map(params),
        voice.remediation.format_map(params),
    )


def build_corpus(seed: int = 7, *, total: int = 200, pairs: int = 20) -> list[Incident]:
    """Build the corpus: `total - pairs` originals plus `pairs` twins.

    Pure and deterministic: the same arguments give the same list. The scenarios
    are dealt out evenly and then shuffled, so every scenario appears and no
    scenario dominates. Twins reuse the facts of their original, tell them in
    the other voice, and are dated one to four days later (a postmortem written
    up after the fact). Ids follow date order, so a twin is not recognisable by
    its position in the file.
    """
    originals = total - pairs
    if pairs < 0 or originals < 1 or pairs > originals:
        raise ValueError(f"cannot make {pairs} twins from {originals} originals")

    rng = random.Random(seed)
    deal = [i % len(SCENARIOS) for i in range(originals)]
    rng.shuffle(deal)

    drafts = [
        _Draft(order, draw_facts(rng, scenario), voice_a=rng.random() < 0.5)
        for order, scenario in enumerate(deal)
    ]
    for offset, source in enumerate(rng.sample(drafts, pairs)):
        later = source.facts.day + timedelta(days=rng.randint(1, 4))
        drafts.append(
            _Draft(
                order=originals + offset,
                # Only the date moves. Same service, host and numbers: it is the
                # same incident, retold.
                facts=replace(source.facts, day=later),
                voice_a=not source.voice_a,
                twin_of=source.order,
            )
        )

    ordered = sorted(drafts, key=lambda d: d.sort_key)
    ids = {d.order: f"INC-{position:04d}" for position, d in enumerate(ordered, 1)}
    incidents = []
    for draft in ordered:
        summary, timeline, root_cause, remediation = render(
            draft.facts, voice_a=draft.voice_a
        )
        incidents.append(
            Incident(
                id=ids[draft.order],
                service=draft.facts.service,
                severity=draft.facts.severity,
                date=draft.facts.day.isoformat(),
                summary=summary,
                timeline=timeline,
                root_cause=root_cause,
                remediation=remediation,
                twin_of=None if draft.twin_of is None else ids[draft.twin_of],
            )
        )
    return incidents


def write_jsonl(incidents: list[Incident], path: Path) -> int:
    """Write one JSON object per line and return the row count.

    JSON Lines lets the ingest job stream the file and lets `git diff` show
    one changed incident as one changed line. Plain ASCII keeps it portable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for incident in incidents:
            handle.write(json.dumps(asdict(incident)) + "\n")
    return len(incidents)


def main() -> None:
    """Write `data/incidents.jsonl` under the project root.

    The path is resolved from this file, not from the current directory, so it
    works from any clone and any working directory.
    """
    count = write_jsonl(build_corpus(), DATA / "incidents.jsonl")
    print(f"wrote {count} synthetic incidents")


if __name__ == "__main__":
    main()
