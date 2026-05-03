# Sploink — Systems Architecture for Execution-Graph Ingestion, Storage, Scoring, and Replay

## 0. What To Build

Sploink ingests every meaningful action from concurrent autonomous-agent sessions — file I/O, shell commands, tool invocations, LLM calls, retries, branches, validations, outputs, and the causal dependencies between them. The events stream in concurrently from many tenants, often **out of order, duplicated, delayed by minutes (sometimes weeks), partially missing, sometimes from sessions whose host machine crashed**, and frequently in **bursts of hundreds of events per second**. We need to:

1. Ingest these events losslessly.
2. Reconstruct each session's **execution graph** — not just an event log, but a typed graph over (action, target) nodes with causal edges between them.
3. Score sessions in **real time** against the four behavioural classes the classifier in `solution.py` already detects (progressing / looping / drifting / failing), with the explicit roadmap that scoring extends to intervention.
4. Support **replay, rewind, and fork** primitives that let us deterministically re-execute a session from any checkpoint.
5. Scale through three operating points — 1M, 10M, 100M events/day — with documented bottlenecks at each.

This document describes the architecture we settled on and the tradeoffs we made deliberately. It is opinionated; we'll call out where reasonable engineers would have picked differently.

---

## 1. High-Level Architecture

```
[CLI tool on user's machine]
   - emits events via SDK hooks
  - holds long-lived outbound control stream
   - in-memory buffer, periodic disk flush
   - local append-only log on disk (eventually-consistent)
  - background pusher with retry + idempotency keying
  - local command executor gated by policy
        │
      │  bidirectional WebSocket (events up, ACKs/commands down)
        ▼
[Gateway / Control Plane] — thin Go service
   - mTLS auth
   - schema validation
   - dedup (Layer 2) via NATS KV, returns ACK or DUPLICATE_ACK
  - fans out queued commands over the existing client stream
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│  NATS JetStream — durable event log (source of truth)          │
│  subject: sploink.events.{tenant}.{session_id}                 │
│  retention: 14d hot in JetStream, 90d mirror to S3, ∞ in cold  │
└─────────────┬─────────────────────────────────┬────────────────┘
              │                                 │
              │ consumed by                     │ consumed by
              ▼                                 ▼
┌────────────────────────────────┐   ┌──────────────────────────────────┐
│  Materialization Service       │   │  Real-Time Scoring Service       │
│  Go worker pool, consistent-   │   │  Go worker pool, same hash ring  │
│  hash sharded by session_id    │   │  - streaming aggregates per sess │
│  - writes nodes to events tbl  │   │  - 200-event sliding window      │
│  - computes edges, writes too  │   │  - triggers: 5-event / 10s / mrk │
│  - writes graph snapshots      │   │  - emits scores to NATS          │
│    every 25 events to S3       │   │  - state in BadgerDB on local SSD│
└──┬──────────────┬───────────┬──┘   └─────────────────┬────────────────┘
   ▼              ▼           ▼                        ▼
┌──────────┐  ┌────────────┐  ┌──────────────┐   [NATS scores subject]
│   HOT    │  │   WARM     │  │   COLD       │            │
│ Postgres │  │ ClickHouse │  │  S3 +        │            ├─► dashboards
│ + AGE    │  │            │  │  Iceberg     │            ├─► alerting
│ active7d │  │  8d–90d    │  │  Parquet     │            ├─► intervention
│ <50ms    │  │  1–5s p99  │  │  90d–∞       │            │   (Part C)
└──────────┘  └────────────┘  └──────────────┘            └─► storage tiers

[Replay Service] (Go, gRPC) — replay / rewind / fork primitives, reads from NATS for hot data, ClickHouse for warm, Parquet for cold
[Control Service] (Go + NATS) — persists operator/model-issued commands, delivers them over the CLI-initiated WebSocket; no inbound reachability to the user machine required
[Batch Reconciler] (Go, cron) — every 5min, runs exact featurize() over a 5% sample of active sessions, pages on persistent disagreement
```

The shape is intentional. The durable event log is the single source of truth. Every downstream view — graph DB, columnar store, cold archive, score stream, snapshots — is a deterministic function of that log. Replay, rewind, and fork are essentially "run the same logic against an earlier offset," which is what makes them safe to operate.

### 1.1 Tech Stack Summary

This is a Go-first system, not a generic "big data" template. The concrete stack is:

- **CLI runtime** — Go binary on the user's machine, with a local append-only disk log and a local executor for backend-issued control intents.
- **Transport** — bidirectional WebSocket over TLS for CLI <-> backend; gRPC remains a later option and is still fine for some backend APIs.
- **Event backbone** — NATS JetStream for durable streams and NATS KV for the short-window dedup cache.
- **Online services** — Go services for the gateway/control plane, materializer, scorer, replay service, and batch reconciler.
- **Worker-local state** — BadgerDB on local NVMe for per-session keyed state and fast crash recovery.
- **Hot operational store** — Postgres + Apache AGE for active-session graph queries and low-latency intervention lookups.
- **Warm analytical store** — ClickHouse for 8d-90d scans, dashboards, and retraining data extraction.
- **Cold archive** — S3 + Iceberg-managed Parquet for indefinite retention, audit, and long-tail replay.

The important non-choice is just as important as the chosen stack: the backend does **not** need a hostable IP address for the CLI, does **not** SSH into customer machines, and does **not** require inbound firewall openings. Control traffic returns over the same outbound stream the CLI already opened.

---

## 2. Event Ingestion Path

### 2.1 The Emitter

The thing emitting events is **Sploink's own CLI tool**, in the spirit of Claude Code — a binary we ship and control, running on the user's machine. The user installs it; it runs locally; it orchestrates one or more agents (potentially a swarm of sub-agents under one parent session). This matters architecturally because the CLI is **trusted code**, not an arbitrary customer integration. We can put real logic in it — local buffering, idempotency keying, retry — without worrying about the user implementing it badly.

A "session" in our model corresponds to what the user thinks of as their long-lived chat or work session. Within a session, the parent agent may spawn sub-agents that interact with each other in a swarm; all of their events belong to the same session, distinguished by an `agent_id` field.

### 2.2 Local Buffering

The CLI maintains an in-memory buffer that flushes to a local append-only log on disk every 100ms or every 50 events, whichever comes first. On graceful shutdown, the buffer drains to disk before exit.

We deliberately **do not fsync per event**. The reasoning: if the agent process crashes, the agent's *work* stops anyway — losing the last 100ms of telemetry from a crashed agent is not the system's worst problem at that moment. Fsync-per-event would put Sploink's durability mechanism on the latency critical path of every agent step, which contradicts the design goal that the agent never blocks on Sploink.

### 2.3 Push Model, Async ACK

The CLI pushes events to Sploink. Sploink does not pull. The agent does not block on the ACK — it emits, keeps working, and the ACK returns asynchronously. The pusher is a background goroutine that reads from the local disk log, sends events over a long-lived WebSocket to the gateway, and waits for ACKs.

We choose WebSocket first because it is simpler to deploy through enterprise proxies and still gives us one long-lived duplex channel per machine. A single user machine running N concurrent agents still multiplexes traffic over **one** connection per machine, not one connection per agent. If we later want stricter generated contracts and richer streaming ergonomics, this layer can move to gRPC without changing the rest of the architecture.

### 2.4 Reverse Control Path (Backend -> CLI)

The ingestion path above is only half of the real system. If we ever want intervention, pause/resume, checkpoint requests, rewind, or fork of a live session, the backend needs a way to talk back to the CLI. The CLI is running on a user's laptop or workstation, usually behind NAT and almost never on a hostable IP, so the backend **cannot** dial it directly.

The correct shape is a **reverse control plane** over the same long-lived outbound connection the CLI already owns:

1. The CLI opens a bidirectional WebSocket at startup and keeps it alive with heartbeats.
2. Uplink frames on that stream carry telemetry events, health, capabilities, and ACKs.
3. Downlink frames on that stream carry control intents addressed to `machine_id` and optionally `session_id`.
4. If the CLI is offline, commands are queued server-side and delivered on the next reconnect if they have not expired.
5. The CLI emits command lifecycle events (`received`, `started`, `completed`, `rejected`, `expired`) into the normal event log so the command path is fully auditable and replayable.

Operationally, this is a reverse tunnel, not inbound RPC. The backend never needs to know a routable address for the user's machine; it only needs a live authenticated stream that the CLI initiated.

The server-side piece is a small control service backed by a second JetStream stream:

```
subject: sploink.commands.{tenant}.{machine_id}
retention: 7d or until terminal command state
```

Every command record includes:

```
{
  command_id,
  tenant_id,
  machine_id,
  session_id,
  lease_id,
  command_type,
  payload,
  issued_at,
  expires_at,
  requires_user_approval,
  signature
}
```

`lease_id` is the fencing token. On reconnect the CLI gets a new lease; any command carrying an old lease is rejected. That prevents a stale stream or duplicated delivery from causing double execution.

The backend should send **control intents**, not arbitrary shell by default. Safe v1 command types are things like `PAUSE_SESSION`, `REQUEST_CHECKPOINT`, `REWIND_TO_STEP`, `FORK_FROM_STEP`, `SWITCH_MODEL_VERSION`, and `PROMPT_FOR_APPROVAL`. If we ever allow raw local command execution, that needs to sit behind explicit tenant policy and local user approval because it is a different trust boundary than telemetry ingestion.

### 2.5 ACK Semantics

The ACK is what authorizes deletion of an event from the local buffer. The protocol has three states:

- **ACK** — first time we've seen this event. Accepted, persisted, you can delete it.
- **DUPLICATE_ACK** — we already have this event from a prior send. You can delete it.
- **(no response)** — network or gateway issue; CLI keeps retrying with backoff.

DUPLICATE_ACK is functionally equivalent to ACK from the buffer-deletion standpoint, but it carries operational signal: if a CLI is seeing DUPLICATE_ACKs frequently, it has a bug in its retry logic, and we surface that as a metric.

### 2.6 Eventually-Consistent Retry

When the network is down or the gateway is unavailable, events stay in the local log. They retry on next successful connection — even if "next successful connection" is three weeks later. There is no upper bound on event age at the SDK side.

This is the right call for Sploink because the audit/replay use case demands completeness, but it has consequences that ripple through the rest of the architecture. Most notably: **late events of arbitrary age are a normal occurrence, not an edge case**. The dedup, scoring, and storage layers all have to handle it.

### 2.7 The Gateway

A thin Go service. It does:
- mTLS authentication against per-tenant certificates.
- Protobuf schema validation.
- Per-tenant rate limiting (sized to detect a runaway tenant, not to absorb bursts; real bursts go through).
- Layer-2 dedup (§4.2).
- Append to NATS JetStream.

It does **not** do ordering, business logic, or any kind of stateful aggregation. Those live downstream where they're testable in isolation. A gateway that "helps" by being smart turns into a hairball within six months.

### 2.8 Idempotency Keying

The CLI computes the idempotency key when it appends an event to its local buffer:

```
idempotency_key = sha256(session_id || turn || agent_id || intra_step || action || input || output)
```

Content-derived. Same content gives the same key, different content gives a different key. We trust the CLI to compute this honestly because the CLI is our code.

This decision is the keystone of the whole dedup story. Two events with the same key are always duplicates of each other regardless of when, how, or how often they arrive. Two events with different keys are always distinct, even if they share `(turn_number, agent_id, intra_turn_step)` — which the simulator deliberately exercises as the "step collision" case.

### 2.9 The Durable Log

NATS JetStream as the durable event spine. We chose NATS over Kafka deliberately:

- Our messages are small (typically <1KB). Kafka's per-message overhead is meaningful at this size.
- We're a Go shop. NATS has best-in-class Go client support.
- JetStream gives us durable streams, per-subject ordering, and at-least-once delivery — the properties we actually rely on — at much lower operational cost than running a Kafka cluster.

Subjects are `sploink.events.{tenant}.{session_id}`. The choice of `session_id` as the key is load-bearing: it guarantees that all events for one session land on one subject, which means one consumer worker owns the full ordering problem for a given session. Every downstream property in this design depends on that.

Retention is 14 days hot in JetStream. Older data is mirrored to S3 (90 days warm) and Iceberg-managed Parquet (∞ cold). Replay queries that target old data go through the warm/cold tiers, not JetStream.

---

## 3. Ordering Strategy

### 3.1 The Problem

Given that events for one session arrive on one NATS subject, in some order that may not match the order they were produced, how do we reconstruct the right order? The complications, all of which we've established are real:

- **Multiple agents in a swarm** are emitting concurrently into the same session. Their events are genuinely concurrent in the distributed-systems sense.
- **Eventually-consistent retries** mean an event from three weeks ago can show up today.
- **Local clock skew** between SDK timestamps and gateway-stamped timestamps.
- **Bursts** — many events sharing a near-identical timestamp.

### 3.2 What Order Even Means

Three notions of "order" are reasonable here:

1. *Wall-clock order* — sort by timestamp. Easy, wrong under skew.
2. *Causal order* — A before B if A's output influenced B. Correct, but requires either vector clocks or explicit dependency edges.
3. *Logical step order* — sort by a counter the agent maintains. Works if there's a single counter; breaks in a swarm where multiple agents have their own counters.

We use a hybrid that lands closest to (3), with a deterministic tiebreaker for swarm concurrency.

### 3.3 The Ordering Key

```
order_key = (turn_number, agent_id, intra_turn_step, ingest_ts)
```

- `turn_number` — the parent session's turn counter, owned by the CLI. Monotonic, single source. Events in turn N happen-before all events in turn N+1.
- `agent_id` — which agent in the swarm produced this event.
- `intra_turn_step` — per-agent counter within a turn. Strictly monotonic per `(turn_number, agent_id)`.
- `ingest_ts` — gateway-stamped wall-clock at first sight. Used **only as a tiebreaker** between swarm-concurrent events.

What this gives us:

- **Across turns**: total order, always correct.
- **Within a turn, same agent**: total order via `intra_turn_step`, always correct.
- **Within a turn, different agents**: tiebroken by `ingest_ts`. Not "true" ordering — these events were genuinely concurrent — but a stable, deterministic ordering that any consumer reading the same data agrees on.

This is honest about what the data actually contains. It claims strong order where the data has it (turn boundaries, single-agent sequences) and admits a deterministic-but-arbitrary tiebreak where the data genuinely doesn't (swarm concurrency).

### 3.4 Late Events

A late event arrives with its original `(turn_number, agent_id, intra_turn_step)` intact. We sort by the key on read, so late events slot back into their correct logical position regardless of when they physically arrived. This is the property that lets eventually-consistent retry not corrupt downstream state.

### 3.5 Watermarks and Turn Finalization

The scoring engine needs a notion of "we've seen everything for turn N." We use a **hybrid finalization model**:

1. The CLI emits an explicit `turn_end` marker as a normal event when a turn completes. The marker has the same ordering key shape as any other event.
2. When the scorer sees the marker, it finalizes that turn — produces a final score and emits it.
3. If no marker arrives within 30 minutes of the most recent event for a turn, the scorer finalizes via timeout. The score is flagged `finalized_via: "timeout"` so consumers know it's less trustworthy.
4. Late events arriving for an already-finalized turn trigger **correction events** — new score records that supersede the prior one. We don't mutate the original.

Corrections are rare by construction: they only happen when (a) a turn was finalized via timeout, *and* (b) a backlog drains later. If the machine was online during the turn, no corrections occur.

### 3.6 What We Don't Solve

We assume **single-machine swarms** — all sub-agents of a session run on the same physical machine. If sub-agents ran on different machines, their `ingest_ts` values would no longer be comparable due to clock skew between machines, and the tiebreaker would become meaningless. Solving cross-machine swarms would require either HLCs (Hybrid Logical Clocks) at the SDK or explicit causal-dependency edges in event payloads. Both are real options we're deferring.

---

## 4. Deduplication Strategy

### 4.1 The Definition

Two events are duplicates iff they share the same idempotency key (§2.7). Same content → same key → caught. Different content → different keys → preserved as siblings, even if they share `(turn_number, agent_id, intra_turn_step)`.

### 4.2 Three Layers

Duplicates arrive from at least five sources: CLI pusher retries, gateway-to-broker retries, manual replay, backlog drains after long offline periods, and CLI bugs. No single layer catches all of them cheaply, so we layer three:

**Layer 1 — CLI-side cache.** The CLI maintains an in-memory set of recently-acknowledged idempotency keys. Before the pusher retries an event, it checks whether the event was already ACKed on a previous send. If yes, drop locally. Catches the most common case (pusher retries) at the cheapest possible point. Cost: a few KB of memory per CLI instance. This is purely an optimization, not a correctness guarantee.

**Layer 2 — Gateway dedup via NATS KV.** The gateway maintains a NATS KV bucket with a 24-hour TTL, keyed by idempotency key. On every incoming event:

- Lookup key in KV. Hit → return DUPLICATE_ACK.
- Miss → write key, forward to JetStream, return ACK.

One lookup, one optional write. Same path either way. Sized at 24h because longer windows trade memory for correctness we already get from Layer 3. NATS KV is the natural choice since we're already on NATS — no extra moving piece — and it scales horizontally per tenant via subject-mapped buckets.

**Layer 3 — Storage-layer upsert.** The materializer and scorer both perform upserts keyed on idempotency key when writing to durable storage. On duplicate key, the operation is a no-op (the content is identical by definition). This is the **end-to-end correctness boundary** — anything that slips past Layers 1 and 2 (long-tail late retries, replays older than 24h, CLI bugs) is caught here. Cost: one extra key lookup per event during materialization. With a properly-indexed storage layer, sub-millisecond.

### 4.3 Why Three Layers, Not One

A pure Layer-3-only design would work, but every duplicate event would consume bandwidth and processing through the streaming layer before being dropped. At 100M events/day a few percent of duplicates is millions of wasted events daily.

A pure Layer-2-only design fails on long-tail retries. Eventually-consistent retry means events can arrive after the 24-hour TTL has expired; without Layer 3, those events would be treated as new and corrupt the graph.

The three layers each catch the cheapest case at their level, and Layer 3 is the unbreakable backstop. This is the same pattern most production systems converge on (Kafka exactly-once semantics, idempotent HTTP APIs, etc.) — duplicates are caught early when convenient, but correctness lives at the storage boundary.

### 4.4 Replay Safety

Layer 3's upsert-on-key behavior makes replay safe by construction. An operator can replay an event stream as many times as they want; every event hits the storage layer, finds its key already present, and becomes a no-op. This is what allows the rewind/fork primitives (§7) to work without coordination — nothing in the system mutates state on duplicate writes.

For re-scoring under a new model: scorer keys are `(idempotency_key, model_version)` rather than just `idempotency_key`. New model version → new key → re-scoring writes new records without overwriting old ones. The audit trail of "what we thought at the time" is preserved.

---

## 5. Storage Model — Events Primary, Graph Derived

### 5.1 The Philosophy Choice

There were three reasonable philosophies for storage:

- **A — Events primary, graph derived.** Event log is the source of truth; graph stores are derived projections that can be rebuilt from the log.
- **B — Graph primary, events as audit trail.** Graph DB holds canonical state; events go to a log mainly for compliance.
- **C — Hybrid co-equal storage.** Both stores written transactionally, both queried directly.

We chose **A**. The arguments:

- *Replayability.* If the graph is primary, replaying or re-scoring under a new model requires either keeping events anyway (defeating the point of B) or reconstructing them from the graph (lossy — we can't recover original timestamps, arrival order, or correction history).
- *Schema migration.* The classifier in `solution.py` uses (action, target) graphs. The intervention engine in Part C might want a different graph (e.g., causal edges between agents in a swarm). Under A, we just rebuild the graph view in a new shape from the log. Under B, schema migrations are nightmares.
- *Out-of-order arrival is easier.* Appending to a log is the simplest possible write. Maintaining graph invariants under late-arrival is much harder; when a three-week-old event lands at step 47, the existing graph has to be reconciled.
- *Two sources of truth always diverge.* Eventually we'd find a session where the graph and the events disagree, and we'd have to pick one as canonical for the bug fix. Better to make that choice up front. (Rules out C.)

### 5.2 Three Tiers

Each tier holds the same data shaped for its access pattern:

| Tier | Engine | Latency Budget | Retention | Use Case |
|------|--------|---------------|-----------|----------|
| Hot  | Postgres + AGE | <50ms p99 | 7 days | Real-time scoring lookups, intervention queries |
| Warm | ClickHouse | 1–5s p99 | 8d–90d | Analytics, classifier retraining, dashboards |
| Cold | S3 + Iceberg Parquet | seconds–minutes | indefinite | Audit, model training, long-tail replay |

**Hot — Postgres + AGE.** Active sessions only (last 7 days). Optimized for "give me the current graph state of session X" queries. Postgres+AGE chosen over Neo4j for operational simplicity: we're already a SQL shop, the AGE extension speaks Cypher when we need it, and the same Postgres cluster serves other operational needs. Schema is portable to a dedicated graph DB if query patterns prove it necessary.

**Warm — ClickHouse.** Two tables: `events` (nodes) and `edges` (relationships). The access pattern here is "scan many sessions, filter on properties" — exactly what columnar is good at. Recursive CTEs would be the bottleneck if `edges` weren't its own first-class table.

**Cold — S3 + Iceberg.** Indefinite retention. Same schema as warm. Read by batch jobs, model training, audit queries, never interactively. Iceberg gives us schema evolution and time-travel for free, which matters when the event schema changes and we need to query old data under a new version.

### 5.3 The Graph-Native Decision: Edges as a First-Class Table

The naive way to model an execution graph in tabular storage is one `events` table with a `parent_event_id` foreign key. Works at small scale. At 100M events/day, every "show me the longest retry chain" query becomes a recursive CTE that takes seconds.

The graph-native alternative: **`edges` is its own first-class table** with one row per edge. Columns: `(session_id, from_node_id, to_node_id, edge_type, created_at)`. Graph queries become column scans instead of recursive joins.

The same logical schema lives in all three tiers:

```
events:
  session_id          string   (partition key)
  tenant_id           string
  turn_number         int
  agent_id            string
  intra_turn_step     int
  ingest_ts           timestamp
  event_ts            timestamp
  action              enum
  input               string
  output              string
  status              enum (success | failure | null)
  latency_ms          float
  target              string
  idempotency_key     string   (unique index — Layer 3 dedup lives here)

edges:
  session_id          string   (partition key)
  from_event_id       string
  to_event_id         string
  edge_type           enum (NEXT | READS | BRANCHED_FROM | RETRIED | VALIDATED_BY)
  created_at          timestamp
```

A query that runs against the warm tier ports to the hot tier with minimal changes. That consistency is on purpose.

### 5.4 The Materializer

The streaming job that reads events from NATS and writes them into the graph stores is doing real work — it's not just an upsert. It has to:

- Determine what new edges to create for an incoming event (e.g., a `READS` edge requires looking at what previous event produced the input).
- Maintain the graph under out-of-order arrival.
- Coordinate with Layer 3 dedup.
- Write graph snapshots to S3 every 25 events (§7).
- Handle corrections from late events that change the graph.

This is where most of the engineering complexity in the storage layer actually lives. The choice of graph DB is mostly aesthetic; the materializer is what makes or breaks the system. We implement it as a stateful Go worker pool consuming from NATS, sharded by `session_id` via consistent hashing across workers, with per-session state held in-process and snapshotted to embedded BadgerDB for crash recovery.

### 5.5 Two-Tier Alternative

We could skip the hot graph store entirely and run everything against ClickHouse with materialized views for common subgraph queries. Trade: query latency for active sessions goes from sub-50ms to a few hundred ms. Operational complexity drops by one whole engine.

Worth considering if the customer doesn't need real-time intervention (Part C), or if we're at the 1M/day or 10M/day tier where ClickHouse can comfortably hold even active-session queries in its sub-second budget. The three-tier design above is what we'd build for the 100M/day target. The two-tier version is what we'd build for a startup at tier 1 in year one and migrate to three-tier as scale demands.

---

## 6. Real-Time Scoring Pipeline

### 6.1 Where the Scorer Sits

The scorer reads from the NATS event log directly, in parallel with the materializer — not downstream of it. This decouples scoring latency from materialization lag, isolates failures (a bad materializer write doesn't corrupt the scoring stream), and keeps replay semantics clean (both materializer and scorer are pure functions of the log, replayable independently).

The cost: the scorer maintains some session state that overlaps with what the materializer maintains. Specifically, the per-session sliding window of recent events. We accept this duplication.

### 6.2 Runtime Shape

The scorer is a Go worker pool consuming from NATS JetStream. Sessions are assigned to workers via a consistent-hash ring on `session_id`, and each worker owns its sessions' state in-process. Per-session state is snapshotted to embedded BadgerDB on local NVMe every 5 seconds; on worker restart, the state is reloaded before the worker starts consuming. This is the same pattern Flink's keyed-state model gives you in JVM-land — we've hand-rolled it because the Go ecosystem doesn't have a single dominant streaming framework and the per-session shape is custom enough that a framework would fight us.

For partitioning, we start static — N workers fixed at deploy time, scale by deploying more — and add dynamic rebalancing only if operational pain forces it.

### 6.3 Triggers

Three trigger conditions, OR'd together:

1. **Event-count trigger** — every 5 new events for a session, emit an interim score.
2. **Wall-clock trigger** — every 10 seconds since the last emission for a session, even if no new events arrived. This is the silent-stall detector; without it, a session that goes quiet would stop being scored.
3. **Turn-end trigger** — on receipt of a `turn_end` marker (or 30-minute timeout finalization), emit a final score.

The first two emit `interim: true` scores. The third emits `interim: false`. Consumers filter on this field — dashboards probably want both, the future intervention engine probably only acts on final scores.

### 6.4 Streaming Aggregates vs Sliding Window

The classifier needs cycle pressure, unique node ratio, output entropy, target entropy, JSD between halves, retry density, success-rate trajectory, max gap. These split cleanly into two computational categories.

**Streaming aggregates** (O(1) per event) for features that have clean incremental forms:
- `retry_density`: running count / total.
- `success_rate_overall`: running mean.
- `success_rate_first_half / second_half`: two running means with mid-session switchover (we don't know the true midpoint until session end, so we switch every time the session length doubles — same trick HLL uses for partition resizing).
- `max_gap_seconds`: running max.
- `unique_node_ratio`: HyperLogLog of (action, target) pairs ÷ total events.
- `target_entropy`: Misra-Gries top-K targets, then approximate entropy.

**Bounded 200-event sliding window** for features that don't have clean incremental forms:
- `cycle_pressure`: count repeated 3-grams in the window.
- `phase_shift` (JSD between halves): JSD between first half of window and second half.
- `output_entropy`: exact entropy over outputs in the window.

This trades a small amount of accuracy (the streaming aggregates are approximate) for huge wins in compute and memory at scale. The exact `featurize()` from `solution.py` becomes the **batch reference implementation**, run periodically against the warm tier as a cross-check.

### 6.5 Lambda Architecture: Streaming + Batch Reconciler

The streaming scorer is approximate. A separate batch reconciler runs every 5 minutes:

- Reads a 5% random sample of active sessions from ClickHouse (full reconciliation across all sessions is too expensive at 100M/day).
- Runs the exact (non-sketch) featurizer over each.
- Compares against the latest streaming score.
- Persistent disagreement on the sample is paged as a sev-2.

The batch path is the reference implementation while the sketches are calibrated. We expect to retire it around v2 once the streaming numbers are trusted, moving to a Kappa-style architecture where reprocessing through the streaming engine is the only path. For v1, the batch reconciler is the safety net that lets us trust streaming scores in production.

### 6.6 Score Records

Scores are themselves events on a separate NATS subject (`sploink.scores.{tenant}.{session_id}`). A score record:

```
{
  session_id: "sess_00042",
  score_ts: 1700000123.4,
  predicted_label: "looping",
  evidence: { progressing: 0.1, looping: 2.3, drifting: 0.0, failing: 0.0 },
  events_seen: 87,
  interim: true,
  finalized_via: null,
  model_version: "v1.2.3"
}
```

The `evidence` field carries the per-class scores from `solution.py`'s classifier, so a human reviewer can see *why* the scorer picked what it did. The `model_version` field is what makes re-scoring under a new model a non-destructive operation (§4.4).

Treating scores as events keeps the system uniform: scores are replayable, scores have idempotency keys (derived from `session_id + score_emission_index + model_version`), and the same three-tier storage materializes them alongside events.

### 6.7 Latency Budget

End-to-end (event arrives at gateway → score emitted to NATS): **<500ms p99**. A degrading session is detectable within half a second. Per-event aggregate update is <100µs. Sliding window update is O(1) amortized. Trigger fires → classify → emit is <5ms typical.

---

## 7. Replay, Rewind, and Fork

All three primitives are read-only against the durable event log. We expose them as gRPC endpoints on a `replay-service` Go binary, with per-tenant rate limits to prevent expensive operations from DOS-ing the live pipeline.

### 7.1 Replay

```
replay(session_id, from_step, to_step, sink)
```

Re-consume events from a chosen offset into a chosen sink. Idempotency keys at every layer make repeat-execution safe; `model_version` in score records makes it useful. The sink parameter is the part that matters — the realistic uses are:

- **A new model version's scoring path.** Re-emits events into a parallel scorer running a different `model_version`. Produces new score records under the new version, leaves old ones intact. The common case.
- **A staging environment.** For reproducing customer issues during debugging.
- **A custom function.** For ad-hoc analysis.
- **The current production pipeline.** No-op due to idempotency; useful for testing the pipeline itself.

For sessions ≤14 days old, replay reads from JetStream (which supports replay-by-sequence cleanly). For sessions 14d–90d, it reads from ClickHouse. For sessions >90d, from S3 Parquet via the Iceberg Go client. Same downstream code paths regardless of source.

### 7.2 Rewind

```
rewind(session_id, to_step) -> graph_snapshot
```

Materialize a session's graph state at an arbitrary step.

We use **snapshot-plus-delta from day one** because we anticipate long-running daemon-style agent sessions where a naive replay-into-memory would be too slow.

**Snapshot mechanism.** Every 25 events for each session, the materializer writes a Parquet snapshot of the current graph state to S3:

```
s3://sploink-snapshots/{tenant}/{session_id}/snapshot_{step}.parquet
```

Each snapshot is the full set of nodes and edges as they existed at that step. Parquet because it's compact, columnar (good for sparse reads), and matches the warm/cold tier format.

**Rewind procedure**:
1. Look up the nearest snapshot at or before `to_step` (small index table of `(session_id, snapshot_steps)`).
2. Read that snapshot from S3 (~30KB-ish per typical session-step).
3. Read events from `snapshot_step+1` to `to_step` from warm/cold tier.
4. Apply those events forward to the snapshot in-memory.
5. Return the resulting graph.

Latency: under 1s for typical rewinds. Even pathological cases (no useful snapshot in a 10,000-step session) are bounded by the cost of replaying ≤25 events, which is sub-second.

If the materializer crashes between snapshots, missing snapshots are reconstructed on demand: rewind notices the gap, replays forward from the previous snapshot, writes the missing snapshot back to S3 as a side effect. Self-healing.

**Storage cost.** A session with 100 nodes and 200 edges is roughly 30KB serialized. At K=25, a 1000-step session has 40 snapshots × 30KB ≈ 1.2MB on top of the events themselves. At 100M events/day across ~1M sessions/day that's roughly 1.2TB/day of snapshot data. Snapshots TTL out alongside events at the cold-tier retention boundary.

### 7.3 Fork

```
fork(session_id, from_step, modifications) -> {
  forked_session_id,
  initial_state,
  divergence_point
}
```

Combines rewind with new-session creation. Reads the rewound state from snapshots, writes it as the genesis state of the new session, records the fork lineage as a session-creation event in the new session's log. The new session's event log starts empty; events accumulate as downstream re-execution emits them.

In Part A, fork is read-only — we present the forked state but don't actually re-run the agent. The intervention engine in Part C extends fork by adding agent re-execution. Recursive forks work as long as each fork creates its own session lineage.

For a **live** agent on a user's machine, backend-side rewind/fork is only half of the story. After the backend computes the desired checkpoint, the control service from §2.4 sends a `REWIND_TO_STEP` or `FORK_FROM_STEP` intent down the CLI-initiated stream. The CLI applies that locally, then emits the resulting execution as ordinary session events.

### 7.4 Why This Is Mostly Free

The combination of decisions made earlier — events as source of truth, idempotency at every layer, model-version tagging on scores — turns replay/rewind/fork from a hard problem into a mostly mechanical one. The only nontrivial implementation work is the snapshot mechanism and the gRPC service. This is the architectural payoff for picking event-sourcing in §5.

---

## 8. Scaling Bottlenecks at Each Operating Point

### 8.1 1M events/day (~12/sec average, ~200/sec burst)

Nothing breaks. Single NATS server. Single Go scoring worker. Single Postgres+AGE node. Single ClickHouse instance. The whole stack runs comfortably on one beefy VM.

The interesting work at this tier is **schema discipline and observability**: every event field added casually here costs us 100× at the highest tier, and every metric we don't emit becomes a debugging gap when traffic grows. We bake in the future-proofing now (consistent-hash sharding, snapshot mechanism, model-version tagging) because adding it later is much harder than doing it from day one.

### 8.2 10M events/day (~120/sec average, ~3,000/sec burst)

**First bottleneck: NATS KV dedup state.** ~10M idempotency keys held with 24h TTL. Response: shard the KV bucket per tenant via subject-mapped buckets, which NATS supports natively.

**Second bottleneck: Postgres+AGE under concurrent active sessions.** At ~5,000–10,000 concurrent sessions, AGE's Cypher query path shows tail latency. Response, in order: PgBouncer in transaction-pool mode with tuned `work_mem`; if that's not enough, partition the AGE schema by tenant and run separate AGE deployments per tenant tier.

**Third bottleneck: scoring worker memory.** ~10K sessions × 200-event window × ~500 bytes/event ≈ 1GB of state per worker if all sessions hashed to one. We run ≥8 workers at this tier with consistent hashing distributing the load. BadgerDB snapshot files start mattering operationally — local NVMe, not network-attached storage.

### 8.3 100M events/day (~1,200/sec average, ~30,000/sec burst)

This is where the architecture stops being boring.

**NATS subject cardinality.** The per-session subject pattern means millions of subjects. JetStream consumer state has per-subject overhead. Response: collapse subjects to `sploink.events.{tenant}.{session_bucket}` where `session_bucket = hash(session_id) % 1024`, and demultiplex inside the consumer. Trades fine-grained per-session isolation for orders of magnitude fewer subjects.

**Postgres+AGE no longer cuts it.** At ~100K concurrent sessions, single-node AGE breaks. Realistic responses, in increasing order of disruption: (a) Tier-2 tenants get demoted to "warm tier only" — sub-second columnar queries instead of sub-50ms graph queries, which many use cases tolerate; (b) Tier-1 tenants migrate to a horizontally-shardable graph engine (TigerGraph, Memgraph, or a managed offering). The schema is portable because we kept it consistent across tiers from day one.

**ClickHouse write amplification.** MergeTree doesn't love high-frequency tiny inserts. Response: micro-batching at the materializer (1-second buffers, ~1,000 events per insert), plus async materialized views for the `edges` table so we're not doing graph reconstruction on the write path.

**Snapshot storage cost.** 1.2TB/day. Mitigations: TTL on snapshots aligned with cold-tier event retention; optional skip-snapshot for tenants who never use rewind (with the tradeoff that rewind for those tenants becomes "full replay from log").

**The single-session-per-subject rule starts to bite.** A rogue session emitting 30k events/sec exceeds a single subject's throughput. Response: per-session sub-sharding for the worst 0.1% of sessions, splitting using `(session_id, turn_number // 100) % N`. We lose strict per-session ordering for that one session, but only that one, and the loss is surfaced via a `sub_sharded: true` flag.

**Lambda batch reconciler.** Reading "all active sessions from ClickHouse" every 5 minutes becomes expensive. Response: sample-based reconciliation — random 5% of sessions per cycle. Persistent disagreement on the sample is still the paged-bug signal; cost drops 20×.

### 8.4 The Cross-Cutting Issue: Cardinality

Several of the above are really one underlying issue — Sploink's per-session-everything model means cardinality explodes with session count. NATS subjects, Postgres rows, snapshot files, dedup keys, score records — all per-session. The architectural response is consistent: bucket per session_id where possible, shard per tenant where necessary. Subject buckets, AGE tenant partitions, S3 hash-prefix partitioning. The pattern is to give up perfect per-session isolation once we have enough sessions that the isolation isn't operationally meaningful anyway.

---

## 9. Tradeoffs We Made Deliberately

The architecture above involved several real forks where reasonable engineers would have picked differently. We're listing them honestly, with what the alternative was and why we rejected it.

1. **Events primary, graph derived (vs graph primary).** Picked events-primary for replayability, schema-migration costs, and out-of-order-handling. Cost: graph queries always go through a materialization layer.

2. **Three-tier storage (vs two-tier).** Three for the 100M/day target. Two-tier is the right call at lower scale and we noted it explicitly. Cost: more engines to run.

3. **Lambda architecture (vs Kappa).** Lambda for v1 because the streaming aggregates use sketches and we want a non-sketch reference. Kappa once the streaming numbers are trusted. Cost: two implementations of the scoring logic in parallel.

4. **Hybrid finalization with corrections (vs wait-for-finalization).** Turn-end markers normally, timeout fallback, corrections for rare late-arrival cases. Cost: every consumer has to handle correction events.

5. **CLI-side idempotency keying (vs gateway-side).** CLI-side because the CLI is Sploink's own trusted code, not customer code. Cost: SDK bugs can in principle break dedup, mitigated by Layer 3 catch-all.

6. **Static worker partitioning (vs dynamic rebalancing).** Static for v1. Cost: scaling up requires deploying more workers and rebalancing partitions, a known operation but not zero-downtime.

7. **Three-layer dedup (vs single layer).** Layered because no single layer catches everything cheaply. Cost: dedup logic exists in three places that all have to stay consistent.

8. **24-hour gateway dedup window (vs longer).** 24h because longer windows trade memory for correctness we already get from the storage layer. Cost: events older than 24h hitting the gateway slip through to storage as "new," relying on storage upserts to catch them.

9. **`ingest_ts` as concurrent-event tiebreaker (vs HLCs).** Pragmatic. Cost: the tiebreaker is meaningless if cross-machine swarms ever become real.

### 9.1 What We Explicitly Deferred

- **Cross-machine swarms.** Sub-agents of one session running on different physical machines. Solving this requires HLCs at the SDK or explicit causal-dependency edges in event payloads.
- **Cross-tenant real-time queries.** Real-time queries across tenants aren't supported; cross-tenant analytics happens as batch jobs over the cold tier.
- **SDK-side HLCs.** Heavyweight to deploy across all customer CLI installations; not worth it until cross-machine swarms force it.

### 9.2 What v2 Looks Like

- Retire the batch reconciler and move to Kappa once streaming sketch numbers are trusted (~12–18 months).
- Per-tenant custom classifiers — let tier-1 tenants train custom models on their own warm-tier history, scored alongside the global model.
- Online model updates via shadow scoring: every score-event topic carries both production and experimental model verdicts, A/B'd in the open before promotion.
- Move dedup state from NATS KV to FoundationDB once the per-tenant sharding pattern shows operational pain.

The system above is sized for 18 months of growth at the current rate. The v2 changes are the things that come due before month 18.
