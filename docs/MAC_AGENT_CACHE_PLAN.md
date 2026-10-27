# Mac Agent Cache Plan

## Goal

Create one local, provenance-aware cache on the MacBook that can be used by:

- Codex CLI in Terminal
- Claude Code in Terminal
- the CATCH/V-Engine web app
- future voice/realtime sessions

The cache is a correctness boundary, not a generic prompt cache.

## Target architecture

```
                         ┌──────────────────────────┐
                         │  MacBook Agent Cache     │
                         │  SQLite + artifact store │
                         └────────────┬─────────────┘
                                      │
                  ┌───────────────────┼──────────────────┐
                  │                   │                  │
             Codex CLI          Claude Code         Web / Voice
                  │                   │                  │
                  └──────────────┬────┴──────────────────┘
                                 │
                         v-agent-cache CLI
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
          verified results                  traces/artifacts
                │                                 │
                └────────────────┬────────────────┘
                                 │
                         CATCH / verifier
```

The local cache should be a shared service/CLI rather than separate cache directories owned by each model.

## What should be cached

### Reusable

- independently VERIFIED deterministic computations
- verified test/build/lint results
- verified repository facts tied to a commit SHA
- verified tool outputs whose inputs and tool revision are stable
- generated artifacts addressed by content hash
- expensive research/tool results with explicit provenance and freshness policy
- normalized summaries that are explicitly marked reusable

### Never replay as a successful result

- UNKNOWN or FAILED verification
- external side effects
- secret access
- production deployments
- destructive commands
- credentials/tokens
- raw private conversation history as a generic reusable answer

For side effects, cache the plan/evidence if useful, but execute the action again under policy and verify the resulting state.

## Cache identity

Every entry must be keyed from canonical structured provenance, not just prompt text.

Minimum identity:

- canonical task input
- model/provider identity
- model revision
- code revision / Git commit
- tool registry revision
- schema version
- policy revision
- execution class
- dependency fingerprint
- environment fingerprint where relevant

Recommended key:

```
SHA256(canonical_json({
  task_input,
  model,
  model_revision,
  code_revision,
  tool_revision,
  schema_version,
  policy_revision,
  execution_class,
  dependency_fingerprint,
  environment_fingerprint
}))
```

Do not include the conversation/session ID unless the result is intentionally session-scoped. This is what allows a verified result to be reused across Claude, Codex, web, and voice sessions.

## Verification contract

Each cache record should contain:

```json
{
  "cache_key": "...",
  "status": "VERIFIED",
  "result": "...",
  "provenance": {
    "agent": "codex|claude|web|voice",
    "model": "...",
    "model_revision": "...",
    "git_commit": "...",
    "tool_revision": "...",
    "policy_revision": "...",
    "execution_class": "READ_ONLY|COMPUTE|EXTERNAL_SIDE_EFFECT|SECRET_ACCESS|PRODUCTION_DEPLOY"
  },
  "verifier": {
    "name": "...",
    "version": "...",
    "status": "VERIFIED"
  },
  "created_at": "...",
  "expires_at": null
}
```

The cache writer must reject anything except VERIFIED results.

## Mac implementation

Use a local Unix-domain or localhost service with:

- SQLite metadata database
- content-addressed artifact directory
- one CLI: `v-agent-cache`
- optional HTTP API for web/voice development
- file permissions restricted to the local user
- atomic writes
- locking for concurrent Codex/Claude processes
- garbage collection and TTL support

Suggested locations:

```
~/.v-agent/
  cache.db
  artifacts/
  traces/
  locks/
  config.toml
```

Do not put secrets in the cache.

## CLI contract

Design the CLI around explicit operations:

```bash
v-agent-cache get <cache-key>
v-agent-cache put <cache-key> --status VERIFIED --provenance provenance.json --input input.json --result result.json
v-agent-cache inspect <cache-key>
v-agent-cache verify <cache-key>
v-agent-cache invalidate <cache-key>
v-agent-cache gc
v-agent-cache stats
v-agent-cache doctor
```

For agent integration, also provide a higher-level operation:

```bash
v-agent-cache run --input input.json --execution-class READ_ONLY --command '...'
```

The command must:

1. canonicalize the input;
2. compute the provenance-aware key;
3. policy-check the execution;
4. check cache;
5. return a hit only when compatible and VERIFIED;
6. otherwise execute;
7. run the visible verifier;
8. run an independent verifier where required;
9. write only VERIFIED results;
10. emit a trace record.

## Codex integration

Codex should call the shared cache through a small wrapper/hook, not maintain its own semantic cache.

For every expensive operation:

1. inspect the current Git commit;
2. construct canonical task identity;
3. ask `v-agent-cache` for a compatible VERIFIED result;
4. on miss, perform the operation;
5. verify it;
6. write the verified result;
7. continue the trajectory.

Examples:

- repository inspection
- deterministic tests
- static analysis
- dependency metadata
- build results
- generated artifacts
- repeated research/tool calls

Never allow a cache hit to skip a required policy or safety check.

## Claude Code integration

Claude Code should use exactly the same cache protocol.

Create a thin adapter that translates Claude's tool/task representation into the canonical V-Agent task schema.

The adapter must not make Claude-specific cache keys.

That gives:

```
Claude result → verifier → shared cache
Codex result  → verifier → shared cache
Web result    → verifier → shared cache
Voice result  → verifier → shared cache
```

A VERIFIED result produced by one worker can therefore become evidence available to another worker, subject to provenance compatibility.

## Web integration

The web app should not directly trust browser localStorage as the authoritative cache.

Use:

```
Browser → V-Engine API → policy → shared cache → verifier
```

Browser localStorage may hold only UI/session convenience state.

The server-side persistent cache should remain authoritative.

For the current CATCH Vercel demo, the existing Supabase-backed cache is useful as the remote persistence layer, but it should eventually share the same schema and cache-key contract as the Mac cache.

## Cross-device model

Use two tiers:

1. **Local cache**
   - fastest
   - private
   - useful for Terminal/agent development
   - stores local environment-sensitive results

2. **Remote verified cache**
   - Supabase/Postgres or another durable store
   - shared by web/voice and optionally Mac clients
   - only stores records safe to share
   - never stores secrets

Promotion rule:

```
local VERIFIED
      ↓
shareability policy
      ↓
remote VERIFIED
```

Do not automatically upload every local cache entry.

## Security requirements

The current demo database policy is not sufficient for a production multi-user cache because anonymous users can access rows broadly.

Before production:

- introduce authenticated user/workspace identity;
- bind session and cache records to an owner/workspace;
- enforce RLS by authenticated identity;
- separate private local cache from shareable remote cache;
- never expose service-role credentials to browser code;
- redact secrets from traces;
- encrypt sensitive local artifacts if they must exist at rest;
- provide explicit cache invalidation.

## Performance strategy

Use a three-level lookup:

1. in-process memory for the current worker;
2. local SQLite/content-addressed cache on the Mac;
3. remote verified cache for shared results.

Lookup order:

```
memory → local → remote → execute → verify → local → optional remote promotion
```

This gives speed without turning stale data into truth.

## Important distinction: context vs cache

Do not use the cache as a substitute for model context.

Use:

- memory for durable user/project facts;
- conversation state for the current interaction;
- cache for verified reusable computation/evidence;
- artifacts for durable outputs;
- traces for audit/replay.

This separation prevents the "giant prompt in a database" failure mode.

## Rollout phases

### Phase A: local foundation

- implement `v-agent-cache`;
- SQLite schema;
- canonical cache key;
- provenance schema;
- locking;
- CLI;
- unit tests.

### Phase B: Codex + Claude

- adapters/hooks;
- shared task identity;
- cache hit/miss telemetry;
- verification enforcement;
- invalidation.

### Phase C: CATCH integration

- connect cache writes to CATCH trajectories/rewards/traces;
- record verifier disagreement;
- measure cache-induced reward hacking;
- prevent stale evidence from inflating reward.

### Phase D: web + voice

- server-side cache API;
- authenticated ownership;
- realtime sideband integration;
- cache lookup before expensive tool/model operations;
- verified result propagation across sessions.

### Phase E: research/evaluation

Measure:

- cache hit rate
- verified-hit rate
- false-hit rate
- stale-hit rate
- verifier disagreement
- time saved
- token/model cost saved
- reward changes with and without cache
- reward-hacking attempts involving cached evidence

## Definition of done

The implementation is not complete until:

- Codex and Claude can use the same local cache;
- cache keys are provenance-aware;
- a cache hit cannot bypass policy;
- UNKNOWN/FAILED results cannot be promoted;
- side effects are never replayed;
- local and remote caches use compatible schemas;
- cache records can be inspected and invalidated;
- concurrent agents cannot corrupt the store;
- tests cover stale provenance and cross-agent reuse;
- CATCH traces can explain why a cached result was accepted;
- authenticated remote isolation is enforced;
- web/voice can use the same verified-cache contract.

---

# Implementation prompt for Codex or Claude Code

You are implementing the Mac shared verified-cache layer for the CATCH/V-Agent project.

Repository:
`sachin7x/CATCH`

Follow this workflow exactly:

**inspect → define → modify → test → verify → report**

Do not guess repository state.

## Mission

Build a production-oriented local cache service on macOS that can be shared by Codex CLI, Claude Code, the CATCH/V-Agent runtime, and the web/voice runtime.

The cache must make agents faster without making them less correct.

## Non-negotiable invariants

1. Cache is a correctness boundary, not merely a performance optimization.
2. Only independently VERIFIED results may be reusable cache entries.
3. UNKNOWN and FAILED results are never reusable success artifacts.
4. Every lookup performs policy/provenance compatibility checks.
5. Cache identity must include task input, model/provider revision, code revision, tool revision, schema version, policy revision, execution class, and relevant dependency/environment fingerprints.
6. Session ID must not be part of the cache key unless the artifact is intentionally session-scoped.
7. Never replay external side effects, secret access, or production deployment.
8. Never store credentials, API keys, or secrets in cache records.
9. Local and remote cache entries must use compatible provenance semantics.
10. Every cache decision must be traceable.
11. Preserve the existing CATCH reward/verifier model. Do not weaken verification merely to increase hit rate.

## First inspect

Inspect:

- current branch and PR state;
- `v_app/cache.py`;
- `v_app/models.py`;
- `v_app/policy.py`;
- `v_app/verifier.py`;
- `v_app/runtime.py`;
- `v_app/memory.py`;
- `v_app/catch_bridge.py`;
- `vercel-app/api/index.py`;
- `docs/CACHE_RULE.md`;
- existing cache/runtime tests;
- CATCH trace/reward primitives relevant to recording cache decisions.

Identify what already exists before adding abstractions.

## Implement

Prefer a small Python package such as:

```
v_cache/
  __init__.py
  models.py
  canonical.py
  store.py
  policy.py
  service.py
  cli.py
  artifacts.py
  locks.py
```

Use SQLite for metadata and content-addressed files for large artifacts.

Expose a stable API for:

- get
- put_verified
- inspect
- invalidate
- garbage collection
- statistics
- doctor/health
- execute-with-cache

Add a `v-agent-cache` executable through the existing project packaging.

## Canonicalization

Implement deterministic canonical serialization.

Never construct keys by concatenating arbitrary strings.

Use canonical JSON followed by SHA-256.

Add tests proving that:

- equivalent structured input yields the same key;
- changed model revision changes the key;
- changed code revision changes the key;
- changed tool revision changes the key;
- changed policy revision changes the key;
- changed execution class changes the key;
- changed environment/dependency fingerprint changes the key;
- session ID does not change a reusable key.

## Concurrency

The store must tolerate multiple local agent processes.

Use safe SQLite transactions and appropriate file locking where needed.

Test concurrent read/write behavior.

## Agent adapters

Create minimal adapters/interfaces so Codex and Claude can call the same cache protocol.

Do not hard-code provider-specific semantic cache keys.

Do not require either model vendor to know the SQLite schema.

## CATCH integration

Record cache events into the existing trace/evidence model where appropriate:

- cache_lookup
- cache_hit
- cache_miss
- cache_rejected
- cache_write
- cache_invalidated

Record the provenance and verifier responsible for acceptance.

If a cached artifact later fails independent verification, invalidate it and record the disagreement.

## Remote cache

Do not duplicate the remote Supabase implementation blindly.

Define a backend interface:

```
CacheBackend
├── LocalSQLiteBackend
└── RemoteVerifiedBackend
```

The local backend is required now.

The remote backend may be an adapter over the existing Vercel/Supabase path, but preserve one canonical schema and key format.

## Security

Review the existing Supabase RLS.

The current anonymous-all policy is acceptable only as a development demo.

Add a production-safe design for authenticated owner/workspace isolation. If auth infrastructure is not yet present, implement the schema/interface and tests without pretending the security boundary is complete.

## Web and voice

Expose a server-side API that can use the same cache service.

Do not put authoritative reusable cache state in browser localStorage.

Voice sessions should call the cache outside the realtime conversation state so the cache survives session boundaries.

## Testing

Add focused tests for:

- key canonicalization;
- provenance mismatch;
- policy mismatch;
- VERIFIED-only writes;
- UNKNOWN/FAILED rejection;
- side-effect non-replay;
- secret-access non-replay;
- production-deploy non-replay;
- cross-session reuse;
- cross-agent reuse;
- concurrent access;
- invalidation;
- artifact integrity;
- trace emission.

Run targeted tests first, then the relevant project test suite.

If the repository has pre-existing failures, distinguish them from failures caused by this change.

## Verification

Verify from actual repository/tool output:

- changed files;
- exact commit SHA;
- test results;
- CI/check results;
- PR state;
- deployment state if web code changes.

Never report success based on intention.

## Final report

Return exactly:

1. VERIFIED: what changed, with exact paths.
2. VERIFIED: tests/checks actually run.
3. VERIFIED: commit SHA and branch.
4. VERIFIED: PR status.
5. INFERRED: expected performance/architecture benefits.
6. UNKNOWN: anything not yet verified.
7. NEXT: the smallest remaining blocker.

Do not mark the PR ready merely because the local cache works. Mark it ready only when the implementation and verification gates are actually satisfied.
