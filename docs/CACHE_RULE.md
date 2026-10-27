# Cache rule

The cache is a correctness boundary, not merely a latency optimization.

## Identity

Include canonical task input, model/provider identity and revision, code revision, tool registry revision, schema version, policy revision, and execution class. Add dependency and environment fingerprints before production use.

## Write rule

Only a result independently verified as successful may enter the reusable cache. FAILED and UNKNOWN results are evidence, never reusable success artifacts.

## Read rule

Every lookup re-checks policy. A hit must be provenance-compatible with the current task. Any incompatible revision is a miss.

## Side effects

Never replay external side effects, secret access, or production deployment from cache. Cache plans/evidence if useful, then execute again under policy and verify the resulting state.

## Research value

CATCH reward and monitor code can measure cache-induced reward hacking, stale-evidence exploitation, verifier disagreement, and attempts to manufacture verified-looking artifacts.
