# Realtime Orchestrator

A verifier-first orchestration layer for Realtime Conversations, derived from the CATCH research model.

## Goals

- Treat every realtime interaction as an auditable event trace.
- Separate the agent's visible/proxy reward from an independent audit reward.
- Support WebRTC, WebSocket, tool/function calls, VAD, interruption/truncation, out-of-band responses, and replay.
- Make long-running work resumable and observable.
- Keep scheduling and automation outside the model's control plane.
- Require explicit policy gates for high-risk actions.

## Control plane

```
trigger -> run coordinator -> agent -> realtime environment
                         -> trace store
                         -> visible verifier
                         -> independent verifier
                         -> reward/audit
                         -> artifact + metrics
                         -> next action / human gate
```

## Event classes

`session`, `conversation`, `audio`, `response`, `tool`, `interrupt`, `verification`, `reward`, `policy`, and `automation`.

## Automation contract

The orchestrator is event-driven, idempotent, and resumable. Every run has:

- run_id
- scenario_id
- agent_id
- model_id
- attempt
- parent_run_id
- policy snapshot
- event trace
- verifier results
- reward record
- artifact manifest

## CATCH relationship

CATCH remains the research baseline. This layer intentionally does not make the realtime model its own judge. The independent verifier is authoritative for genuine success and hack classification.
