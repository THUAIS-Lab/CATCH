# Open Agent App architecture

This is not a copy of proprietary ChatGPT source code. It is an independently implementable architecture for the same broad capability categories: conversation, sessions, model routing, tools, files, memory, realtime, background work, approvals, artifacts, evaluation, and a web client.

## Core law

TASK -> POLICY -> CACHE LOOKUP -> AGENT -> TRACE -> VISIBLE VERIFIER -> INDEPENDENT VERIFIER -> CLASSIFY -> VERIFIED ARTIFACT

A cache hit never bypasses policy. A reusable cache entry must be VERIFIED and provenance-compatible. Side-effecting actions are never replayed from cache.

## Product planes

- Web plane: chat UI, conversations, attachments, task status, artifacts.
- Conversation plane: sessions, context assembly, memory, summaries, truncation, streaming.
- Agent plane: CATCH BaseAgent, tool agents, SWE/code/browser agents, model adapters.
- Tool plane: CATCH registry/MCP/tool abstractions with explicit execution classes.
- Realtime plane: normalized event stream, audio transport adapters, interruption, function calls.
- Control plane: routing, scheduling, retry/recovery, approvals, quotas, policy.
- Evidence plane: CATCH trajectories, SDK traces, verifier outputs, provenance.
- Cache plane: deterministic identity, verified artifacts, invalidation, replay.
- Research plane: CATCH rewards, monitors, trainer, datasets, RLVR/GRPO experiments.

## Freedom boundary

The deployment owner controls runtime, tools, data, model routing, storage, and policies. Freedom does not mean bypassing authorization or third-party terms.

## CATCH reuse map

Agent state -> rllm.agents.agent.BaseAgent, Trajectory, Step
Tool calling -> rllm.agents.tool_agent.ToolAgent and rllm.tools
Tool registration -> rllm.tools.registry and Tool
Parsing -> rllm.parser
Evaluation -> rllm.rewards and monitors
Workflows -> rllm.workflows
Traces -> rllm.sdk.protocol.Trace and SDK store/session
Training -> rllm.trainer
Trajectory inspection -> rllm.trajectory_visualizer

The application layer should import these abstractions instead of forking duplicate implementations.
