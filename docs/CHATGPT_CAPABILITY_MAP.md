# Capability map

| Capability category | Open implementation target | Status |
|---|---|---|
| Chat UI | Next.js client | planned |
| Streaming text | SSE/WebSocket adapter | planned |
| Conversation persistence | Postgres/SQLite repository | planned |
| File upload/search | object store + index | planned |
| Model routing | provider-neutral adapter | planned |
| Tool calling | CATCH tools/MCP | substrate exists |
| Code execution | sandbox adapter | planned |
| Browser automation | browser adapter | planned |
| Realtime voice | WebRTC/WebSocket adapters | bootstrap exists in branch |
| Memory | explicit user-controlled store | v0 exists |
| Background tasks | scheduler + worker queue | planned |
| Approvals | policy gate | v0 exists |
| Artifacts | content-addressed artifact store | planned |
| Replay | trace-driven replay | planned |
| Evaluation | CATCH rewards/verifiers | substrate exists |
| RLVR/GRPO | CATCH trainer | substrate exists |

This is capability decomposition, not access to OpenAI's private implementation.
