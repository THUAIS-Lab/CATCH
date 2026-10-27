"""Thin integration layer over CATCH primitives. The app owns orchestration; CATCH owns agent/eval primitives."""
from typing import Any
from rllm.agents.agent import BaseAgent, Step, Trajectory
from rllm.rewards.reward_types import RewardOutput
from rllm.sdk.protocol import Trace
from rllm.tools.registry import ToolRegistry
from rllm.tools.tool_base import Tool, ToolOutput

def trajectory_from_agent(agent: BaseAgent) -> Trajectory:
    return agent.trajectory

def trace_metadata(trace: Trace) -> dict[str, Any]:
    return {"trace_id": trace.trace_id, "session_name": trace.session_name, "model": trace.model, "latency_ms": trace.latency_ms, "tokens": trace.tokens}

def reward_metadata(reward: RewardOutput) -> dict[str, Any]:
    return {"reward": reward.reward, "is_correct": reward.is_correct, **reward.metadata}

__all__ = ["BaseAgent", "Step", "Trajectory", "Trace", "Tool", "ToolOutput", "ToolRegistry", "trajectory_from_agent", "trace_metadata", "reward_metadata"]
