from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Route:
    agent_id: str
    reason: str
    confidence: float


def route_scenario(
    *,
    requires_voice: bool,
    requires_tools: bool,
    requires_browser: bool,
    preferred_agent: str | None = None,
) -> Route:
    if preferred_agent:
        return Route(preferred_agent, "explicit preference", 1.0)
    if requires_browser:
        return Route("browser-capable-agent", "browser capability required", 0.95)
    if requires_voice or requires_tools:
        return Route("realtime-capable-agent", "realtime/tool capability required", 0.9)
    return Route("general-agent", "default capability match", 0.5)
