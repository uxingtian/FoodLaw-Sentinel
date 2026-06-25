from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app.models import ChatResponse


@dataclass
class RuntimeMetrics:
    total_requests: int = 0
    fallback_requests: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    role_counts: dict[str, int] = field(
        default_factory=lambda: {"regulator": 0, "consumer": 0, "producer": 0, "general": 0}
    )
    tool_counts: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict:
        avg_latency = self.total_latency_ms / self.total_requests if self.total_requests else 0.0
        return {
            "total_requests": self.total_requests,
            "fallback_requests": self.fallback_requests,
            "fallback_rate": round(self.fallback_requests / self.total_requests, 4) if self.total_requests else 0.0,
            "avg_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "role_counts": dict(self.role_counts),
            "tool_counts": dict(self.tool_counts),
        }


class MetricsRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics = RuntimeMetrics()

    def record_chat(self, response: ChatResponse) -> None:
        trace = response.route.get("trace", {})
        latency = float(trace.get("total_ms", 0.0) or 0.0)
        tools = response.route.get("tools", [])
        with self._lock:
            self._metrics.total_requests += 1
            if response.fallback_used:
                self._metrics.fallback_requests += 1
            self._metrics.total_latency_ms += latency
            self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency)
            self._metrics.role_counts[response.role] = self._metrics.role_counts.get(response.role, 0) + 1
            for tool in tools:
                name = tool.get("name", "unknown")
                self._metrics.tool_counts[name] = self._metrics.tool_counts.get(name, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return self._metrics.snapshot()

    def reset(self) -> None:
        with self._lock:
            self._metrics = RuntimeMetrics()


metrics_recorder = MetricsRecorder()
