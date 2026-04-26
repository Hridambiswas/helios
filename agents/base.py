# agents/base.py — Helios shared agent base class
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from observability.metrics import agent_latency_histogram, agent_error_counter


class BaseAgent(ABC):
    """
    Shared scaffolding for all Helios agents:
      - structured logging with agent name
      - latency instrumentation (Prometheus histogram)
      - error counting
    """

    name: str = "base"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"helios.agents.{self.name}")

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Public entry point called by LangGraph.
        Wraps _run() with timing and error counting.
        """
        t0 = time.perf_counter()
        try:
            result = self._run(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            agent_latency_histogram.labels(agent=self.name).observe(elapsed_ms)
            self.logger.info("%s finished in %.1f ms", self.name, elapsed_ms)
            return result
        except Exception as exc:
            agent_error_counter.labels(agent=self.name).inc()
            self.logger.exception("%s raised %s: %s", self.name, type(exc).__name__, exc)
            return {**state, "error": str(exc), "failed_agent": self.name}

    @abstractmethod
    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Agent-specific logic. Must return updated state dict."""
        ...
