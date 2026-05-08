# agents/base.py — Helios shared agent base class
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from observability.metrics import agent_latency_histogram, agent_error_counter
from resilience.circuit_breaker import get_breaker, CircuitBreakerOpen
from resilience.bulkhead import bulkhead, BulkheadRejected


class BaseAgent(ABC):
    """
    Shared scaffolding for all Helios agents:
      - Bulkhead isolation: per-agent concurrency cap via a thread semaphore
      - Circuit breaker: fast-fail when an agent is consistently failing
      - Prometheus latency histogram + error counter
      - Structured logging
    """

    name: str = "base"

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"helios.agents.{self.name}")

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Public entry point called by LangGraph.
        Wraps _run() with bulkhead, circuit breaker, timing, and error counting.
        """
        t0 = time.perf_counter()
        breaker = get_breaker(self.name)

        try:
            with bulkhead(self.name):
                result = breaker.call_sync(self._run, state)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            agent_latency_histogram.labels(agent=self.name).observe(elapsed_ms)
            self.logger.info("%s finished in %.1f ms", self.name, elapsed_ms)
            return result

        except BulkheadRejected as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            agent_error_counter.labels(agent=self.name).inc()
            self.logger.warning(
                "%s bulkhead full after %.1f ms: %s", self.name, elapsed_ms, exc
            )
            return {**state, "error": f"Capacity exceeded: {exc}", "failed_agent": self.name}

        except CircuitBreakerOpen as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            agent_error_counter.labels(agent=self.name).inc()
            from observability.metrics import circuit_breaker_trips_counter
            circuit_breaker_trips_counter.labels(circuit=self.name).inc()
            self.logger.warning(
                "%s circuit open after %.1f ms: %s", self.name, elapsed_ms, exc
            )
            return {**state, "error": f"Service unavailable: {exc}", "failed_agent": self.name}

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            agent_error_counter.labels(agent=self.name).inc()
            self.logger.exception(
                "%s raised %s after %.1f ms: %s",
                self.name, type(exc).__name__, elapsed_ms, exc,
            )
            return {**state, "error": str(exc), "failed_agent": self.name}

    @abstractmethod
    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Agent-specific logic. Must return updated state dict."""
        ...
