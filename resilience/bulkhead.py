# resilience/bulkhead.py — Helios Bulkhead Isolation
# Author: Hridam Biswas | Project: Helios
#
# Limits concurrent executions per agent via thread semaphores.
# If all slots are occupied, new calls are rejected immediately
# (non-blocking acquire) rather than queued — this is the bulkhead
# pattern: isolate failures so one overloaded agent cannot starve others.

from __future__ import annotations
import threading
import logging

logger = logging.getLogger("helios.resilience.bulkhead")

# Per-agent concurrency limits
_AGENT_LIMITS: dict[str, int] = {
    "executor":    3,   # sandboxed code — most resource-intensive
    "retriever":   8,
    "planner":    10,
    "synthesizer": 10,
    "critic":     10,
}

_pools: dict[str, threading.BoundedSemaphore] = {}
_lock = threading.Lock()


def _get_semaphore(agent_name: str) -> threading.BoundedSemaphore:
    with _lock:
        if agent_name not in _pools:
            from config import cfg
            limit = _AGENT_LIMITS.get(agent_name, cfg.bulkhead_default_limit)
            _pools[agent_name] = threading.BoundedSemaphore(limit)
            logger.debug("Bulkhead created for '%s' (limit=%d)", agent_name, limit)
        return _pools[agent_name]


class BulkheadRejected(Exception):
    pass


class bulkhead:
    """
    Context manager that acquires a semaphore slot or raises BulkheadRejected.

    Usage::

        with bulkhead("executor"):
            result = heavy_work()
    """

    def __init__(self, agent_name: str) -> None:
        self._name = agent_name
        self._sem = _get_semaphore(agent_name)

    def __enter__(self) -> "bulkhead":
        acquired = self._sem.acquire(blocking=False)
        if not acquired:
            from observability.metrics import bulkhead_rejected_counter
            bulkhead_rejected_counter.labels(agent=self._name).inc()
            raise BulkheadRejected(
                f"Bulkhead for agent '{self._name}' is full — try again later"
            )
        return self

    def __exit__(self, *_: object) -> None:
        self._sem.release()
