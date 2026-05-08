# resilience/backpressure.py — Helios Backpressure / Load Shedding
# Author: Hridam Biswas | Project: Helios
#
# Two signals drive backpressure:
#   1. Celery queue depth — proxy for pending work in the system
#   2. Active in-flight pipeline count — proxy for current CPU/LLM pressure
#
# When either threshold is exceeded the request is rejected with 503
# so upstream can retry or callers can back off.

from __future__ import annotations
import logging

logger = logging.getLogger("helios.resilience.backpressure")

# Global counter for in-flight synchronous pipelines (GIL protects int ops)
_active_pipelines: int = 0


class BackpressureError(Exception):
    pass


async def check_backpressure() -> None:
    """
    Raise BackpressureError if the system is overloaded.
    Safe to call from async route handlers; Redis errors are swallowed
    so a cache outage never blocks requests.
    """
    from config import cfg
    from observability.metrics import backpressure_shed_counter

    # ── Signal 1: active in-flight pipelines ─────────────────────────────────
    if _active_pipelines >= cfg.backpressure_active_pipelines_threshold:
        backpressure_shed_counter.inc()
        logger.warning(
            "Backpressure: active_pipelines=%d >= threshold=%d",
            _active_pipelines, cfg.backpressure_active_pipelines_threshold,
        )
        raise BackpressureError(
            f"System at capacity — {_active_pipelines} pipelines active"
        )

    # ── Signal 2: Celery queue depth ──────────────────────────────────────────
    try:
        import redis.asyncio as aioredis
        client: aioredis.Redis = aioredis.from_url(cfg.celery_broker_url, decode_responses=True)
        raw = await client.llen("celery")  # type: ignore[misc]
        depth: int = int(raw)
        await client.aclose()
        if depth > cfg.backpressure_queue_depth_threshold:
            backpressure_shed_counter.inc()
            logger.warning(
                "Backpressure: celery queue depth=%d > threshold=%d",
                depth, cfg.backpressure_queue_depth_threshold,
            )
            raise BackpressureError(f"Queue overloaded — depth {depth}")
    except BackpressureError:
        raise
    except Exception:
        pass  # Redis unreachable → don't shed on uncertainty


class active_pipeline:
    """
    Async context manager that tracks in-flight synchronous pipeline executions.

    Usage::

        async with active_pipeline():
            state = run_pipeline(query)
    """

    async def __aenter__(self) -> "active_pipeline":
        global _active_pipelines
        _active_pipelines += 1
        from observability.metrics import active_pipelines_gauge
        active_pipelines_gauge.set(_active_pipelines)
        return self

    async def __aexit__(self, *_: object) -> None:
        global _active_pipelines
        _active_pipelines -= 1
        from observability.metrics import active_pipelines_gauge
        active_pipelines_gauge.set(_active_pipelines)


def get_active_pipeline_count() -> int:
    return _active_pipelines
