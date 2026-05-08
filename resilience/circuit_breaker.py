# resilience/circuit_breaker.py — Helios Circuit Breaker
# Author: Hridam Biswas | Project: Helios
#
# States:
#   CLOSED   — normal operation; failures are counted
#   OPEN     — fast-fail all calls; entered after failure_threshold consecutive failures
#   HALF_OPEN — one probe call allowed through after recovery_timeout elapses;
#               success → CLOSED, failure → back to OPEN

from __future__ import annotations
import logging
import threading
import time
from enum import Enum
from typing import Any, Awaitable, Callable


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker usable from both sync (Celery) and
    async (FastAPI/LangGraph) contexts.  Uses threading.Lock so it is
    safe across threads without requiring an event loop.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"helios.resilience.cb.{name}")

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    # ── Internal state transitions ────────────────────────────────────────────

    def _maybe_transition_to_half_open(self) -> None:
        """Call under lock. OPEN → HALF_OPEN when recovery_timeout has elapsed."""
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._logger.info("Circuit '%s' → HALF_OPEN", self.name)

    def _on_success(self) -> None:
        with self._lock:
            if self._state != CircuitState.CLOSED:
                self._logger.info("Circuit '%s' → CLOSED (recovered)", self.name)
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._logger.warning(
                        "Circuit '%s' → OPEN after %d failures",
                        self.name, self._failure_count,
                    )
                self._state = CircuitState.OPEN

    def _acquire(self) -> None:
        """Check whether a call is permitted; raise CircuitBreakerOpen if not."""
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpen(
                    f"Circuit '{self.name}' is OPEN — service unavailable"
                )

    # ── Call wrappers ─────────────────────────────────────────────────────────

    def call_sync(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Wrap a synchronous callable."""
        self._acquire()
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpen:
            raise
        except Exception:
            self._on_failure()
            raise

    async def call_async(
        self, coro_func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        """Wrap an async callable."""
        self._acquire()
        try:
            result = await coro_func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpen:
            raise
        except Exception:
            self._on_failure()
            raise


# ── Registry ──────────────────────────────────────────────────────────────────

_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    """Return (or lazily create) the named circuit breaker."""
    with _registry_lock:
        if name not in _registry:
            from config import cfg
            _registry[name] = CircuitBreaker(
                name,
                failure_threshold=cfg.circuit_breaker_failure_threshold,
                recovery_timeout=cfg.circuit_breaker_recovery_timeout_seconds,
            )
        return _registry[name]
