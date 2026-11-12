# agents/_groq_call.py — shared "8s + one retry" wrapper for Groq agents.
# Author: Hridam Biswas | Project: Helios
#
# Groq is normally fast (<2 s per call) but occasionally stalls. We want
# every debate-pipeline Groq call to be bounded: at most two attempts, each
# capped by a hard thread-based timeout, so one bad roll can't push the
# query past the 25 s budget.

from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

logger = logging.getLogger("helios.agents.groq_call")


def invoke_groq_with_retry(
    llm,
    messages,
    *,
    agent_name: str,
    per_call_timeout: float = 8.0,
    attempts: int = 2,
) -> str:
    """
    Call ``llm.invoke(messages)`` up to ``attempts`` times, each capped at
    ``per_call_timeout`` seconds. Returns the stripped text content of the
    first successful call. Raises if every attempt fails.

    The timeout uses ThreadPoolExecutor.future.result(timeout=…) because
    the Groq client's own timeout handling is unreliable under load.
    """
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        def _call() -> str:
            resp = llm.invoke(messages, timeout=per_call_timeout)
            return (
                resp.content if isinstance(resp.content, str) else str(resp.content)
            ).strip()

        # NOT using `with ThreadPoolExecutor` — its __exit__ calls
        # shutdown(wait=True), which would block on the still-running
        # retry-chain inside the SDK even after we've given up.
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{agent_name}-call")
        fut = pool.submit(_call)
        try:
            result = fut.result(timeout=per_call_timeout)
            pool.shutdown(wait=False)
            return result
        except FuturesTimeout:
            fut.cancel()
            pool.shutdown(wait=False)  # abandon the hung worker thread
            last_exc = TimeoutError(
                f"{agent_name} exceeded {per_call_timeout}s (attempt {attempt})"
            )
            logger.warning(
                "%s timed out at %.1fs on attempt %d/%d",
                agent_name, per_call_timeout, attempt, attempts,
            )
        except Exception as exc:  # pragma: no cover
            pool.shutdown(wait=False)
            last_exc = exc
            logger.warning(
                "%s raised %s on attempt %d/%d",
                agent_name, type(exc).__name__, attempt, attempts,
            )

    assert last_exc is not None
    raise last_exc
