# resilience/retry.py — Helios Exponential-Backoff Retry Decorator
# Author: Hridam Biswas | Project: Helios
#
# Thin wrapper over tenacity (already in requirements.txt).
# Use on any sync or async function that calls an external service
# and may experience transient failures (network blips, rate limits).

from __future__ import annotations
from typing import Type

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

_logger = logging.getLogger("helios.resilience.retry")


def with_exponential_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: tuple[Type[BaseException], ...] = (Exception,),
):
    """
    Decorator that retries the wrapped function with jittered exponential backoff.

    Args:
        max_attempts: Total attempts including the first call.
        min_wait: Minimum seconds to wait between retries.
        max_wait: Maximum seconds to wait between retries.
        exceptions: Only retry on these exception types.

    Example::

        @with_exponential_backoff(max_attempts=3, exceptions=(OpenAIError,))
        async def call_openai(...):
            ...
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1.5, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
