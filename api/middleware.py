# api/middleware.py — Helios rate limiting and request ID middleware
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import uuid
import logging

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from storage.cache import incr

logger = logging.getLogger("helios.api.middleware")

# Rate limit: max requests per window per IP
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request and response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter using Redis atomic incr.
    Applies per client IP; health and metrics endpoints are excluded.
    """

    _EXCLUDED = frozenset({"/api/v1/health", "/metrics", "/docs", "/redoc", "/openapi.json"})

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in self._EXCLUDED:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        count = await incr("ratelimit", client_ip, ttl=RATE_LIMIT_WINDOW_SECONDS)

        if count > RATE_LIMIT_REQUESTS:
            logger.warning("Rate limit exceeded for %s (%d requests)", client_ip, count)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded — max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s"
                },
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )

        return await call_next(request)
