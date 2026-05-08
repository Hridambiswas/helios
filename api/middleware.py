# api/middleware.py — Helios rate limiting and request ID middleware
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import uuid

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from storage.cache import incr

logger = logging.getLogger("helios.api.middleware")

# Rate limit: max requests per window per IP
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60

_EXCLUDED = frozenset({
    "/api/v1/health", "/metrics", "/docs", "/redoc", "/openapi.json",
    "/api/v1/auth/register", "/api/v1/auth/login", "/api/v1/auth/refresh",
})


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

    @staticmethod
    def _extract_rate_key(request: Request) -> str:
        """
        Return user:{user_id} if a valid Bearer token is present,
        otherwise ip:{client_ip} as a fallback.
        """
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            try:
                from jose import jwt as jose_jwt
                from config import cfg
                payload = jose_jwt.decode(
                    token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm]
                )
                return f"user:{payload['sub']}"
            except Exception:
                pass
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _EXCLUDED:
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
