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
    Per-user sliding-window rate limiter backed by Redis atomic INCR.

    Authenticated routes key on the JWT subject (user_id) for a limit
    of RATE_LIMIT_PER_USER requests per window.  Unauthenticated
    requests (auth endpoints are excluded entirely) key on client IP
    with the same limit as a safety net.

    Auth, health, and doc endpoints are excluded from rate limiting.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _EXCLUDED:
            return await call_next(request)

        from config import cfg

        rate_key = self._extract_rate_key(request)
        count = await incr("ratelimit_user", rate_key, ttl=cfg.rate_limit_window_seconds)

        if count > cfg.rate_limit_per_user:
            logger.warning(
                "Rate limit exceeded: key=%s count=%d limit=%d",
                rate_key, count, cfg.rate_limit_per_user,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"Rate limit exceeded — max {cfg.rate_limit_per_user} "
                        f"requests per {cfg.rate_limit_window_seconds}s"
                    )
                },
                headers={"Retry-After": str(cfg.rate_limit_window_seconds)},
            )

        return await call_next(request)

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
