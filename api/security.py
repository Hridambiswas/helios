# api/security.py — Helios security middleware
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from storage.cache import incr

logger = logging.getLogger("helios.api.security")

_AUTH_PATHS = frozenset({"/api/v1/auth/login", "/api/v1/auth/register"})
_BRUTE_MAX = 5
_BRUTE_WINDOW = 300  # 5 minutes


def extract_client_ip(request: Request) -> str:
    """Return real client IP, peeling back cfg.trusted_proxy_count X-Forwarded-For hops."""
    from config import cfg
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for and cfg.trusted_proxy_count > 0:
        hops = [h.strip() for h in forwarded_for.split(",")]
        idx = max(0, len(hops) - cfg.trusted_proxy_count)
        return hops[idx]
    return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach defensive HTTP security headers to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        from config import cfg
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        if cfg.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
        return response


class AuthBruteForceMiddleware(BaseHTTPMiddleware):
    """Block IPs that exceed 5 auth attempts in 5 minutes."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path not in _AUTH_PATHS:
            return await call_next(request)

        client_ip = extract_client_ip(request)
        count = await incr("brute", client_ip, ttl=_BRUTE_WINDOW)

        if count > _BRUTE_MAX:
            logger.warning("Brute-force detected: ip=%s count=%d", client_ip, count)
            from observability.metrics import brute_force_blocked_counter
            brute_force_blocked_counter.labels(path=request.url.path).inc()
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many login attempts — try again later"},
                headers={"Retry-After": str(_BRUTE_WINDOW)},
            )

        return await call_next(request)
