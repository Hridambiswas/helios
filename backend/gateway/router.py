# gateway/router.py — Helios API Gateway Middleware
# Author: Hridam Biswas | Project: Helios
#
# Responsibilities:
#   Canary routing — X-Canary: true header or random canary_percentage of
#                    traffic is marked canary; downstream pipeline can read
#                    request.state.is_canary to enable experimental behaviour.
#   A/B testing    — Each request is assigned variant "A" or "B", sticky by
#                    client IP so the same user consistently gets the same
#                    variant within a session.
#   Response headers — X-Canary and X-AB-Variant are added to every response
#                      for client-side observability.

from __future__ import annotations
import hashlib
import logging
import random

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("helios.gateway.router")

_EXCLUDED = frozenset({
    "/api/v1/health", "/api/v1/version", "/metrics",
    "/docs", "/redoc", "/openapi.json",
    "/api/v1/auth/register", "/api/v1/auth/login", "/api/v1/auth/refresh",
})


class GatewayMiddleware(BaseHTTPMiddleware):
    """
    API Gateway: canary routing + A/B variant assignment.

    request.state attributes set for downstream handlers:
      is_canary  (bool) — True if this request runs the canary pipeline
      ab_variant (str)  — "A" or "B"
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _EXCLUDED:
            request.state.is_canary = False
            request.state.ab_variant = "A"
            return await call_next(request)

        from config import cfg

        # ── Canary routing ────────────────────────────────────────────────────
        explicit = request.headers.get("X-Canary", "").lower() == "true"
        rolled = random.randint(1, 100) <= cfg.canary_percentage
        is_canary: bool = explicit or rolled
        request.state.is_canary = is_canary

        # ── A/B variant assignment (IP-hashed for consistency) ────────────────
        ab_header = request.headers.get("X-AB-Test", "").upper()
        if ab_header in ("A", "B"):
            variant = ab_header
        else:
            client_ip = request.client.host if request.client else "0.0.0.0"
            # Deterministic: same IP always gets same variant
            digest = int(hashlib.md5(client_ip.encode(), usedforsecurity=False).hexdigest(), 16)
            variant = "A" if digest % 2 == 0 else "B"
        request.state.ab_variant = variant

        if is_canary:
            logger.debug(
                "Canary request path=%s ab=%s", request.url.path, variant
            )

        response = await call_next(request)
        response.headers["X-Canary"] = "true" if is_canary else "false"
        response.headers["X-AB-Variant"] = variant
        return response
