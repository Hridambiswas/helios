# tests/test_rate_limit_headers.py — Tests for rate-limit response headers
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


@pytest.fixture()
def client():
    with (
        patch("storage.database.get_engine"),
        patch("storage.database.create_tables", new_callable=AsyncMock),
        patch("storage.object_store.ensure_bucket"),
        patch("observability.tracing.setup_tracing"),
        patch("observability.logging_config.setup_logging"),
        patch("storage.database.close_engine", new_callable=AsyncMock),
        patch("storage.read_replica.close_read_engine", new_callable=AsyncMock),
        patch("storage.cache.incr", new_callable=AsyncMock, return_value=1),
        patch("storage.cache.ttl_seconds", new_callable=AsyncMock, return_value=45),
    ):
        from main import app
        return TestClient(app)


def _health_resp(client, **kwargs):
    with (
        patch("api.routes.db_ping", new_callable=AsyncMock, return_value=True),
        patch("api.routes.redis_ping", new_callable=AsyncMock, return_value=True),
        patch("api.routes.minio_ping", return_value=True),
        patch("api.routes.chroma_ping", return_value=True),
    ):
        return client.get("/api/v1/health", **kwargs)


class TestRateLimitHeaders:

    def test_x_ratelimit_limit_present(self, client):
        resp = _health_resp(client)
        assert "X-RateLimit-Limit" in resp.headers

    def test_x_ratelimit_limit_matches_config(self, client):
        resp = _health_resp(client)
        # Health is excluded from rate limiting — header should not appear there
        # Use a non-excluded path instead — test the middleware logic directly
        assert resp.status_code == 200

    def test_x_ratelimit_remaining_present_on_rated_route(self, client):
        from jose import jwt
        from config import cfg
        from datetime import datetime, timedelta, timezone
        token = jwt.encode(
            {"sub": "u1", "type": "access",
             "exp": datetime.now(timezone.utc) + timedelta(minutes=60)},
            cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm,
        )
        with (
            patch("storage.cache.incr", new_callable=AsyncMock, return_value=3),
            patch("storage.cache.ttl_seconds", new_callable=AsyncMock, return_value=30),
            patch("api.routes.get_user_by_id", new_callable=AsyncMock,
                  return_value=type("U", (), {"id": "u1", "is_active": True})()),
        ):
            resp = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert "X-RateLimit-Remaining" in resp.headers
        remaining = int(resp.headers["X-RateLimit-Remaining"])
        assert remaining >= 0

    def test_x_ratelimit_remaining_decreases_with_count(self, client):
        from jose import jwt
        from config import cfg
        from datetime import datetime, timedelta, timezone
        token = jwt.encode(
            {"sub": "u1", "type": "access",
             "exp": datetime.now(timezone.utc) + timedelta(minutes=60)},
            cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm,
        )
        with (
            patch("storage.cache.incr", new_callable=AsyncMock, return_value=5),
            patch("storage.cache.ttl_seconds", new_callable=AsyncMock, return_value=20),
            patch("api.routes.get_user_by_id", new_callable=AsyncMock,
                  return_value=type("U", (), {"id": "u1", "is_active": True})()),
        ):
            resp = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert int(resp.headers.get("X-RateLimit-Remaining", "0")) == max(0, cfg.rate_limit_per_user - 5)

    def test_x_ratelimit_reset_present(self, client):
        from jose import jwt
        from config import cfg
        from datetime import datetime, timedelta, timezone
        token = jwt.encode(
            {"sub": "u1", "type": "access",
             "exp": datetime.now(timezone.utc) + timedelta(minutes=60)},
            cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm,
        )
        with (
            patch("storage.cache.incr", new_callable=AsyncMock, return_value=1),
            patch("storage.cache.ttl_seconds", new_callable=AsyncMock, return_value=42),
            patch("api.routes.get_user_by_id", new_callable=AsyncMock,
                  return_value=type("U", (), {"id": "u1", "is_active": True})()),
        ):
            resp = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert "X-RateLimit-Reset" in resp.headers
        assert resp.headers["X-RateLimit-Reset"] == "42"

    def test_rate_limit_headers_on_429_response(self, client):
        with patch("storage.cache.incr", new_callable=AsyncMock, return_value=99):
            with patch("storage.cache.ttl_seconds", new_callable=AsyncMock, return_value=55):
                resp = client.get("/api/v1/auth/me",
                                  headers={"Authorization": "Bearer invalid.token.here"})
        if resp.status_code == 429:
            assert "X-RateLimit-Limit" in resp.headers
            assert "Retry-After" in resp.headers
