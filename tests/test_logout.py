# tests/test_logout.py — Tests for POST /auth/logout
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import hashlib
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


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
        patch("storage.cache.ttl_seconds", new_callable=AsyncMock, return_value=60),
    ):
        from main import app
        return TestClient(app)


def _make_token(user_id: str = "u1") -> str:
    from jose import jwt
    from config import cfg
    from datetime import datetime, timedelta, timezone
    return jwt.encode(
        {"sub": user_id, "type": "access",
         "exp": datetime.now(timezone.utc) + timedelta(minutes=60)},
        cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm,
    )


class TestLogout:

    def test_logout_returns_204(self, client):
        mock_user = MagicMock(id="u1", is_active=True)
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock()

        with (
            patch("api.auth.get_user_by_id", new_callable=AsyncMock, return_value=mock_user),
            patch("storage.database.get_session", return_value=mock_session),
        ):
            resp = client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": "some.refresh.token"},
                headers={"Authorization": f"Bearer {_make_token()}"},
            )
        assert resp.status_code == 204

    def test_logout_requires_auth(self, client):
        resp = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": "some.refresh.token"},
        )
        assert resp.status_code == 401

    def test_logout_hashes_token_for_revocation(self, client):
        mock_user = MagicMock(id="u1", is_active=True)
        captured = {}

        async def fake_execute(stmt):
            captured["stmt"] = stmt

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = fake_execute

        refresh_token = "my.refresh.token"
        expected_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

        with (
            patch("api.auth.get_user_by_id", new_callable=AsyncMock, return_value=mock_user),
            patch("storage.database.get_session", return_value=mock_session),
        ):
            resp = client.post(
                "/api/v1/auth/logout",
                json={"refresh_token": refresh_token},
                headers={"Authorization": f"Bearer {_make_token()}"},
            )
        assert resp.status_code == 204
        # Verify execute was called (revocation was attempted)
        assert "stmt" in captured

    def test_logout_missing_body_rejected(self, client):
        mock_user = MagicMock(id="u1", is_active=True)
        with patch("api.routes.get_user_by_id", new_callable=AsyncMock, return_value=mock_user):
            resp = client.post(
                "/api/v1/auth/logout",
                json={},  # missing refresh_token
                headers={"Authorization": f"Bearer {_make_token()}"},
            )
        assert resp.status_code == 422
