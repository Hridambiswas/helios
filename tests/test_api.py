# tests/test_api.py — FastAPI route integration tests (no live services needed)
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture()
def client():
    """Create a test client with all external dependencies mocked."""
    with (
        patch("storage.database.get_engine"),
        patch("storage.database.create_tables", new_callable=AsyncMock),
        patch("storage.object_store.ensure_bucket"),
        patch("observability.tracing.setup_tracing"),
        patch("observability.logging_config.setup_logging"),
        patch("storage.database.close_engine", new_callable=AsyncMock),
        patch("storage.cache.incr", new_callable=AsyncMock, return_value=1),
    ):
        from main import app
        return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        with (
            patch("api.routes.db_ping", new_callable=AsyncMock, return_value=True),
            patch("api.routes.redis_ping", new_callable=AsyncMock, return_value=True),
            patch("api.routes.minio_ping", return_value=True),
            patch("api.routes.chroma_ping", return_value=True),
        ):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["postgres"] is True

    def test_health_degraded_when_redis_down(self, client):
        with (
            patch("api.routes.db_ping", new_callable=AsyncMock, return_value=True),
            patch("api.routes.redis_ping", new_callable=AsyncMock, return_value=False),
            patch("api.routes.minio_ping", return_value=True),
            patch("api.routes.chroma_ping", return_value=True),
        ):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "degraded"


class TestAuthEndpoints:
    def test_register_returns_tokens(self, client):
        mock_user = MagicMock(id="u1", username="hridam", email="h@test.com",
                              is_active=True, created_at=None)
        with (
            patch("api.routes.get_user_by_username", new_callable=AsyncMock, return_value=None),
            patch("api.routes.create_user", new_callable=AsyncMock, return_value=mock_user),
            patch("api.routes.issue_tokens", new_callable=AsyncMock, return_value=MagicMock(
                access_token="a", refresh_token="r", token_type="bearer", expires_in=3600,
                model_dump=lambda: {"access_token": "a", "refresh_token": "r",
                                    "token_type": "bearer", "expires_in": 3600}
            )),
        ):
            resp = client.post("/api/v1/auth/register", json={
                "username": "hridam", "email": "h@test.com", "password": "Password123"
            })
            assert resp.status_code == 201

    def test_register_duplicate_username(self, client):
        mock_user = MagicMock()
        with patch("api.routes.get_user_by_username", new_callable=AsyncMock, return_value=mock_user):
            resp = client.post("/api/v1/auth/register", json={
                "username": "hridam", "email": "h@test.com", "password": "Password123"
            })
            assert resp.status_code == 400


class TestConversationRoutes:
    """Tests for GET/POST /conversations endpoints."""

    def _mock_user(self):
        return MagicMock(id="user-1", username="hridam", is_active=True)

    def _mock_conv(self, conv_id="conv-1", title="Test Chat"):
        c = MagicMock()
        c.id = conv_id
        c.title = title
        c.created_at = None
        c.updated_at = None
        c.message_count = 0
        return c

    def test_list_conversations_requires_auth(self, client):
        resp = client.get("/api/v1/conversations")
        assert resp.status_code == 401

    def test_list_conversations_returns_list(self, client):
        conv = self._mock_conv()
        with (
            patch("api.routes.get_current_user", new_callable=AsyncMock, return_value=self._mock_user()),
            patch("api.routes.list_conversations", new_callable=AsyncMock, return_value=[conv]),
        ):
            resp = client.get("/api/v1/conversations", headers={"Authorization": "Bearer tok"})
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)

    def test_create_conversation_returns_201(self, client):
        conv = self._mock_conv(title="My Chat")
        with (
            patch("api.routes.get_current_user", new_callable=AsyncMock, return_value=self._mock_user()),
            patch("api.routes.create_conversation", new_callable=AsyncMock, return_value=conv),
        ):
            resp = client.post(
                "/api/v1/conversations",
                json={"title": "My Chat"},
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 201

    def test_create_conversation_default_title(self, client):
        conv = self._mock_conv(title="New Chat")
        with (
            patch("api.routes.get_current_user", new_callable=AsyncMock, return_value=self._mock_user()),
            patch("api.routes.create_conversation", new_callable=AsyncMock, return_value=conv),
        ):
            resp = client.post(
                "/api/v1/conversations",
                json={},
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 201

    def test_get_conversation_detail(self, client):
        conv = self._mock_conv()
        conv.messages = []
        with (
            patch("api.routes.get_current_user", new_callable=AsyncMock, return_value=self._mock_user()),
            patch("api.routes.get_conversation", new_callable=AsyncMock, return_value=conv),
        ):
            resp = client.get(
                "/api/v1/conversations/conv-1",
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 200

    def test_get_conversation_not_found(self, client):
        with (
            patch("api.routes.get_current_user", new_callable=AsyncMock, return_value=self._mock_user()),
            patch("api.routes.get_conversation", new_callable=AsyncMock, return_value=None),
        ):
            resp = client.get(
                "/api/v1/conversations/missing",
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 404

    def test_append_message_to_conversation(self, client):
        msg = MagicMock()
        msg.id = "msg-1"
        msg.role = "user"
        msg.content = "Hello"
        msg.created_at = None
        with (
            patch("api.routes.get_current_user", new_callable=AsyncMock, return_value=self._mock_user()),
            patch("api.routes.get_conversation", new_callable=AsyncMock, return_value=self._mock_conv()),
            patch("api.routes.add_message", new_callable=AsyncMock, return_value=msg),
        ):
            resp = client.post(
                "/api/v1/conversations/conv-1/messages",
                json={"role": "user", "content": "Hello"},
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 201

    def test_delete_conversation(self, client):
        with (
            patch("api.routes.get_current_user", new_callable=AsyncMock, return_value=self._mock_user()),
            patch("api.routes.get_conversation", new_callable=AsyncMock, return_value=self._mock_conv()),
            patch("api.routes.delete_conversation", new_callable=AsyncMock),
        ):
            resp = client.delete(
                "/api/v1/conversations/conv-1",
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 204

    def test_delete_conversation_not_found(self, client):
        with (
            patch("api.routes.get_current_user", new_callable=AsyncMock, return_value=self._mock_user()),
            patch("api.routes.get_conversation", new_callable=AsyncMock, return_value=None),
        ):
            resp = client.delete(
                "/api/v1/conversations/missing",
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 404
