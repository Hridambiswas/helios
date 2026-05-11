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
        c.messages = []
        return c

    @staticmethod
    def _set_auth(client, mock_user):
        from api.auth import get_current_user
        client.app.dependency_overrides[get_current_user] = lambda: mock_user
        return get_current_user

    @staticmethod
    def _clear_auth(client):
        client.app.dependency_overrides.clear()

    def test_list_conversations_requires_auth(self, client):
        resp = client.get("/api/v1/conversations")
        assert resp.status_code == 401

    def test_list_conversations_returns_empty_list(self, client):
        dep_key = self._set_auth(client, self._mock_user())
        try:
            with patch("storage.database.get_session_factory") as mock_sf:
                mock_session = AsyncMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock(return_value=False)
                mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
                mock_sf.return_value.return_value = mock_session
                resp = client.get("/api/v1/conversations", headers={"Authorization": "Bearer tok"})
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)
        finally:
            self._clear_auth(client)

    def test_create_conversation_requires_auth(self, client):
        resp = client.post("/api/v1/conversations", json={"title": "Test"})
        assert resp.status_code == 401

    def test_delete_conversation_requires_auth(self, client):
        resp = client.delete("/api/v1/conversations/any-id")
        assert resp.status_code == 401

    def test_append_message_requires_auth(self, client):
        resp = client.post("/api/v1/conversations/any-id/messages",
                           json={"role": "user", "content": "Hi"})
        assert resp.status_code == 401

    def test_create_conversation_title_too_long_rejected(self, client):
        dep_key = self._set_auth(client, self._mock_user())
        try:
            resp = client.post(
                "/api/v1/conversations",
                json={"title": "x" * 256},
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 422
        finally:
            self._clear_auth(client)


class TestDocumentChunkRoutes:
    """Tests for /documents/{id}/chunks and /documents/{id}/search."""

    def test_get_chunks_requires_auth(self, client):
        resp = client.get("/api/v1/documents/doc-1/chunks")
        assert resp.status_code == 401

    def test_search_requires_auth(self, client):
        resp = client.post("/api/v1/documents/doc-1/search", json={"query": "test"})
        assert resp.status_code == 401

    def test_search_document_empty_query_rejected(self, client):
        from api.auth import get_current_user
        mock_user = MagicMock(id="user-1", username="hridam", is_active=True)
        client.app.dependency_overrides[get_current_user] = lambda: mock_user
        try:
            resp = client.post(
                "/api/v1/documents/doc-1/search",
                json={"query": ""},
                headers={"Authorization": "Bearer tok"},
            )
            assert resp.status_code == 422
        finally:
            client.app.dependency_overrides.clear()
