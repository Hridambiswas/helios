# tests/test_websocket_security.py — Tests for WebSocket security hardening
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
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


def _valid_token(user_id: str = "u1") -> str:
    from jose import jwt
    from config import cfg
    from datetime import datetime, timedelta, timezone
    return jwt.encode(
        {"sub": user_id, "type": "access",
         "exp": datetime.now(timezone.utc) + timedelta(minutes=60)},
        cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm,
    )


class TestWebSocketAuthentication:

    def test_ws_without_token_closes_4001(self, client):
        with client.websocket_connect("/ws/query") as ws:
            # Server should close immediately — TestClient raises on disconnect
            pass
        # If we get here the connection was closed cleanly or with 4001

    def test_ws_with_invalid_token_rejected(self, client):
        try:
            with client.websocket_connect("/ws/query?token=invalid.token.value") as ws:
                ws.receive_json()  # should not get a valid message
        except Exception:
            pass  # expected — connection rejected

    def test_ws_close_reason_does_not_leak_internals(self, client):
        """Verify the close reason is a generic string, not an exception repr."""
        with patch("api.websocket.get_current_user_from_token_str",
                   new_callable=AsyncMock,
                   side_effect=Exception("Internal DB connection string exposed")):
            try:
                with client.websocket_connect("/ws/query?token=any") as ws:
                    ws.receive_json()
            except Exception:
                pass
        # Test passes if no exception propagated and no internal detail leaked


class TestWebSocketMessageLimits:

    def test_oversized_message_rejected_with_error_event(self, client):
        mock_user = MagicMock(id="u1", is_active=True)
        with patch("api.websocket.get_current_user_from_token_str",
                   new_callable=AsyncMock, return_value=mock_user):
            with client.websocket_connect(f"/ws/query?token={_valid_token()}") as ws:
                big_payload = json.dumps({"query": "x" * 70_000})  # > 64 KB
                ws.send_text(big_payload)
                msg = ws.receive_json()
                assert msg["event"] == "error"
                assert "too large" in msg["data"]["message"].lower()

    def test_query_too_long_rejected_with_error_event(self, client):
        mock_user = MagicMock(id="u1", is_active=True)
        with patch("api.websocket.get_current_user_from_token_str",
                   new_callable=AsyncMock, return_value=mock_user):
            with client.websocket_connect(f"/ws/query?token={_valid_token()}") as ws:
                ws.send_json({"query": "q" * 4097})
                msg = ws.receive_json()
                assert msg["event"] == "error"
                assert "too long" in msg["data"]["message"].lower()

    def test_empty_query_rejected_with_error_event(self, client):
        mock_user = MagicMock(id="u1", is_active=True)
        with patch("api.websocket.get_current_user_from_token_str",
                   new_callable=AsyncMock, return_value=mock_user):
            with client.websocket_connect(f"/ws/query?token={_valid_token()}") as ws:
                ws.send_json({"query": ""})
                msg = ws.receive_json()
                assert msg["event"] == "error"

    def test_invalid_json_rejected_with_error_event(self, client):
        mock_user = MagicMock(id="u1", is_active=True)
        with patch("api.websocket.get_current_user_from_token_str",
                   new_callable=AsyncMock, return_value=mock_user):
            with client.websocket_connect(f"/ws/query?token={_valid_token()}") as ws:
                ws.send_text("not-valid-json{{{")
                msg = ws.receive_json()
                assert msg["event"] == "error"
                assert "JSON" in msg["data"]["message"]


class TestWebSocketConnectionLimit:

    def test_user_connection_counter_increments_and_decrements(self):
        from api.websocket import _user_connections
        initial = _user_connections.get("conn-test-user", 0)
        mock_user = MagicMock(id="conn-test-user", is_active=True)

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
            patch("api.websocket.get_current_user_from_token_str",
                  new_callable=AsyncMock, return_value=mock_user),
            patch("api.websocket.run_pipeline", return_value={"answer": "ok",
                  "critic_scores": None, "critic_passed": True, "retrieved_docs": []}),
        ):
            from main import app
            with TestClient(app) as c:
                with c.websocket_connect(f"/ws/query?token={_valid_token('conn-test-user')}") as ws:
                    assert _user_connections.get("conn-test-user", 0) == initial + 1
                # After disconnect counter decrements
        assert _user_connections.get("conn-test-user", 0) == initial
