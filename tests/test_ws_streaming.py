# tests/test_ws_streaming.py — WebSocket streaming token tests
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


def _valid_token(user_id: str = "u1") -> str:
    from jose import jwt
    from config import cfg
    from datetime import datetime, timedelta, timezone
    return jwt.encode(
        {"sub": user_id, "type": "access",
         "exp": datetime.now(timezone.utc) + timedelta(minutes=60)},
        cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm,
    )


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


class TestWebSocketTokenStreaming:
    """Verify that the WS endpoint forwards token events and synthesizing step."""

    def _pipeline_with_tokens(self, tokens: list[str]):
        """Return a run_pipeline mock that fires _token_callback for each token."""
        def _run(query, user_id, conversation_history=None, token_callback=None, **kw):
            if token_callback:
                for tok in tokens:
                    token_callback(tok)
            return {
                "answer": "".join(tokens),
                "critic_scores": None,
                "critic_passed": True,
                "retrieved_docs": [],
                "retry_count": 0,
            }
        return _run

    def test_token_events_emitted_per_chunk(self, client):
        tokens = ["Hello", " ", "world"]
        mock_user = MagicMock(id="u1", is_active=True)
        with (
            patch("api.websocket.get_current_user_from_token_str",
                  new_callable=AsyncMock, return_value=mock_user),
            patch("api.websocket.run_pipeline", side_effect=self._pipeline_with_tokens(tokens)),
        ):
            with client.websocket_connect(f"/ws/query?token={_valid_token()}") as ws:
                ws.send_json({"query": "hi"})
                events = []
                while True:
                    msg = ws.receive_json()
                    events.append(msg["event"])
                    if msg["event"] in ("done", "error"):
                        break
                token_events = [e for e in events if e == "token"]
                assert len(token_events) == len(tokens)

    def test_synthesizing_event_emitted_before_tokens(self, client):
        tokens = ["Answer"]
        mock_user = MagicMock(id="u1", is_active=True)
        with (
            patch("api.websocket.get_current_user_from_token_str",
                  new_callable=AsyncMock, return_value=mock_user),
            patch("api.websocket.run_pipeline", side_effect=self._pipeline_with_tokens(tokens)),
        ):
            with client.websocket_connect(f"/ws/query?token={_valid_token()}") as ws:
                ws.send_json({"query": "test"})
                events = []
                while True:
                    msg = ws.receive_json()
                    events.append(msg["event"])
                    if msg["event"] in ("done", "error"):
                        break
                assert "synthesizing" in events
                # synthesizing must come before first token
                synth_idx = events.index("synthesizing")
                first_token_idx = next((i for i, e in enumerate(events) if e == "token"), len(events))
                assert synth_idx < first_token_idx

    def test_history_forwarded_to_pipeline(self, client):
        captured = {}
        mock_user = MagicMock(id="u1", is_active=True)

        def _run(query, user_id, conversation_history=None, token_callback=None, **kw):
            captured["history"] = conversation_history
            return {
                "answer": "ok",
                "critic_scores": None,
                "critic_passed": True,
                "retrieved_docs": [],
                "retry_count": 0,
            }

        with (
            patch("api.websocket.get_current_user_from_token_str",
                  new_callable=AsyncMock, return_value=mock_user),
            patch("api.websocket.run_pipeline", side_effect=_run),
        ):
            history = [{"role": "user", "content": "prev question"}]
            with client.websocket_connect(f"/ws/query?token={_valid_token()}") as ws:
                ws.send_json({"query": "follow-up", "history": history})
                while True:
                    msg = ws.receive_json()
                    if msg["event"] in ("done", "error"):
                        break
            assert captured.get("history") is not None
            assert len(captured["history"]) == 1
            assert captured["history"][0]["role"] == "user"

    def test_no_token_events_when_pipeline_does_not_stream(self, client):
        mock_user = MagicMock(id="u1", is_active=True)

        def _run(query, user_id, conversation_history=None, token_callback=None, **kw):
            # deliberately never calls token_callback
            return {
                "answer": "plain answer",
                "critic_scores": None,
                "critic_passed": True,
                "retrieved_docs": [],
                "retry_count": 0,
            }

        with (
            patch("api.websocket.get_current_user_from_token_str",
                  new_callable=AsyncMock, return_value=mock_user),
            patch("api.websocket.run_pipeline", side_effect=_run),
        ):
            with client.websocket_connect(f"/ws/query?token={_valid_token()}") as ws:
                ws.send_json({"query": "silent"})
                events = []
                while True:
                    msg = ws.receive_json()
                    events.append(msg["event"])
                    if msg["event"] in ("done", "error"):
                        break
                assert "token" not in events
