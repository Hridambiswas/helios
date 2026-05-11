# api/websocket.py — Helios WebSocket streaming endpoint
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.auth import get_current_user_from_token_str
from pipeline.run import run_pipeline
from observability.metrics import active_websocket_gauge

logger = logging.getLogger("helios.api.websocket")

ws_router = APIRouter()

# Active sessions: session_id → WebSocket
_sessions: dict[str, WebSocket] = {}
# Per-user connection counts
_user_connections: dict[str, int] = {}


async def _authenticate_ws(websocket: WebSocket) -> str | None:
    """
    Validate JWT from query-param ?token=... on WebSocket upgrade.
    Returns user_id or None on failure.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return None
    try:
        user = await get_current_user_from_token_str(token)
        return user.id
    except Exception:
        await websocket.close(code=4001, reason="Authentication failed")
        return None


async def _send(ws: WebSocket, event: str, data: Any) -> None:
    try:
        await ws.send_text(json.dumps({"event": event, "data": data}))
    except Exception as exc:
        logger.warning("WebSocket send failed: %s", exc)


@ws_router.websocket("/ws/query")
async def ws_query(websocket: WebSocket):
    """
    WebSocket query endpoint.

    Protocol:
      Client → {"query": "..."} JSON message
      Server ← {"event": "planning",    "data": {...}}
      Server ← {"event": "retrieving",  "data": {}}
      Server ← {"event": "executing",   "data": {}}
      Server ← {"event": "synthesizing","data": {}}
      Server ← {"event": "evaluating",  "data": {}}
      Server ← {"event": "done",        "data": {answer, critic_scores, ...}}
      Server ← {"event": "error",       "data": {"message": "..."}}
    """
    await websocket.accept()
    user_id = await _authenticate_ws(websocket)
    if not user_id:
        return

    from config import cfg
    current_conns = _user_connections.get(user_id, 0)
    if current_conns >= cfg.ws_max_connections_per_user:
        await websocket.close(code=4029, reason="Too many open connections")
        logger.warning("WS connection limit reached for user %s (%d)", user_id, current_conns)
        return

    session_id = str(uuid.uuid4())
    _sessions[session_id] = websocket
    _user_connections[user_id] = current_conns + 1
    active_websocket_gauge.inc()
    logger.info("WS session %s opened for user %s", session_id, user_id)

    try:
        while True:
            raw = await websocket.receive_text()

            from config import cfg
            if len(raw.encode()) > cfg.ws_max_message_bytes:
                await _send(websocket, "error", {
                    "message": f"Message too large — max {cfg.ws_max_message_bytes // 1024} KB"
                })
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, "error", {"message": "Invalid JSON"})
                continue

            query = msg.get("query", "").strip()
            if not query:
                await _send(websocket, "error", {"message": "Empty query"})
                continue

            if len(query) > 4096:
                await _send(websocket, "error", {"message": "Query too long — max 4096 characters"})
                continue

            raw_history = msg.get("history", [])
            history = [
                {"role": str(h.get("role", "")), "content": str(h.get("content", ""))[:4096]}
                for h in raw_history[:20]
                if h.get("role") in ("user", "assistant") and h.get("content")
            ]

            await _send(websocket, "planning", {"query": query})

            loop = asyncio.get_running_loop()
            token_queue: asyncio.Queue[str | None] = asyncio.Queue()

            import threading
            _first_token_lock = threading.Lock()
            _first_token_fired = [False]

            def _token_cb(token: str) -> None:
                with _first_token_lock:
                    if not _first_token_fired[0]:
                        _first_token_fired[0] = True
                        loop.call_soon_threadsafe(
                            lambda: asyncio.ensure_future(_send(websocket, "synthesizing", {}))
                        )
                loop.call_soon_threadsafe(token_queue.put_nowait, token)

            state: dict = {}

            async def _drain_tokens() -> None:
                while True:
                    tok = await token_queue.get()
                    if tok is None:
                        break
                    await _send(websocket, "token", {"token": tok})

            def _run() -> dict:
                result = run_pipeline(
                    query, user_id=user_id,
                    conversation_history=history,
                    token_callback=_token_cb,
                )
                if result.get("retry_count", 0) > 1:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(
                            _send(websocket, "retrying", {"attempt": result.get("retry_count", 1)})
                        )
                    )
                loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel
                return result

            try:
                await _send(websocket, "retrieving", {})
                pipeline_task = loop.run_in_executor(None, _run)
                drain_task = asyncio.ensure_future(_drain_tokens())
                state = await pipeline_task
                await drain_task
                await _send(websocket, "evaluating", {})

                if state.get("error"):
                    from config import cfg
                    err_msg = state["error"] if cfg.is_development else "Pipeline error — please try again"
                    await _send(websocket, "error", {"message": err_msg})
                else:
                    await _send(websocket, "done", {
                        "answer": state.get("answer", ""),
                        "critic_scores": state.get("critic_scores"),
                        "critic_passed": state.get("critic_passed"),
                        "retrieved_doc_count": len(state.get("retrieved_docs", [])),
                    })
            except Exception as exc:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)
                logger.exception("WS pipeline error: %s", exc)
                from config import cfg
                msg = str(exc) if cfg.is_development else "Pipeline error — please try again"
                await _send(websocket, "error", {"message": msg})

    except WebSocketDisconnect:
        logger.info("WS session %s disconnected", session_id)
    finally:
        _sessions.pop(session_id, None)
        if user_id in _user_connections:
            _user_connections[user_id] = max(0, _user_connections[user_id] - 1)
        active_websocket_gauge.dec()
