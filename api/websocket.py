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
    except Exception as exc:
        await websocket.close(code=4001, reason=str(exc))
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

    session_id = str(uuid.uuid4())
    _sessions[session_id] = websocket
    active_websocket_gauge.inc()
    logger.info("WS session %s opened for user %s", session_id, user_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, "error", {"message": "Invalid JSON"})
                continue

            query = msg.get("query", "").strip()
            if not query:
                await _send(websocket, "error", {"message": "Empty query"})
                continue

            # Stream progress events while pipeline runs in executor
            await _send(websocket, "planning", {"query": query})

            loop = asyncio.get_event_loop()
            state: dict = {}

            def _run() -> dict:
                return run_pipeline(query, user_id=user_id)

            try:
                # Pipeline runs in thread pool so it doesn't block the event loop
                await _send(websocket, "retrieving", {})
                state = await loop.run_in_executor(None, _run)
                await _send(websocket, "evaluating", {})

                await _send(websocket, "done", {
                    "answer": state.get("answer", ""),
                    "critic_scores": state.get("critic_scores"),
                    "critic_passed": state.get("critic_passed"),
                    "retrieved_doc_count": len(state.get("retrieved_docs", [])),
                })
            except Exception as exc:
                logger.exception("WS pipeline error: %s", exc)
                await _send(websocket, "error", {"message": str(exc)})

    except WebSocketDisconnect:
        logger.info("WS session %s disconnected", session_id)
    finally:
        _sessions.pop(session_id, None)
        active_websocket_gauge.dec()
