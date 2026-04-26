# graph/checkpointing.py — Helios LangGraph state checkpointing via Redis
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
import logging
from typing import Any

from storage.cache import set as cache_set, get as cache_get, TTL_LONG

logger = logging.getLogger("helios.graph.checkpointing")

_NS = "checkpoint"


async def save_checkpoint(session_id: str, agent_name: str, state: dict[str, Any]) -> None:
    """
    Persist agent output state after each hop.
    Key: checkpoint:<session_id>:<agent_name>
    TTL: TTL_LONG (1h) — long enough to support retries and debugging.
    """
    # Exclude non-serialisable items from state before caching
    serialisable = {
        k: v for k, v in state.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    key = f"{session_id}:{agent_name}"
    await cache_set(_NS, key, serialisable, ttl=TTL_LONG)
    logger.debug("Checkpoint saved: %s/%s (%d keys)", session_id, agent_name, len(serialisable))


async def load_checkpoint(session_id: str, agent_name: str) -> dict[str, Any] | None:
    """Load a previously saved agent checkpoint, or None if not found."""
    key = f"{session_id}:{agent_name}"
    data = await cache_get(_NS, key)
    if data is not None:
        logger.debug("Checkpoint loaded: %s/%s", session_id, agent_name)
    return data


async def list_checkpoints(session_id: str) -> list[str]:
    """Return the agent names for which checkpoints exist in this session."""
    agents = ["planner", "retriever", "executor", "synthesizer", "critic"]
    found = []
    for agent in agents:
        data = await load_checkpoint(session_id, agent)
        if data is not None:
            found.append(agent)
    return found
