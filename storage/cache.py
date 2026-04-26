# storage/cache.py — Helios Redis cache layer
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from config import cfg

logger = logging.getLogger("helios.storage.cache")

_redis: aioredis.Redis | None = None

# Default TTLs (seconds)
TTL_SHORT = 60          # ephemeral — live streaming state
TTL_MEDIUM = 300        # 5 min — query in-progress
TTL_LONG = 3600         # 1 hr — retrieved docs, embeddings


def _client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            cfg.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )
    return _redis


def _key(namespace: str, key: str) -> str:
    return f"helios:{namespace}:{key}"


async def set(namespace: str, key: str, value: Any, ttl: int = TTL_MEDIUM) -> None:
    """Serialise value to JSON and store with TTL."""
    try:
        await _client().setex(_key(namespace, key), ttl, json.dumps(value))
    except Exception as exc:
        logger.warning("Cache set failed [%s:%s]: %s", namespace, key, exc)


async def get(namespace: str, key: str) -> Any | None:
    """Return parsed JSON value or None on miss / error."""
    try:
        raw = await _client().get(_key(namespace, key))
        return json.loads(raw) if raw is not None else None
    except Exception as exc:
        logger.warning("Cache get failed [%s:%s]: %s", namespace, key, exc)
        return None


async def delete(namespace: str, key: str) -> None:
    try:
        await _client().delete(_key(namespace, key))
    except Exception as exc:
        logger.warning("Cache delete failed [%s:%s]: %s", namespace, key, exc)


async def exists(namespace: str, key: str) -> bool:
    try:
        return bool(await _client().exists(_key(namespace, key)))
    except Exception:
        return False


async def incr(namespace: str, key: str, ttl: int = TTL_MEDIUM) -> int:
    """Atomic increment — used for rate limiting counters."""
    full_key = _key(namespace, key)
    client = _client()
    count = await client.incr(full_key)
    if count == 1:
        await client.expire(full_key, ttl)
    return count


async def ping() -> bool:
    try:
        return await _client().ping()
    except Exception as exc:
        logger.error("Redis ping failed: %s", exc)
        return False


async def close() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
        logger.info("Redis connection closed")
