# workers/beat_tasks.py — Helios Celery Beat periodic tasks
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging

from workers.celery_app import app

logger = logging.getLogger("helios.workers.beat_tasks")


@app.task(name="workers.beat_tasks.expire_refresh_tokens")
def expire_refresh_tokens() -> dict:
    """
    Celery beat task: clean up expired and revoked refresh tokens from Postgres.
    Runs via asyncio.run since SQLAlchemy sessions are async.
    """
    import asyncio
    import time
    from datetime import datetime, timezone
    from sqlalchemy import delete
    from storage.database import get_session
    from storage.models import RefreshToken

    async def _cleanup():
        now = datetime.now(timezone.utc)
        async with get_session() as session:
            result = await session.execute(
                delete(RefreshToken).where(
                    (RefreshToken.expires_at < now) | (RefreshToken.revoked == True)  # noqa: E712
                )
            )
            return result.rowcount  # type: ignore[union-attr]

    t0 = time.perf_counter()
    deleted = asyncio.run(_cleanup())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Expired %d refresh tokens in %.0f ms", deleted, elapsed_ms)
    return {"deleted_tokens": deleted, "elapsed_ms": round(elapsed_ms, 1)}


@app.task(name="workers.beat_tasks.bm25_index_stats")
def bm25_index_stats() -> dict:
    """Periodic task: log BM25 index size for monitoring."""
    from retrieval.bm25_search import get_index
    idx = get_index()
    n = len(idx)
    logger.info("BM25 index size: %d documents", n)
    return {"bm25_index_size": n}
