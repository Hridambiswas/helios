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
            deleted = result.rowcount
        logger.info("Expired %d refresh tokens", deleted)
        return deleted

    deleted = asyncio.run(_cleanup())
    return {"deleted_tokens": deleted}


@app.task(name="workers.beat_tasks.bm25_index_stats")
def bm25_index_stats() -> dict:
    """Periodic task: log BM25 index size for monitoring."""
    from retrieval.bm25_search import get_index
    idx = get_index()
    n = len(idx)
    logger.info("BM25 index size: %d documents", n)
    return {"bm25_index_size": n}
