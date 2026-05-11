# storage/database.py — Helios PostgreSQL async engine
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from config import cfg

logger = logging.getLogger("helios.storage.database")

# ── ORM base ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Engine & session factory ──────────────────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _effective_database_url() -> tuple[str, dict]:
    """Return (url, connect_args) for the active database backend."""
    if cfg.supabase_database_url:
        url = cfg.supabase_database_url
        if url.startswith("postgresql://") and "+asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url, {"ssl": True}
    return cfg.database_url, {}


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url, connect_args = _effective_database_url()
        _engine = create_async_engine(
            url,
            pool_size=5,               # Supabase free tier: 60 connection limit
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_timeout=30,
            echo=cfg.is_development,
            connect_args=connect_args,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a transactional session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Create all ORM-mapped tables (idempotent — safe to call on startup)."""
    from storage import models  # noqa: F401 — register models before create_all
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured")




async def wait_for_db(retries: int = 5, delay: float = 2.0) -> None:
    """Block until the database is reachable, with exponential backoff."""
    import asyncio
    for attempt in range(1, retries + 1):
        if await ping():
            return
        if attempt < retries:
            wait = delay * (2 ** (attempt - 1))
            logger.warning("DB not ready (attempt %d/%d), retrying in %.0fs", attempt, retries, wait)
            await asyncio.sleep(wait)
    raise RuntimeError("Database unreachable after %d attempts" % retries)

async def ping() -> bool:
    """Health check — returns True if Postgres is reachable."""
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database ping failed: %s", exc)
        return False


def connection_info() -> dict:
    """Return non-sensitive connection metadata for health/debug endpoints."""
    url, _ = _effective_database_url()
    engine = _engine
    pool_status: dict = {}
    if engine is not None:
        pool = engine.pool
        pool_status = {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }
    # Mask credentials from URL for safe logging
    from urllib.parse import urlparse
    parsed = urlparse(url)
    safe_url = f"{parsed.scheme}://***@{parsed.hostname}:{parsed.port}{parsed.path}"
    backend = "supabase" if cfg.supabase_database_url else "local"
    return {"backend": backend, "url": safe_url, "pool": pool_status}


async def close_engine() -> None:
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine disposed")
