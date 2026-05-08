# storage/read_replica.py — Helios CQRS Read-Side Session Factory
# Author: Hridam Biswas | Project: Helios
#
# Provides a read-only async session that routes to a PostgreSQL read
# replica when POSTGRES_READ_URL is configured, and falls back to the
# primary otherwise.  Use get_read_session() for all SELECT-only
# operations (query history, document listing, health checks) to keep
# write load off the primary.

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

from config import cfg

logger = logging.getLogger("helios.storage.read_replica")

_read_engine: AsyncEngine | None = None
_read_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_read_engine() -> AsyncEngine:
    global _read_engine
    if _read_engine is None:
        url = cfg.postgres_read_url or cfg.database_url
        source = "read-replica" if cfg.postgres_read_url else "primary (no replica configured)"
        _read_engine = create_async_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
        logger.info("Read session factory using %s", source)
    return _read_engine


def _get_read_session_factory() -> async_sessionmaker[AsyncSession]:
    global _read_session_factory
    if _read_session_factory is None:
        _read_session_factory = async_sessionmaker(
            _get_read_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _read_session_factory


@asynccontextmanager
async def get_read_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager yielding a read-only session.
    Never commits — callers must not issue writes through this session.
    """
    factory = _get_read_session_factory()
    async with factory() as session:
        yield session


async def close_read_engine() -> None:
    global _read_engine
    if _read_engine:
        await _read_engine.dispose()
        _read_engine = None
        logger.info("Read replica engine disposed")
