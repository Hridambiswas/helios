# storage/crud.py — Helios reusable database CRUD helpers
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
from sqlalchemy import select, desc, update, delete

from storage.database import get_session
from storage.models import QueryRecord, Document


# ── QueryRecord ───────────────────────────────────────────────────────────────

async def get_query(query_id: str) -> QueryRecord | None:
    async with get_session() as session:
        result = await session.execute(
            select(QueryRecord).where(QueryRecord.id == query_id)
        )
        return result.scalar_one_or_none()


async def update_query_status(query_id: str, status: str) -> None:
    async with get_session() as session:
        await session.execute(
            update(QueryRecord)
            .where(QueryRecord.id == query_id)
            .values(status=status)
        )


async def list_user_queries(
    user_id: str, limit: int = 20, offset: int = 0
) -> list[QueryRecord]:
    async with get_session() as session:
        result = await session.execute(
            select(QueryRecord)
            .where(QueryRecord.user_id == user_id)
            .order_by(desc(QueryRecord.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())


# ── Document ──────────────────────────────────────────────────────────────────

async def get_document(doc_id: str) -> Document | None:
    async with get_session() as session:
        result = await session.execute(
            select(Document).where(Document.id == doc_id)
        )
        return result.scalar_one_or_none()


async def list_documents(user_id: str | None = None, limit: int = 50) -> list[Document]:
    async with get_session() as session:
        q = select(Document).order_by(desc(Document.created_at)).limit(limit)
        if user_id:
            q = q.where(Document.uploaded_by == user_id)
        result = await session.execute(q)
        return list(result.scalars().all())


async def mark_document_indexed(doc_id: str) -> None:
    async with get_session() as session:
        await session.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(indexed=True)
        )


async def delete_user_data(user_id: str) -> tuple[int, int]:
    """Delete all QueryRecords and Documents owned by user_id.

    Returns (queries_deleted, documents_deleted) for audit logging.
    """
    async with get_session() as session:
        q_result = await session.execute(
            delete(QueryRecord).where(QueryRecord.user_id == user_id)
        )
        d_result = await session.execute(
            delete(Document).where(Document.uploaded_by == user_id)
        )
        return q_result.rowcount, d_result.rowcount
