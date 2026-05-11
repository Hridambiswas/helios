# api/routes.py — Helios FastAPI REST routes
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import asyncio
import logging
import pathlib
import re
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload

from api.auth import (
    CurrentUser, OptionalUser, get_user_by_username, create_user,
    verify_password, issue_tokens, refresh_access_token,
    validate_password_strength,
    get_user_by_id,  # noqa: F401 — re-exported so tests can patch api.routes.get_user_by_id
)
from celery.result import AsyncResult
from api.schemas import (
    RegisterRequest, TokenResponse, RefreshRequest, LogoutRequest,
    UserResponse, QueryRequest, QueryResponse, IngestResponse,
    QueryHistoryItem, HealthResponse, TaskStatusResponse, WebSource,
    ConversationOut, ConversationDetailOut, CreateConversationRequest,
    AppendMessageRequest, ConversationMessageOut,
    DocumentChunksResponse, ChunkPreview, TestRetrievalResponse, TestRetrievalResult,
)
from storage.database import get_session, ping as db_ping
from storage.models import QueryRecord, Document, RefreshToken, Conversation, ConversationMessage
from storage.read_replica import get_read_session
from storage.cache import ping as redis_ping
from storage.object_store import ping as minio_ping, upload, ensure_bucket, delete as minio_delete
from storage.crud import mark_document_indexed
from retrieval.vector_store import ping as chroma_ping, upsert_batch, delete_batch as chroma_delete_batch
from pipeline.run import run_pipeline
from workers.tasks import run_pipeline_task
from resilience.backpressure import check_backpressure, active_pipeline, BackpressureError
from resilience.saga import Saga, SagaExecutionError

logger = logging.getLogger("helios.api.routes")

router = APIRouter()


def _safe_error(internal: str, public: str = "Internal server error") -> str:
    """Return detailed message in dev, generic message in production."""
    from config import cfg
    return internal if cfg.is_development else public


# ── Shared dependencies ───────────────────────────────────────────────────────

async def _backpressure_guard() -> None:
    """FastAPI dependency — rejects new work when the system is overloaded."""
    try:
        await check_backpressure()
    except BackpressureError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── Auth ──────────────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest):
    pw_errors = validate_password_strength(body.password)
    if pw_errors:
        raise HTTPException(400, f"Password must contain: {', '.join(pw_errors)}")
    existing = await get_user_by_username(body.username)
    if existing:
        raise HTTPException(400, "Username already taken")
    user = await create_user(body.username, body.email, body.password)
    return await issue_tokens(user)


@router.post("/auth/login", response_model=TokenResponse)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    from observability.metrics import auth_failure_counter
    user = await get_user_by_username(form.username)
    if not user or not user.hashed_password or not verify_password(form.password, user.hashed_password):
        auth_failure_counter.labels(reason="bad_credentials").inc()
        raise HTTPException(401, "Invalid credentials")
    return await issue_tokens(user)


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest):
    return await refresh_access_token(body.refresh_token)


@router.post("/auth/logout", status_code=204)
async def logout(body: LogoutRequest, current_user: CurrentUser):
    """Revoke the supplied refresh token so it can no longer be exchanged."""
    import hashlib
    from sqlalchemy import update
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    async with get_session() as session:
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.token_hash == token_hash, RefreshToken.user_id == current_user.id)
            .values(revoked=True)
        )
    logger.info("User %s revoked refresh token", current_user.id)


@router.get("/auth/me", response_model=UserResponse)
async def me(current_user: CurrentUser):
    return current_user


@router.get("/auth/me/stats")
async def my_stats(current_user: CurrentUser):
    """Return per-user query and document counts."""
    from sqlalchemy import func
    async with get_read_session() as session:
        q_total = await session.scalar(
            select(func.count(QueryRecord.id)).where(QueryRecord.user_id == current_user.id)
        ) or 0
        q_done = await session.scalar(
            select(func.count(QueryRecord.id)).where(
                QueryRecord.user_id == current_user.id,
                QueryRecord.status == "done",
            )
        ) or 0
        doc_total = await session.scalar(
            select(func.count(Document.id)).where(Document.uploaded_by == current_user.id)
        ) or 0
    return {
        "total_queries": q_total,
        "successful_queries": q_done,
        "total_documents": doc_total,
    }


# ── Query (write path) ────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(_backpressure_guard)],
)
async def query(body: QueryRequest, current_user: OptionalUser):
    """Run the full Helios pipeline. Auth is optional — first query is free for guests."""
    query_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    if current_user:
        async with get_session() as session:
            record = QueryRecord(
                id=query_id,
                user_id=current_user.id,
                query_text=body.query,
                status="running",
            )
            session.add(record)

    # Check cache for identical recent queries (TTL 5 min)
    from storage.cache import get as cache_get, set as cache_set
    import hashlib as _hl
    cache_key = _hl.sha256(body.query.strip().lower().encode()).hexdigest()
    cached = await cache_get("query_result", cache_key)
    if cached and not current_user:
        # Return cached result for unauthenticated duplicate queries
        return QueryResponse(**cached)

    history = [{"role": m.role, "content": m.content} for m in (body.history or [])]
    async with active_pipeline():
        state = await asyncio.to_thread(
            run_pipeline, body.query,
            user_id=current_user.id if current_user else None,
            conversation_history=history,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    status_str = "done" if not state.get("error") else "failed"

    if current_user:
        async with get_session() as session:
            result = await session.execute(select(QueryRecord).where(QueryRecord.id == query_id))
            rec = result.scalar_one_or_none()
            if rec:
                rec.answer = state.get("answer")
                rec.plan = state.get("plan")
                rec.retrieved_docs = [
                    {"id": d["id"], "score": d.get("score", 0)}
                    for d in state.get("retrieved_docs", [])
                ]
                rec.critic_scores = state.get("critic_scores")
                rec.latency_ms = elapsed_ms
                rec.status = status_str

    if state.get("error"):
        raise HTTPException(
            500,
            detail=_safe_error(state["error"], "Pipeline execution failed"),
        )

    docs = [
        {"id": d["id"], "document": d["document"], "metadata": d.get("metadata", {}),
         "score": d.get("score", 0.0), "source": d.get("source", "unknown")}
        for d in state.get("retrieved_docs", [])
    ]

    web_sources = [
        WebSource(
            title=w.get("title", ""),
            url=w.get("url", ""),
            snippet=w.get("snippet", ""),
        )
        for w in state.get("web_sources", [])
        if w.get("url")
    ]

    response = QueryResponse(
        query_id=query_id, query=body.query,
        answer=state.get("answer", ""),
        plan=state.get("plan"),
        retrieved_docs=docs,  # type: ignore[arg-type]
        web_sources=web_sources,
        execution_result=state.get("execution_result"),
        critic_scores=state.get("critic_scores"),
        critic_passed=state.get("critic_passed"),
        follow_up_questions=state.get("follow_up_questions", []),
        latency_ms=round(elapsed_ms, 1),
        status=status_str,
    )

    if not current_user and status_str == "done":
        await cache_set("query_result", cache_key, response.model_dump(mode="json"), ttl=300)

    return response


@router.post(
    "/query/async",
    status_code=202,
    dependencies=[Depends(_backpressure_guard)],
)
async def query_async(body: QueryRequest, current_user: CurrentUser):
    """Dispatch pipeline to Celery worker pool; return task ID immediately."""
    query_id = str(uuid.uuid4())
    async with get_session() as session:
        record = QueryRecord(
            id=query_id,
            user_id=current_user.id,
            query_text=body.query,
            status="queued",
        )
        session.add(record)

    task = run_pipeline_task.delay(body.query, user_id=current_user.id, query_id=query_id)  # type: ignore[attr-defined]
    return {"task_id": task.id, "query_id": query_id, "status": "queued"}


@router.get("/query/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, current_user: CurrentUser):
    """Poll the Celery task state for an async pipeline request."""
    from workers.celery_app import app as celery_app

    result = AsyncResult(task_id, app=celery_app)
    celery_state = result.state  # PENDING / STARTED / SUCCESS / FAILURE / RETRY

    _state_map = {
        "PENDING": "queued",
        "STARTED": "running",
        "RETRY": "running",
        "SUCCESS": "done",
        "FAILURE": "failed",
    }
    status = _state_map.get(celery_state, "unknown")

    payload: dict | None = None
    if celery_state == "SUCCESS" and isinstance(result.result, dict):
        payload = result.result

    query_id_header = result.info.get("query_id") if isinstance(result.info, dict) else None

    return TaskStatusResponse(
        task_id=task_id,
        query_id=query_id_header,
        celery_state=celery_state,
        status=status,
        result=payload,
    )


# ── Query (read path — CQRS: routed to read replica) ─────────────────────────

@router.get("/query/history", response_model=list[QueryHistoryItem])
async def query_history(
    current_user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    async with get_read_session() as session:
        result = await session.execute(
            select(QueryRecord)
            .where(QueryRecord.user_id == current_user.id)
            .order_by(desc(QueryRecord.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())


@router.get("/query/{query_id}", response_model=QueryHistoryItem)
async def get_query_detail(query_id: str, current_user: CurrentUser):
    async with get_read_session() as session:
        result = await session.execute(
            select(QueryRecord).where(QueryRecord.id == query_id)
        )
        record = result.scalar_one_or_none()
    if not record or record.user_id != current_user.id:
        raise HTTPException(404, "Query not found")
    return record


@router.get("/query/history/export")
async def export_query_history(current_user: CurrentUser):
    """Export the current user's full query history as a CSV download."""
    import csv
    import io
    from fastapi.responses import StreamingResponse

    async with get_read_session() as session:
        result = await session.execute(
            select(QueryRecord)
            .where(QueryRecord.user_id == current_user.id)
            .order_by(desc(QueryRecord.created_at))
            .limit(10_000)
        )
        records = list(result.scalars().all())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "query", "status", "latency_ms", "critic_passed", "created_at"])
    for r in records:
        critic_passed = ""
        if r.critic_scores:
            critic_passed = str(r.critic_scores.get("overall", ""))
        writer.writerow([
            r.id,
            r.query_text,
            r.status,
            r.latency_ms or "",
            critic_passed,
            r.created_at.isoformat(),
        ])

    buf.seek(0)
    filename = f"helios_history_{current_user.username}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Document ingest (Saga pattern) ────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(file: Annotated[UploadFile, File()], current_user: CurrentUser):
    """
    Ingest a document using the Saga pattern.
    Steps: MinIO upload → ChromaDB embed+index → BM25 index → PostgreSQL record.
    Any failure triggers compensating transactions in reverse order.
    """
    from config import cfg

    # ── Filename sanitization (path traversal prevention) ─────────────────────
    raw_name = pathlib.Path(file.filename or "upload").name
    safe_name = re.sub(r"[^\w\-.]", "_", raw_name)[:255] or "upload"

    # ── Extension whitelist ───────────────────────────────────────────────────
    ext = pathlib.Path(safe_name).suffix.lower()
    if ext not in cfg.allowed_extensions_list:
        raise HTTPException(
            400,
            detail=f"File type '{ext}' not allowed. Permitted: {cfg.allowed_extensions_list}",
        )

    ensure_bucket()
    doc_id = str(uuid.uuid4())
    minio_key = f"docs/{doc_id}/{safe_name}"
    content = await file.read()

    # ── Size limit ────────────────────────────────────────────────────────────
    if len(content) > cfg.max_upload_bytes:
        raise HTTPException(
            413,
            detail=f"File too large — max {cfg.max_upload_bytes // 1_048_576} MB",
        )

    # ── Prepare chunks + embeddings (before the saga so failures here are cheap) ─
    text = content.decode("utf-8", errors="replace")
    raw_chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
    chunks: list[str] = []
    chunk_size, overlap = 500, 50
    for para in raw_chunks:
        start = 0
        while start < len(para):
            chunks.append(para[start: start + chunk_size])
            if start + chunk_size >= len(para):
                break
            start += chunk_size - overlap

    from langchain_community.embeddings import FastEmbedEmbeddings
    import retrieval.bm25_search as bm25

    embedder = FastEmbedEmbeddings(model_name=cfg.embedding_model)
    chunk_ids = [f"{doc_id}::chunk::{i}" for i in range(len(chunks))]
    embeddings = embedder.embed_documents(chunks)
    metas = [{"doc_id": doc_id, "filename": safe_name, "chunk_idx": i} for i in range(len(chunks))]

    # ── Compensators ──────────────────────────────────────────────────────────
    def _delete_minio() -> None:
        try:
            minio_delete(minio_key)
        except Exception:
            pass

    def _delete_chroma() -> None:
        try:
            chroma_delete_batch(chunk_ids)
        except Exception:
            pass

    async def _insert_db() -> None:
        async with get_session() as session:
            session.add(Document(
                id=doc_id, filename=safe_name,
                content_type=file.content_type or "text/plain",
                minio_key=minio_key, chunk_count=len(chunks),
                size_bytes=len(content), indexed=True,
                uploaded_by=current_user.id,
            ))

    async def _delete_db() -> None:
        from sqlalchemy import delete as sa_delete
        async with get_session() as session:
            await session.execute(sa_delete(Document).where(Document.id == doc_id))

    # ── Execute saga ──────────────────────────────────────────────────────────
    saga = (
        Saga("document-ingest")
        .step(
            "upload-minio",
            action=lambda: upload(minio_key, content, content_type=file.content_type or "application/octet-stream"),
            compensate=_delete_minio,
        )
        .step(
            "embed-index-chroma",
            action=lambda: upsert_batch(chunk_ids, embeddings, chunks, metas),
            compensate=_delete_chroma,
        )
        .step(
            "index-bm25",
            action=lambda: bm25.get_index().add_batch(chunk_ids, chunks, metas),
            compensate=lambda: None,  # BM25 is in-memory; no persistent compensation needed
        )
        .step(
            "persist-db",
            action=_insert_db,
            compensate=_delete_db,
        )
    )

    try:
        await saga.execute()
    except SagaExecutionError as exc:
        logger.error("Ingest saga failed at step '%s': %s", exc.step, exc.cause)
        raise HTTPException(
            500,
            detail=_safe_error(
                f"Ingest failed at '{exc.step}': {exc.cause}",
                "Document ingest failed — please try again",
            ),
        )

    await mark_document_indexed(doc_id)
    logger.info("Ingested doc %s: %d chunks", doc_id, len(chunks))
    return IngestResponse(
        document_id=doc_id, filename=safe_name,
        chunk_count=len(chunks), size_bytes=len(content), indexed=True,
    )


# ── Documents (read path — CQRS) ──────────────────────────────────────────────

@router.get("/documents", response_model=list[dict])
async def list_documents(
    current_user: CurrentUser,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    async with get_read_session() as session:
        result = await session.execute(
            select(Document)
            .where(Document.uploaded_by == current_user.id)
            .order_by(desc(Document.created_at))
            .limit(limit)
            .offset(offset)
        )
        docs = list(result.scalars().all())
    return [
        {"id": d.id, "filename": d.filename, "chunk_count": d.chunk_count,
         "size_bytes": d.size_bytes, "indexed": d.indexed,
         "created_at": d.created_at.isoformat()}
        for d in docs
    ]


@router.get("/documents/{doc_id}", response_model=dict)
async def get_document(doc_id: str, current_user: CurrentUser):
    async with get_read_session() as session:
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
    if not doc or doc.uploaded_by != current_user.id:
        raise HTTPException(404, "Document not found")
    return {
        "id": doc.id, "filename": doc.filename, "chunk_count": doc.chunk_count,
        "size_bytes": doc.size_bytes, "indexed": doc.indexed,
        "content_type": doc.content_type, "minio_key": doc.minio_key,
        "created_at": doc.created_at.isoformat(),
    }


@router.get("/documents/{doc_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
    doc_id: str,
    current_user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Return a preview of the stored text chunks for a document."""
    async with get_read_session() as session:
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
    if not doc or doc.uploaded_by != current_user.id:
        raise HTTPException(404, "Document not found")

    from retrieval.vector_store import get_chunks_for_doc
    raw_chunks = await asyncio.to_thread(get_chunks_for_doc, doc_id, limit=limit, offset=offset)
    chunks = [
        ChunkPreview(chunk_index=i + offset, text=c.get("text", "")[:500], char_count=len(c.get("text", "")))
        for i, c in enumerate(raw_chunks)
    ]
    return DocumentChunksResponse(
        document_id=doc_id, filename=doc.filename,
        total_chunks=doc.chunk_count, chunks=chunks,
    )


@router.post("/documents/{doc_id}/search", response_model=TestRetrievalResponse)
async def test_document_retrieval(
    doc_id: str,
    body: QueryRequest,
    current_user: CurrentUser,
):
    """Run a test retrieval query scoped to a single document."""
    async with get_read_session() as session:
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
    if not doc or doc.uploaded_by != current_user.id:
        raise HTTPException(404, "Document not found")

    from retrieval.vector_store import query_by_doc
    raw_results = await asyncio.to_thread(query_by_doc, body.query, doc_id, top_k=5)
    results = [
        TestRetrievalResult(
            chunk_index=int(r.get("metadata", {}).get("chunk_idx", i)),
            text=r.get("document", "")[:500],
            score=round(r.get("score", 0.0), 4),
            source=r.get("source", "dense"),
        )
        for r in raw_results
    ]
    return TestRetrievalResponse(query=body.query, results=results)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, current_user: CurrentUser):
    """Delete document from MinIO, ChromaDB, and PostgreSQL."""
    from sqlalchemy import delete as sa_delete
    async with get_read_session() as session:
        result = await session.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
    if not doc or doc.uploaded_by != current_user.id:
        raise HTTPException(404, "Document not found")

    chunk_count = doc.chunk_count or 0
    chunk_ids = [f"{doc_id}::chunk::{i}" for i in range(chunk_count)]

    try:
        minio_delete(doc.minio_key)
    except Exception:
        pass
    try:
        if chunk_ids:
            chroma_delete_batch(chunk_ids)
    except Exception:
        pass

    async with get_session() as session:
        await session.execute(sa_delete(Document).where(Document.id == doc_id))
    logger.info("Deleted document %s (%s)", doc_id, doc.filename)


# ── Conversations ─────────────────────────────────────────────────────────────

@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    current_user: CurrentUser,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    async with get_read_session() as session:
        result = await session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.user_id == current_user.id)
            .order_by(desc(Conversation.updated_at))
            .limit(limit).offset(offset)
        )
        convs = list(result.scalars().all())
        return [
            ConversationOut(
                id=c.id, title=c.title,
                created_at=c.created_at, updated_at=c.updated_at,
                message_count=len(c.messages),
            )
            for c in convs
        ]


@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def create_conversation(body: CreateConversationRequest, current_user: CurrentUser):
    conv = Conversation(user_id=current_user.id, title=body.title)
    async with get_session() as session:
        session.add(conv)
        await session.flush()
        conv_id, title, created_at, updated_at = conv.id, conv.title, conv.created_at, conv.updated_at
    return ConversationOut(id=conv_id, title=title, created_at=created_at, updated_at=updated_at)


@router.get("/conversations/{conv_id}", response_model=ConversationDetailOut)
async def get_conversation(conv_id: str, current_user: CurrentUser):
    async with get_read_session() as session:
        result = await session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conv_id)
        )
        conv = result.scalar_one_or_none()
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(404, "Conversation not found")
        msgs = [ConversationMessageOut(id=m.id, role=m.role, content=m.content, created_at=m.created_at)
                for m in conv.messages]
        return ConversationDetailOut(
            id=conv.id, title=conv.title,
            created_at=conv.created_at, updated_at=conv.updated_at,
            message_count=len(msgs), messages=msgs,
        )


@router.post("/conversations/{conv_id}/messages", response_model=ConversationMessageOut, status_code=201)
async def append_message(conv_id: str, body: AppendMessageRequest, current_user: CurrentUser):
    async with get_read_session() as session:
        result = await session.execute(select(Conversation).where(Conversation.id == conv_id))
        conv = result.scalar_one_or_none()
    if not conv or conv.user_id != current_user.id:
        raise HTTPException(404, "Conversation not found")
    msg = ConversationMessage(conversation_id=conv_id, role=body.role, content=body.content)
    async with get_session() as session:
        session.add(msg)
        # Update conversation title from first user message
        if body.role == "user" and conv.title == "New Chat":
            result2 = await session.execute(select(Conversation).where(Conversation.id == conv_id))
            c = result2.scalar_one_or_none()
            if c:
                c.title = body.content[:55]
        await session.flush()
        msg_id, created_at = msg.id, msg.created_at
    return ConversationMessageOut(id=msg_id, role=body.role, content=body.content, created_at=created_at)


@router.delete("/conversations/{conv_id}", status_code=204)
async def delete_conversation(conv_id: str, current_user: CurrentUser):
    from sqlalchemy import delete as sa_delete
    async with get_read_session() as session:
        result = await session.execute(select(Conversation).where(Conversation.id == conv_id))
        conv = result.scalar_one_or_none()
    if not conv or conv.user_id != current_user.id:
        raise HTTPException(404, "Conversation not found")
    async with get_session() as session:
        await session.execute(sa_delete(Conversation).where(Conversation.id == conv_id))
    logger.info("Deleted conversation %s for user %s", conv_id, current_user.id)


# ── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def stats():
    """Public aggregate stats (no auth required)."""
    from sqlalchemy import func
    async with get_read_session() as session:
        query_count = await session.scalar(select(func.count(QueryRecord.id))) or 0
        doc_count = await session.scalar(select(func.count(Document.id))) or 0
        user_count = await session.scalar(
            select(func.count()).select_from(__import__('storage.models', fromlist=['User']).User)
        ) or 0
    return {
        "total_queries": query_count,
        "total_documents": doc_count,
        "total_users": user_count,
    }


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    pg, rd, mn, ch = await asyncio.gather(
        db_ping(),
        redis_ping(),
        asyncio.to_thread(minio_ping),
        asyncio.to_thread(chroma_ping),
        return_exceptions=True,
    )
    pg = pg is True
    rd = rd is True
    mn = mn is True
    ch = ch is True
    overall = "ok" if all([pg, rd, mn, ch]) else ("degraded" if any([pg, rd]) else "down")
    return HealthResponse(status=overall, postgres=pg, redis=rd, minio=mn, chroma=ch)


@router.get("/version")
async def version():
    from config import cfg
    return {"version": "1.0.0", "environment": cfg.app_env}


@router.get("/health/circuits")
async def circuit_breaker_status(current_user: CurrentUser):
    """Return state of all registered circuit breakers (authenticated users only)."""
    from resilience.circuit_breaker import list_breakers
    return {"circuits": list_breakers()}
