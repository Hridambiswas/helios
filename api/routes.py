# api/routes.py — Helios FastAPI REST routes
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import pathlib
import re
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select, desc

from api.auth import (
    CurrentUser, get_user_by_username, create_user,
    verify_password, issue_tokens, refresh_access_token,
)
from api.schemas import (
    RegisterRequest, TokenResponse, RefreshRequest, LogoutRequest,
    UserResponse, QueryRequest, QueryResponse, IngestResponse,
    QueryHistoryItem, HealthResponse,
)
from storage.database import get_session, ping as db_ping
from storage.models import QueryRecord, Document, RefreshToken
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
    existing = await get_user_by_username(body.username)
    if existing:
        raise HTTPException(400, "Username already taken")
    user = await create_user(body.username, body.email, body.password)
    return await issue_tokens(user)


@router.post("/auth/login", response_model=TokenResponse)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    from observability.metrics import auth_failure_counter
    user = await get_user_by_username(form.username)
    if not user or not verify_password(form.password, user.hashed_password):
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


@router.get("/auth/me", response_model=UserResponse)
async def me(current_user: CurrentUser):
    return current_user


# ── Query (write path) ────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(_backpressure_guard)],
)
async def query(body: QueryRequest, current_user: CurrentUser):
    """Run the full Helios pipeline synchronously."""
    query_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    async with get_session() as session:
        record = QueryRecord(
            id=query_id,
            user_id=current_user.id,
            query_text=body.query,
            status="running",
        )
        session.add(record)

    async with active_pipeline():
        state = run_pipeline(body.query, user_id=current_user.id)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    status_str = "done" if not state.get("error") else "failed"

    async with get_session() as session:
        result = await session.execute(select(QueryRecord).where(QueryRecord.id == query_id))
        rec = result.scalar_one_or_none()
        if rec is None:
            raise HTTPException(500, detail="Query record lost — storage error")
        rec.answer = state.get("answer")
        rec.plan = state.get("plan")
        rec.retrieved_docs = [
            {"id": d["id"], "score": d.get("score", 0)} for d in state.get("retrieved_docs", [])
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

    return QueryResponse(
        query_id=query_id, query=body.query,
        answer=state.get("answer", ""),
        plan=state.get("plan"),
        retrieved_docs=docs,
        execution_result=state.get("execution_result"),
        critic_scores=state.get("critic_scores"),
        critic_passed=state.get("critic_passed"),
        latency_ms=round(elapsed_ms, 1),
        status=status_str,
    )


@router.post(
    "/query/async",
    status_code=202,
    dependencies=[Depends(_backpressure_guard)],
)
async def query_async(body: QueryRequest, current_user: CurrentUser):
    """Dispatch pipeline to Celery worker pool; return task ID immediately."""
    task = run_pipeline_task.delay(body.query, user_id=current_user.id)
    return {"task_id": task.id, "status": "queued"}


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
    if ext not in cfg.allowed_upload_extensions:
        raise HTTPException(
            400,
            detail=f"File type '{ext}' not allowed. Permitted: {cfg.allowed_upload_extensions}",
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
    for chunk in raw_chunks:
        for i in range(0, len(chunk), 500):
            chunks.append(chunk[i: i + 500])

    from langchain_openai import OpenAIEmbeddings
    import retrieval.bm25_search as bm25

    embedder = OpenAIEmbeddings(model=cfg.openai_embedding_model, api_key=cfg.openai_api_key)
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
async def list_documents(current_user: CurrentUser, limit: int = Query(50, ge=1, le=200)):
    async with get_read_session() as session:
        result = await session.execute(
            select(Document)
            .where(Document.uploaded_by == current_user.id)
            .order_by(desc(Document.created_at))
            .limit(limit)
        )
        docs = list(result.scalars().all())
    return [
        {"id": d.id, "filename": d.filename, "chunk_count": d.chunk_count,
         "size_bytes": d.size_bytes, "indexed": d.indexed,
         "created_at": d.created_at.isoformat()}
        for d in docs
    ]


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    pg = await db_ping()
    rd = await redis_ping()
    mn = minio_ping()
    ch = chroma_ping()
    overall = "ok" if all([pg, rd, mn, ch]) else ("degraded" if any([pg, rd]) else "down")
    return HealthResponse(status=overall, postgres=pg, redis=rd, minio=mn, chroma=ch)
