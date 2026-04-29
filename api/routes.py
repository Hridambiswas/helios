# api/routes.py — Helios FastAPI REST routes
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select, desc

from api.auth import (
    CurrentUser, get_user_by_username, create_user,
    verify_password, issue_tokens, refresh_access_token,
)
from api.schemas import (
    RegisterRequest, TokenResponse, RefreshRequest,
    UserResponse, QueryRequest, QueryResponse, IngestResponse,
    QueryHistoryItem, HealthResponse,
)
from storage.database import get_session, ping as db_ping
from storage.cache import ping as redis_ping
from storage.object_store import ping as minio_ping, upload, ensure_bucket
from storage.models import QueryRecord, Document
from storage.crud import list_user_queries, mark_document_indexed
from retrieval.vector_store import ping as chroma_ping, upsert_batch
from pipeline.run import run_pipeline
from workers.tasks import run_pipeline_task

logger = logging.getLogger("helios.api.routes")

router = APIRouter()


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
    user = await get_user_by_username(form.username)
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    return await issue_tokens(user)


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest):
    return await refresh_access_token(body.refresh_token)


@router.get("/auth/me", response_model=UserResponse)
async def me(current_user: CurrentUser):
    return current_user


# ── Query ─────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
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
        raise HTTPException(500, detail=state["error"])

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


@router.post("/query/async", status_code=202)
async def query_async(body: QueryRequest, current_user: CurrentUser):
    """Dispatch pipeline to Celery worker pool; return task ID immediately."""
    task = run_pipeline_task.delay(body.query, user_id=current_user.id)
    return {"task_id": task.id, "status": "queued"}


@router.get("/query/history", response_model=list[QueryHistoryItem])
async def query_history(
    current_user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    records = await list_user_queries(current_user.id, limit=limit, offset=offset)
    return records


@router.get("/query/{query_id}", response_model=QueryHistoryItem)
async def get_query_detail(query_id: str, current_user: CurrentUser):
    from storage.crud import get_query
    record = await get_query(query_id)
    if not record or record.user_id != current_user.id:
        raise HTTPException(404, "Query not found")
    return record


# ── Document ingest ───────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(file: Annotated[UploadFile, File()], current_user: CurrentUser):
    ensure_bucket()
    doc_id = str(uuid.uuid4())
    minio_key = f"docs/{doc_id}/{file.filename}"

    content = await file.read()
    upload(minio_key, content, content_type=file.content_type or "application/octet-stream")

    text = content.decode("utf-8", errors="replace")
    raw_chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
    chunks = []
    for chunk in raw_chunks:
        for i in range(0, len(chunk), 500):
            chunks.append(chunk[i: i + 500])

    from langchain_openai import OpenAIEmbeddings
    from config import cfg
    import retrieval.bm25_search as bm25

    embedder = OpenAIEmbeddings(model=cfg.openai_embedding_model, api_key=cfg.openai_api_key)
    chunk_ids = [f"{doc_id}::chunk::{i}" for i in range(len(chunks))]
    embeddings = embedder.embed_documents(chunks)
    metas = [{"doc_id": doc_id, "filename": file.filename, "chunk_idx": i} for i in range(len(chunks))]

    upsert_batch(chunk_ids, embeddings, chunks, metas)
    bm25.get_index().add_batch(chunk_ids, chunks, metas)

    async with get_session() as session:
        session.add(Document(
            id=doc_id, filename=file.filename,
            content_type=file.content_type or "text/plain",
            minio_key=minio_key, chunk_count=len(chunks),
            size_bytes=len(content), indexed=True,
            uploaded_by=current_user.id,
        ))

    await mark_document_indexed(doc_id)
    logger.info("Ingested doc %s: %d chunks", doc_id, len(chunks))
    return IngestResponse(
        document_id=doc_id, filename=file.filename,
        chunk_count=len(chunks), size_bytes=len(content), indexed=True,
    )


@router.get("/documents", response_model=list[dict])
async def list_documents(current_user: CurrentUser, limit: int = Query(50, ge=1, le=200)):
    from storage.crud import list_documents as _list
    docs = await _list(user_id=current_user.id, limit=limit)
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
