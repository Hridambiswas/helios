# workers/tasks.py — Helios Celery task definitions
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging

from celery import Task

from workers.celery_app import app

logger = logging.getLogger("helios.workers.tasks")


class PipelineTask(Task):
    """Custom Task base that propagates OTEL trace context."""

    def apply_async(self, args=None, kwargs=None, **options):  # type: ignore[override]
        from observability.tracing import inject_celery_context
        headers = options.pop("headers", {}) or {}
        inject_celery_context(headers)
        return super().apply_async(args=args, kwargs=kwargs, headers=headers, **options)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Task %s failed: %s", task_id, exc)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning("Task %s retrying: %s", task_id, exc)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("Task %s completed successfully", task_id)


# ── Pipeline task ─────────────────────────────────────────────────────────────

@app.task(
    base=PipelineTask,
    bind=True,
    name="workers.tasks.run_pipeline_task",
    max_retries=3,
    default_retry_delay=10,
    soft_time_limit=120,
    time_limit=180,
)
def run_pipeline_task(
    self, query: str, user_id: str | None = None, query_id: str | None = None
) -> dict:
    """
    Celery task that executes the full Helios agent pipeline.
    Retries up to 3 times on transient failures.
    Persists result to QueryRecord when query_id is provided.
    """
    import asyncio
    import time
    from pipeline.run import run_pipeline
    from observability.tracing import extract_celery_context

    from celery.exceptions import SoftTimeLimitExceeded

    extract_celery_context(self.request.headers or {})
    logger.info("Task %s: pipeline start user=%s query=%.80s", self.request.id, user_id, query)

    t0 = time.perf_counter()
    try:
        state = run_pipeline(query, user_id=user_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if state.get("error"):
            raise RuntimeError(state["error"])

        if query_id:
            async def _save():
                from sqlalchemy import select
                from storage.database import get_session
                from storage.models import QueryRecord
                async with get_session() as session:
                    result = await session.execute(
                        select(QueryRecord).where(QueryRecord.id == query_id)
                    )
                    rec = result.scalar_one_or_none()
                    if rec:
                        rec.answer = state.get("answer")
                        rec.plan = state.get("plan")
                        rec.retrieved_docs = [
                            {"id": d["id"], "score": d.get("score", 0)}
                            for d in state.get("retrieved_docs", [])
                        ]
                        rec.critic_scores = state.get("critic_scores")
                        rec.latency_ms = round(elapsed_ms, 1)
                        rec.status = "done"
            asyncio.run(_save())

        return {
            "answer": state.get("answer"),
            "critic_passed": state.get("critic_passed"),
            "critic_scores": state.get("critic_scores"),
            "latency_ms": round(elapsed_ms, 1),
        }
    except SoftTimeLimitExceeded:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "Task %s hit soft time limit after %.0f ms — marking failed",
            self.request.id, elapsed_ms,
        )
        if query_id:
            async def _mark_timeout():
                from sqlalchemy import update
                from storage.database import get_session
                from storage.models import QueryRecord
                async with get_session() as session:
                    await session.execute(
                        update(QueryRecord)
                        .where(QueryRecord.id == query_id)
                        .values(status="timeout")
                    )
            try:
                asyncio.run(_mark_timeout())
            except Exception:
                pass
        raise
    except Exception as exc:
        if query_id:
            async def _mark_failed():
                from sqlalchemy import update
                from storage.database import get_session
                from storage.models import QueryRecord
                async with get_session() as session:
                    await session.execute(
                        update(QueryRecord)
                        .where(QueryRecord.id == query_id)
                        .values(status="failed")
                    )
            try:
                asyncio.run(_mark_failed())
            except Exception:
                pass
        logger.exception("Pipeline task %s failed: %s", self.request.id, exc)
        raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1))


# ── Ingest task ───────────────────────────────────────────────────────────────

@app.task(
    name="workers.tasks.ingest_document_task",
    max_retries=2,
    default_retry_delay=5,
    soft_time_limit=60,
)
def ingest_document_task(doc_id: str, minio_key: str, filename: str) -> dict:
    """
    Download document from MinIO, chunk, embed, and upsert into ChromaDB + BM25.
    Runs in Celery worker so HTTP requests don't block the FastAPI event loop.
    """
    import time as _time
    from storage.object_store import download
    from retrieval.vector_store import upsert_batch
    from retrieval.bm25_search import get_index
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from config import cfg
    from observability.metrics import (
        ingest_latency_histogram,
        ingest_chunk_count_histogram,
        ingest_bytes_histogram,
    )

    _t0 = _time.perf_counter()
    logger.info("Ingesting doc %s from MinIO key %s", doc_id, minio_key)
    content = download(minio_key)
    ingest_bytes_histogram.observe(len(content))
    text = content.decode("utf-8", errors="replace")

    chunk_size = cfg.ingest_chunk_size
    overlap = cfg.ingest_chunk_overlap
    max_chunks = cfg.ingest_max_chunks

    raw_chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
    if not raw_chunks:
        # Fallback: treat entire text as one paragraph if no double-newlines
        raw_chunks = [text.strip()]
    chunks: list[str] = []
    for para in raw_chunks:
        start = 0
        while start < len(para):
            chunks.append(para[start: start + chunk_size])
            if start + chunk_size >= len(para):
                break
            start += chunk_size - overlap
    if len(chunks) > max_chunks:
        logger.warning("Doc %s produced %d chunks; capping at %d", doc_id, len(chunks), max_chunks)
        chunks = chunks[:max_chunks]

    embedder = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    chunk_ids = [f"{doc_id}::chunk::{i}" for i in range(len(chunks))]
    embeddings = embedder.embed_documents(chunks)
    metas = [{"doc_id": doc_id, "filename": filename, "chunk_idx": i} for i in range(len(chunks))]

    upsert_batch(chunk_ids, embeddings, chunks, metas)
    get_index().add_batch(chunk_ids, chunks, metas)

    import asyncio
    from storage.crud import mark_document_indexed
    asyncio.run(mark_document_indexed(doc_id))

    elapsed_ms = (_time.perf_counter() - _t0) * 1000
    ingest_chunk_count_histogram.observe(len(chunks))
    ingest_latency_histogram.observe(elapsed_ms)
    logger.info("Ingested %d chunks for doc %s in %.0f ms", len(chunks), doc_id, elapsed_ms)
    return {"doc_id": doc_id, "chunk_count": len(chunks)}


# ── Health check task ─────────────────────────────────────────────────────────

@app.task(name="workers.tasks.health_check_task")
def health_check_task() -> dict:
    """Periodic beat task: check all services and update Celery queue depth gauge."""
    from retrieval.vector_store import ping as chroma_ping
    from observability.metrics import celery_queue_depth_gauge

    ch = chroma_ping()
    logger.info("Periodic health check — chroma=%s", ch)

    try:
        import redis
        from config import cfg
        r = redis.from_url(cfg.celery_broker_url)
        depth: int = r.llen("celery")  # type: ignore[assignment]  # sync client always returns int
        celery_queue_depth_gauge.set(float(depth))
    except Exception:
        pass

    return {"chroma": ch}
