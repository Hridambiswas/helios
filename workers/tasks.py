# workers/tasks.py — Helios Celery task definitions
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import asyncio
import logging

from celery import Task

from workers.celery_app import app

logger = logging.getLogger("helios.workers.tasks")


class PipelineTask(Task):
    """Custom Task base that propagates OTEL trace context."""

    def apply_async(self, args=None, kwargs=None, **options):
        from observability.tracing import inject_celery_context
        headers = options.pop("headers", {}) or {}
        inject_celery_context(headers)
        return super().apply_async(args=args, kwargs=kwargs, headers=headers, **options)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Task %s failed: %s", task_id, exc)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning("Task %s retrying: %s", task_id, exc)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("Task %s completed", task_id)


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
def run_pipeline_task(self, query: str, user_id: str | None = None) -> dict:
    """
    Celery task that executes the full Helios agent pipeline.
    Retries up to 3 times on transient failures.
    """
    from pipeline.run import run_pipeline
    from observability.tracing import extract_celery_context

    extract_celery_context(self.request.headers or {})
    logger.info("Task %s: pipeline for user=%s query=%.60s...", self.request.id, user_id, query)

    try:
        state = run_pipeline(query, user_id=user_id)
        if state.get("error"):
            raise RuntimeError(state["error"])
        return {
            "answer": state.get("answer"),
            "critic_passed": state.get("critic_passed"),
            "critic_scores": state.get("critic_scores"),
        }
    except Exception as exc:
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
    from storage.object_store import download
    from retrieval.vector_store import upsert_batch
    from retrieval.bm25_search import get_index
    from langchain_openai import OpenAIEmbeddings
    from config import cfg

    logger.info("Ingesting doc %s from MinIO key %s", doc_id, minio_key)
    content = download(minio_key)
    text = content.decode("utf-8", errors="replace")

    raw_chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
    chunks: list[str] = []
    for chunk in raw_chunks:
        for i in range(0, len(chunk), 500):
            chunks.append(chunk[i: i + 500])

    embedder = OpenAIEmbeddings(model=cfg.openai_embedding_model, api_key=cfg.openai_api_key)
    chunk_ids = [f"{doc_id}::chunk::{i}" for i in range(len(chunks))]
    embeddings = embedder.embed_documents(chunks)
    metas = [{"doc_id": doc_id, "filename": filename, "chunk_idx": i} for i in range(len(chunks))]

    upsert_batch(chunk_ids, embeddings, chunks, metas)
    get_index().add_batch(chunk_ids, chunks, metas)

    logger.info("Ingested %d chunks for doc %s", len(chunks), doc_id)
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
        depth = r.llen("celery")
        celery_queue_depth_gauge.set(depth)
    except Exception:
        pass

    return {"chroma": ch}
