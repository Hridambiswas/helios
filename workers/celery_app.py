# workers/celery_app.py — Helios Celery application
# Author: Hridam Biswas | Project: Helios

import logging

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure, task_retry, worker_ready

logger = logging.getLogger("helios.workers")

from config import cfg

app = Celery(
    "helios",
    broker=cfg.celery_broker_url,
    backend=cfg.celery_result_backend,
    include=["workers.tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                   # ack after completion, not on receipt
    worker_prefetch_multiplier=1,          # one task at a time per worker slot
    task_soft_time_limit=120,              # SIGTERM at 120s
    task_time_limit=180,                   # SIGKILL at 180s
    task_max_retries=3,
    task_default_retry_delay=10,           # seconds between retries
    result_expires=3600,                   # results kept for 1h
    broker_connection_retry_on_startup=True,
)

# ── Celery Beat schedule (periodic tasks) ─────────────────────────────────────
app.conf.beat_schedule = {
    "chroma-index-health-check": {
        "task": "workers.tasks.health_check_task",
        "schedule": 60.0,
    },
    "expire-refresh-tokens": {
        "task": "workers.beat_tasks.expire_refresh_tokens",
        "schedule": 3600.0,   # every hour
    },
    "bm25-index-stats": {
        "task": "workers.beat_tasks.bm25_index_stats",
        "schedule": 300.0,    # every 5 min
    },
}


# ── Signal hooks for Prometheus metrics ──────────────────────────────────────

@task_prerun.connect
def on_task_prerun(task_id, task, *args, **kwargs):
    from observability.metrics import celery_task_counter
    celery_task_counter.labels(task_name=task.name, status="sent").inc()


@task_postrun.connect
def on_task_postrun(task_id, task, retval, state, *args, **kwargs):
    from observability.metrics import celery_task_counter
    celery_task_counter.labels(task_name=task.name, status="success").inc()


@task_failure.connect
def on_task_failure(task_id, exception, traceback, sender, *args, **kwargs):
    from observability.metrics import celery_task_counter
    celery_task_counter.labels(task_name=sender.name, status="failure").inc()


@task_retry.connect
def on_task_retry(request, reason, einfo, *args, **kwargs):
    from observability.metrics import celery_task_counter
    celery_task_counter.labels(task_name=request.task, status="retry").inc()


@worker_ready.connect
def rebuild_bm25_index(sender, **kwargs):
    """
    Rebuild the in-memory BM25 index from ChromaDB on worker startup.

    Each worker process starts with an empty index. Without this, documents
    ingested before the current worker process started would be invisible to
    BM25 search. ChromaDB stores the original document text, so we can
    reconstruct the full corpus on startup.
    """
    try:
        from retrieval.vector_store import _get_collection
        from retrieval.bm25_search import get_index

        collection = _get_collection()
        results = collection.get(include=["documents", "metadatas"])
        ids: list[str] = results.get("ids", [])
        documents: list[str] = results.get("documents", []) or []
        raw_metas = results.get("metadatas") or []
        metadatas: list[dict] = [dict(m) for m in raw_metas]

        if ids:
            get_index().add_batch(ids, documents, metadatas)
            logger.info("BM25 index rebuilt: %d documents loaded from ChromaDB", len(ids))
        else:
            logger.info("BM25 index rebuild: ChromaDB collection is empty")
    except Exception as exc:
        logger.warning("BM25 index rebuild failed on worker startup: %s", exc)
