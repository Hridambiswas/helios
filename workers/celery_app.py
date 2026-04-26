# workers/celery_app.py — Helios Celery application
# Author: Hridam Biswas | Project: Helios

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure, task_retry

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
        "schedule": 60.0,   # every 60 seconds
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
