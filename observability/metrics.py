# observability/metrics.py — Helios Prometheus metrics
# Author: Hridam Biswas | Project: Helios

from prometheus_client import Counter, Histogram, Gauge, Summary

# ── Agent metrics ─────────────────────────────────────────────────────────────

agent_latency_histogram = Histogram(
    "helios_agent_latency_ms",
    "Per-agent execution latency in milliseconds",
    ["agent"],
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

agent_error_counter = Counter(
    "helios_agent_errors_total",
    "Total errors raised by each agent",
    ["agent"],
)

# ── Pipeline metrics ──────────────────────────────────────────────────────────

pipeline_latency_histogram = Histogram(
    "helios_pipeline_latency_ms",
    "End-to-end pipeline latency per query",
    buckets=[100, 250, 500, 1000, 2500, 5000, 10000, 30000],
)

pipeline_requests_counter = Counter(
    "helios_pipeline_requests_total",
    "Total queries processed by the pipeline",
    ["status"],   # labels: success / failed / critic_failed
)

# ── Retrieval metrics ─────────────────────────────────────────────────────────

retrieval_docs_histogram = Histogram(
    "helios_retrieval_docs_returned",
    "Number of docs returned per retrieval call",
    ["source"],   # dense / clip / bm25 / merged
    buckets=[0, 1, 2, 5, 10, 20, 50],
)

retrieval_score_summary = Summary(
    "helios_retrieval_score",
    "Distribution of retrieval scores",
    ["source"],
)

# ── Critic metrics ────────────────────────────────────────────────────────────

critic_score_histogram = Histogram(
    "helios_critic_score",
    "Critic overall score distribution",
    ["dimension"],   # groundedness / faithfulness / completeness / overall
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

critic_pass_counter = Counter(
    "helios_critic_pass_total",
    "Critic pass/fail counts",
    ["result"],   # pass / fail
)

# ── API metrics ───────────────────────────────────────────────────────────────

http_requests_counter = Counter(
    "helios_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

active_websocket_gauge = Gauge(
    "helios_active_websockets",
    "Currently open WebSocket connections",
)

# ── Celery metrics ────────────────────────────────────────────────────────────

celery_task_counter = Counter(
    "helios_celery_tasks_total",
    "Total Celery tasks dispatched",
    ["task_name", "status"],   # status: sent / success / failure / retry
)

celery_queue_depth_gauge = Gauge(
    "helios_celery_queue_depth",
    "Approximate Celery queue depth",
)
