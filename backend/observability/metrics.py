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

# ── Resilience metrics ────────────────────────────────────────────────────────

circuit_breaker_trips_counter = Counter(
    "helios_circuit_breaker_trips_total",
    "Total times a circuit breaker transitioned to OPEN",
    ["circuit"],
)

bulkhead_rejected_counter = Counter(
    "helios_bulkhead_rejected_total",
    "Requests rejected by bulkhead isolation",
    ["agent"],
)

backpressure_shed_counter = Counter(
    "helios_backpressure_shed_total",
    "Requests shed due to backpressure (queue depth or active pipelines)",
)

active_pipelines_gauge = Gauge(
    "helios_active_pipelines",
    "Currently executing synchronous pipeline runs",
)

# ── Security metrics ──────────────────────────────────────────────────────────

auth_failure_counter = Counter(
    "helios_auth_failures_total",
    "Total failed authentication attempts",
    ["reason"],   # bad_credentials / inactive_user / bad_token
)

brute_force_blocked_counter = Counter(
    "helios_brute_force_blocked_total",
    "Requests blocked by the brute-force protection middleware",
    ["path"],
)

# ── Ingest metrics ────────────────────────────────────────────────────────────

ingest_latency_histogram = Histogram(
    "helios_ingest_latency_ms",
    "Document ingestion pipeline latency in milliseconds",
    buckets=[500, 1000, 2500, 5000, 10000, 30000, 60000],
)

ingest_chunk_count_histogram = Histogram(
    "helios_ingest_chunk_count",
    "Number of chunks produced per ingested document",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000],
)

ingest_bytes_histogram = Histogram(
    "helios_ingest_bytes",
    "Raw document size in bytes at ingest time",
    buckets=[1024, 4096, 16384, 65536, 262144, 1048576, 10485760],
)

# ── Guest metrics ─────────────────────────────────────────────────────────────

guest_query_counter = Counter(
    "helios_guest_queries_total",
    "Unauthenticated (guest) queries processed",
)

guest_blocked_counter = Counter(
    "helios_guest_blocked_total",
    "Guest queries blocked due to exceeding the free-tier limit",
)

# ── Storage metrics ───────────────────────────────────────────────────────────

db_query_latency_histogram = Histogram(
    "helios_db_query_latency_ms",
    "SQLAlchemy query latency in milliseconds",
    ["operation"],   # select / insert / update / delete
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

cache_hit_counter = Counter(
    "helios_cache_hits_total",
    "Redis cache hits",
    ["key_prefix"],
)

cache_miss_counter = Counter(
    "helios_cache_misses_total",
    "Redis cache misses",
    ["key_prefix"],
)
