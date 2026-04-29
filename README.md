# Helios

**Distributed Multi-Modal Agentic GenAI Platform**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-OTLP-purple.svg)](https://opentelemetry.io)

Helios is a production-grade, five-agent RAG pipeline with hybrid retrieval (dense + CLIP + BM25), sandboxed Python execution, LLM-as-judge critic scoring, Celery async workers, JWT auth, WebSocket streaming, and full OpenTelemetry + Prometheus observability — all deployable via Docker Compose or Kubernetes.

---

## Architecture

```
                              ┌────────────────────────────────────────────────┐
                              │                  FastAPI  :8000                │
                              │                                                 │
  Client ──REST──────────────▶│  POST /api/v1/query     (sync)                 │
  Client ──REST──────────────▶│  POST /api/v1/query/async (Celery)             │
  Client ──WebSocket─────────▶│  WS   /ws/query         (streaming events)     │
  Client ──REST──────────────▶│  POST /api/v1/ingest    (doc upload)           │
                              └──────────────┬─────────────────────────────────┘
                                             │
                                     run_pipeline()
                                             │
                              ┌──────────────▼─────────────────────────────────┐
                              │         LangGraph  StateGraph                  │
                              │                                                 │
                              │   ┌──────────┐    ┌───────────┐               │
                              │   │ Planner  │───▶│ Retriever │               │
                              │   │ (GPT-4o) │    │ (hybrid)  │               │
                              │   └──────────┘    └─────┬─────┘               │
                              │         │               │                      │
                              │         │         ┌─────▼──────┐              │
                              │         └────────▶│  Executor  │              │
                              │                   │ (sandbox)  │              │
                              │                   └─────┬──────┘              │
                              │                         │                      │
                              │                   ┌─────▼──────────┐          │
                              │                   │  Synthesizer   │          │
                              │                   │  (GPT-4o 0.2)  │          │
                              │                   └─────┬──────────┘          │
                              │                         │                      │
                              │                   ┌─────▼──────┐              │
                              │                   │   Critic   │              │
                              │                   │ (LLM judge)│              │
                              │                   └────────────┘              │
                              └────────────────────────────────────────────────┘
                                             │
               ┌─────────────────────────────┼────────────────────────────────┐
               │                             │                                │
      ┌────────▼──────┐          ┌───────────▼───────┐           ┌───────────▼───────┐
      │  PostgreSQL   │          │  Redis  (cache,   │           │  ChromaDB         │
      │  (users,      │          │  rate-limit,      │           │  (cosine vectors) │
      │   queries,    │          │  Celery broker,   │           │                   │
      │   documents)  │          │  checkpoints)     │           │  MinIO            │
      └───────────────┘          └───────────────────┘           │  (raw doc files)  │
                                                                  └───────────────────┘
```

---

## Performance Graphs

Visualisation scripts are in `graphs/`. Run from the repo root after installing requirements:

```bash
python graphs/agent_latency.py             # p50/p95/p99 latency per agent stage
python graphs/retrieval_score_fusion.py    # Dense / CLIP / BM25 weight breakdown by query type
python graphs/critic_score_distribution.py # Groundedness, faithfulness, completeness distributions
```

| Script | What it shows |
|---|---|
| `agent_latency.py` | Grouped bar chart of p50/p95/p99 latency for each of the 5 pipeline agents |
| `retrieval_score_fusion.py` | Stacked bar + pie of retrieval signal weights across query categories |
| `critic_score_distribution.py` | Histograms of critic scores with gate threshold and block-rate annotation |

---

## Agents

| Agent | Model | Role | Key behaviour |
|---|---|---|---|
| **Planner** | GPT-4o (T=0) | Query decomposition | Produces typed `subtasks[]`, `requires_retrieval`, `requires_code` JSON; caps at `PLANNER_MAX_SUBTASKS` |
| **Retriever** | OpenAI Ada + CLIP ViT-B/32 + BM25Okapi | Hybrid document retrieval | Weighted score fusion across three paths; deduplicates by `doc_id`; emits Prometheus histograms |
| **Executor** | CPython 3.11 | Sandboxed code runner | AST import whitelist guard; forbidden builtins stripped; daemon thread with configurable timeout; stdout capped at 8 KB |
| **Synthesizer** | GPT-4o (T=0.2) | Grounded answer generation | Enforces context-only citation; appends `[doc_id]` citation block with filenames |
| **Critic** | GPT-4o (T=0) | LLM-as-judge QA | Scores `groundedness`, `faithfulness`, `completeness` ∈ [0,1]; blocks response if overall < `CRITIC_MIN_SCORE` |

---

## Retrieval

Helios uses three complementary retrieval signals fused with configurable weights:

```
query
  ├── OpenAI text-embedding-3-small  ──weight 0.6──▶  ChromaDB (cosine)
  ├── CLIP openai/clip-vit-base-patch32 ──weight 0.3──▶  ChromaDB (cosine)
  └── BM25Okapi (rank_bm25)  ──weight 0.1──▶  in-memory inverted index
                                              (thread-safe, dedup on batch add)
                    └──── weighted sum ───▶ top-K deduplicated docs
```

Tune weights via env vars: `RETRIEVER_DENSE_WEIGHT`, `RETRIEVER_CLIP_WEIGHT`, `RETRIEVER_BM25_WEIGHT`.

---

## Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.111, Pydantic v2, OAuth2 Bearer (JWT HS256) |
| **Agent graph** | LangGraph 0.2 `StateGraph` with conditional routing |
| **LLM** | OpenAI GPT-4o (planner, synthesizer, critic) |
| **Dense retrieval** | OpenAI `text-embedding-3-small` → ChromaDB HTTP |
| **Multi-modal retrieval** | CLIP `openai/clip-vit-base-patch32` (HuggingFace) |
| **Sparse retrieval** | BM25Okapi (`rank-bm25`) in-memory |
| **Relational DB** | PostgreSQL 16 via async SQLAlchemy 2.0 |
| **Cache / broker** | Redis 7 (aioredis + Celery) |
| **Object store** | MinIO (S3-compatible, `minio` Python SDK) |
| **Async workers** | Celery 5 (`ack_late=True`, beat scheduler) |
| **Migrations** | Alembic async |
| **Auth** | bcrypt passwords, JWT access + refresh tokens (rotation in Postgres) |
| **Tracing** | OpenTelemetry SDK → OTLP gRPC exporter |
| **Metrics** | Prometheus + prometheus-fastapi-instrumentator |
| **Logging** | structlog (JSON in prod, ConsoleRenderer in dev) |
| **Containers** | Docker + Docker Compose v2 |
| **Orchestration** | Kubernetes manifests (Deployment, Service, Ingress, ConfigMap) |

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/hridambiswas/helios.git
cd helios
cp .env.example .env
# Fill in OPENAI_API_KEY and JWT_SECRET_KEY at minimum
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

Services started:

| Service | Port | Purpose |
|---|---|---|
| `api` | 8000 | FastAPI (4 uvicorn workers) |
| `worker` | — | Celery task pool |
| `beat` | — | Periodic task scheduler |
| `postgres` | 5432 | PostgreSQL 16 |
| `redis` | 6379 | Cache + Celery broker |
| `minio` | 9000 / 9001 | Object store + console |
| `chroma` | 8001 | ChromaDB vector store |
| `otel-collector` | 4317 | OTLP trace receiver |
| `prometheus` | 9090 | Metrics scraper |

### 3. Register and query

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","email":"alice@example.com","password":"secret"}'

# Get token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -d 'username=alice&password=secret' | jq -r .access_token)

# Run a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "Explain transformer attention in one paragraph"}'
```

### 4. Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/paper.txt"
```

Or bulk-ingest a directory:

```bash
python scripts/ingest_demo.py --dir ./docs --ext .txt .md .pdf
```

### 5. WebSocket streaming

```javascript
const ws = new WebSocket(
  `ws://localhost:8000/ws/query?token=${TOKEN}`
);
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(JSON.stringify({ query: "What is BM25?" }));
// Events: planning → retrieving → evaluating → done
```

---

## Configuration

All settings are loaded from environment variables (or `.env`). See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `JWT_SECRET_KEY` | — | **Required.** HS256 signing key |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async Postgres DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis DSN |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO host:port |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `RETRIEVER_TOP_K` | `10` | Documents per retrieval path |
| `RETRIEVER_DENSE_WEIGHT` | `0.6` | Dense (OpenAI) retrieval weight |
| `RETRIEVER_CLIP_WEIGHT` | `0.3` | CLIP retrieval weight |
| `RETRIEVER_BM25_WEIGHT` | `0.1` | BM25 sparse weight |
| `EXECUTOR_TIMEOUT_SECONDS` | `15` | Sandboxed execution timeout |
| `CRITIC_MIN_SCORE` | `0.7` | Minimum critic score to pass |
| `PLANNER_MAX_SUBTASKS` | `5` | Cap on decomposed subtasks |
| `APP_ENV` | `development` | `development` or `production` |

---

## Evaluation

Helios ships a 30-question evaluation harness covering factual recall, analytical reasoning, code generation, multi-step problems, and no-context (hallucination-probe) queries.

### Run the harness

```bash
# All 30 questions
python scripts/run_eval.py

# Subset by type
python scripts/run_eval.py --type factual

# Specific question IDs, skip file output
python scripts/run_eval.py --ids 1 3 7 --no-json --no-csv
```

### Metrics

| Metric | Formula | Target |
|---|---|---|
| **Keyword coverage** | `hits / expected_keywords` | ≥ 0.75 |
| **Hallucination penalty** | −50% if any forbidden keyword present | 0 hits |
| **Groundedness** | Critic LLM score ∈ [0,1] | ≥ 0.7 |
| **Faithfulness** | Critic LLM score ∈ [0,1] | ≥ 0.7 |
| **Completeness** | Critic LLM score ∈ [0,1] | ≥ 0.6 |
| **Combined** | 0.4×keyword + 0.6×avg(critic dims) | ≥ 0.65 |
| **ECE** | Expected Calibration Error (10-bin) | < 0.10 |

Reports are written to `eval/reports/` as timestamped JSON and CSV.

### Latency benchmark (no live services required)

```bash
python scripts/benchmark.py --n 20 --agent executor
```

```
Helios Agent Benchmark — 20 iterations per agent
=======================================================
  executor (sqrt loop x1000)
    mean  : 2.4ms
    median: 2.3ms
    p95   : 3.1ms
    p99   : 3.8ms
    min   : 2.1ms  max: 4.2ms
```

---

## Project layout

```
helios/
├── agents/
│   ├── base.py            # Timing + Prometheus instrumentation mixin
│   ├── planner.py         # GPT-4o query decomposition
│   ├── retriever.py       # Hybrid dense+CLIP+BM25 retrieval
│   ├── executor.py        # AST-guarded sandboxed Python runner
│   ├── synthesizer.py     # Grounded answer synthesis with citations
│   └── critic.py          # LLM-as-judge scoring
├── api/
│   ├── auth.py            # bcrypt + JWT + refresh token rotation
│   ├── middleware.py      # Request-ID header, Redis rate limiter
│   ├── routes.py          # REST endpoints
│   ├── schemas.py         # Pydantic v2 request/response models
│   └── websocket.py       # JWT-authenticated streaming WebSocket
├── eval/
│   ├── questions.py       # 30-question bank
│   ├── scorers.py         # Keyword + critic score computation
│   ├── harness.py         # End-to-end eval runner with CSV/JSON reports
│   └── calibration.py     # ECE + overconfidence check
├── graph/
│   ├── pipeline.py        # LangGraph StateGraph — 5 nodes, conditional routing
│   └── checkpointing.py   # Redis-backed per-session state snapshots
├── observability/
│   ├── metrics.py         # All Prometheus metric definitions
│   ├── tracing.py         # OTLP setup + span() context manager
│   └── logging_config.py  # structlog JSON/Console renderer
├── retrieval/
│   ├── vector_store.py    # ChromaDB HTTP client wrapper
│   ├── clip_encoder.py    # CLIP text+image encoder (L2-normalised)
│   └── bm25_search.py     # Thread-safe BM25 in-memory index
├── scripts/
│   ├── ingest_demo.py     # Bulk directory ingest CLI
│   ├── run_eval.py        # Eval harness CLI wrapper
│   └── benchmark.py       # Agent latency benchmark
├── storage/
│   ├── database.py        # Async SQLAlchemy engine + session factory
│   ├── models.py          # ORM: User, QueryRecord, Document, RefreshToken
│   ├── crud.py            # Reusable async CRUD helpers
│   ├── cache.py           # Redis async wrapper with TTL tiers
│   ├── object_store.py    # MinIO upload/download/presign/copy
│   └── migrations/        # Alembic async migration environment
├── workers/
│   ├── celery_app.py      # Celery app — broker, beat schedule, signals
│   ├── tasks.py           # run_pipeline_task, ingest_document_task
│   └── beat_tasks.py      # Periodic: token cleanup, BM25 stats
├── tests/
│   ├── test_agents.py     # Executor, scorer, BM25 unit tests
│   ├── test_api.py        # Health + auth route tests (mocked)
│   ├── test_pipeline.py   # LangGraph integration smoke tests
│   └── test_storage.py    # Cache + object store unit tests
├── k8s/                   # Kubernetes Deployment, Service, Ingress, ConfigMap
├── config.py              # Pydantic BaseSettings with lru_cache
├── main.py                # FastAPI app factory + lifespan hooks
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
└── otel-config.yaml
```

---

## Observability

### Prometheus metrics

| Metric | Type | Labels |
|---|---|---|
| `helios_agent_latency_seconds` | Histogram | `agent` |
| `helios_agent_errors_total` | Counter | `agent` |
| `helios_pipeline_latency_seconds` | Histogram | — |
| `helios_pipeline_requests_total` | Counter | `status` (success/failed/critic_failed) |
| `helios_retrieval_docs_total` | Histogram | `source` (dense/clip/bm25) |
| `helios_critic_score` | Histogram | `dimension` (groundedness/faithfulness/completeness) |
| `helios_critic_pass_total` | Counter | `result` (pass/fail) |
| `helios_active_websockets` | Gauge | — |
| `helios_celery_tasks_total` | Counter | `task`, `state` |

### Tracing

All agents and the pipeline entry point are instrumented with OTLP spans. Celery tasks propagate trace context via task headers, maintaining a single trace per request across sync and async execution paths.

### Structured logging

```json
{
  "event": "Execution done — success=True  stdout_len=42",
  "logger": "helios.agents.executor",
  "level": "info",
  "request_id": "4f3a2b1c",
  "timestamp": "2026-04-26T10:30:00.000Z"
}
```

---

## Security

- **Rate limiting**: 60 requests / 60 seconds per IP via Redis atomic `INCR`. Returns `429` with `Retry-After` header. Exempt: `/health`, `/metrics`, `/docs`.
- **Sandboxed execution**: User-submitted code passes AST validation before running. Only whitelisted top-level modules are importable (`math`, `numpy`, `pandas`, `scipy`, etc.). Forbidden builtins (`open`, `exec`, `eval`, `__import__`, …) are stripped from `__builtins__`. Execution runs in a daemon thread with a hard timeout.
- **JWT rotation**: Refresh tokens are stored hashed (bcrypt) in Postgres. Each refresh call atomically revokes the previous token and issues a new one.
- **Non-root container**: The `Dockerfile` creates and switches to a `helios` user — no process runs as root inside the image.

---

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

No live services required — all external dependencies are mocked.

---

## Author

**Hridam Biswas**  
B.Tech Computer Science, KIIT University  
[hridambiswas2005@gmail.com](mailto:hridambiswas2005@gmail.com)
