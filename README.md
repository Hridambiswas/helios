# Helios

**Distributed Multi-Agent RAG Platform**

[![CI](https://github.com/hridambiswas/helios/actions/workflows/ci.yml/badge.svg)](https://github.com/hridambiswas/helios/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange.svg)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-OTLP-purple.svg)](https://opentelemetry.io)

🌐 **Live:** [frontend-omega-blush-87.vercel.app](https://frontend-omega-blush-87.vercel.app) — API: [helios-hridam.ddns.net](https://helios-hridam.ddns.net)

Helios is a production-grade, five-agent RAG pipeline with hybrid retrieval (dense + CLIP + BM25), sandboxed Python execution, LLM-as-judge critic scoring, Celery async workers, JWT + GitHub OAuth, WebSocket streaming, and full OpenTelemetry + Prometheus observability — deployed on EC2 (backend) and Vercel (frontend) with Supabase PostgreSQL.

### What's new in v1.1

| # | Feature | Summary |
|---|---------|---------|
| 1 | **Multi-turn memory** | Conversation history sent with every query; Planner and Synthesizer use prior turns for context |
| 2 | **Per-token streaming** | Groq `llm.stream()` feeds tokens through an asyncio queue to WebSocket; answers appear word-by-word |
| 3 | **Server-side conversations** | Conversations and messages persist in Supabase; cross-device history for logged-in users |
| 4 | **Critic retry loop** | Failed critic score triggers one re-synthesis pass with improvement suggestions injected |
| 5 | **Guest query limit ×5** | Raised from 1 to 5 free queries before sign-in is required |
| 6 | **Mobile layout** | Bottom nav bar, safe-area input, auto-collapsing sidebar, full-width bubbles on xs |
| 7 | **Chunk feedback** | Upload panel shows chunk previews and a test-retrieval search box per document |

---

## Architecture

```
  React + Vite (Vercel)
  ├── Chat UI (sidebar + conversation history)
  ├── REST/WebSocket → FastAPI :8000 (EC2)
  └── GitHub OAuth → /api/v1/auth/github

                              ┌────────────────────────────────────────────────┐
                              │              FastAPI  :8000  (EC2)             │
                              │                                                 │
  Client ──REST──────────────▶│  POST /api/v1/query     (sync)                 │
  Client ──REST──────────────▶│  POST /api/v1/query/async (Celery)             │
  Client ──WebSocket─────────▶│  WS   /ws/query         (streaming events)     │
  Client ──REST──────────────▶│  POST /api/v1/ingest    (doc upload)           │
  Client ──GET───────────────▶│  GET  /api/v1/auth/github (OAuth redirect)     │
                              └──────────────┬─────────────────────────────────┘
                                             │
                                     run_pipeline()
                                             │
                              ┌──────────────▼─────────────────────────────────┐
                              │         LangGraph  StateGraph                  │
                              │                                                 │
                              │  ┌─────────────┐   ┌────────────────────┐     │
                              │  │   Planner   │──▶│     Retriever      │     │
                              │  │ Llama 3.3   │   │ BAAI/bge + CLIP    │     │
                              │  │   70B (T=0) │   │ + BM25 → fused K   │     │
                              │  └─────────────┘   └──────────┬─────────┘     │
                              │        │                       │               │
                              │        │             ┌─────────▼──────┐       │
                              │        └────────────▶│    Executor    │       │
                              │                      │  (AST sandbox) │       │
                              │                      └─────────┬──────┘       │
                              │                                │               │
                              │                      ┌─────────▼──────────┐   │
                              │                      │    Synthesizer     │   │
                              │                      │  Llama 3.3 70B     │   │
                              │                      │  streams tokens    │   │
                              │                      │  + history context │   │
                              │                      └─────────┬──────────┘   │
                              │                                │               │
                              │                      ┌─────────▼──────┐       │
                              │                      │     Critic     │       │
                              │                      │  Llama 3.3 70B │       │
                              │                      │  min score 0.5 │       │
                              │                      │  → retry once  │       │
                              │                      └────────────────┘       │
                              └────────────────────────────────────────────────┘
                                             │
        ┌────────────────────────────────────┼──────────────────────────────┐
        │                                    │                              │
┌───────▼──────────┐           ┌─────────────▼──────┐          ┌───────────▼──────┐
│ Supabase         │           │ Redis 7            │          │ ChromaDB         │
│ PostgreSQL       │           │ (rate-limit,       │          │ (cosine vectors) │
│ (users, queries, │           │  Celery broker,    │          │                  │
│  documents,      │           │  checkpoints)      │          │ MinIO            │
│  conversations,  │           └────────────────────┘          │ (raw doc files)  │
│  refresh tokens) │                                            └──────────────────┘
└──────────────────┘
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
| `critic_score_distribution.py` | Histograms of critic scores with `CRITIC_MIN_SCORE=0.5` gate threshold and block-rate annotation |

---

## Agents

| Agent | Model | Role | Key behaviour |
|---|---|---|---|
| **Planner** | Groq `llama-3.3-70b-versatile` (T=0) | Query decomposition | Produces typed `subtasks[]`, `requires_retrieval`, `requires_code` JSON; caps at `PLANNER_MAX_SUBTASKS` |
| **Retriever** | `BAAI/bge-small-en-v1.5` + CLIP `ViT-B/32` + BM25Okapi | Hybrid document retrieval | Weighted score fusion (dense 0.6 + CLIP 0.3 + BM25 0.1); deduplicates by `doc_id`; emits Prometheus histograms |
| **Executor** | CPython 3.11 | Sandboxed code runner | AST import whitelist guard; forbidden builtins stripped; daemon thread with configurable timeout; stdout capped at 8 KB |
| **Synthesizer** | Groq `llama-3.3-70b-versatile` (T=0.4) | Grounded answer generation | Cites local docs as [D1], web sources as [W1]; uses emojis naturally; uses `conversation_history` for multi-turn context; per-token streaming via `llm.stream()`; generates follow-up questions; injects critic suggestions on retry |
| **Critic** | Groq `llama-3.3-70b-versatile` (T=0) | LLM-as-judge QA + retry trigger | Scores `groundedness`, `faithfulness`, `completeness` ∈ [0,1]; if overall < `CRITIC_MIN_SCORE` (0.5) routes back to synthesizer with suggestions (max 1 retry) |

---

## Retrieval

Helios uses three complementary retrieval signals fused with configurable weights:

```
query
  ├── BAAI/bge-small-en-v1.5 (HuggingFace, local)  ──weight 0.6──▶  ChromaDB (cosine)
  ├── CLIP openai/clip-vit-base-patch32 (HuggingFace) ──weight 0.3──▶  ChromaDB (cosine)
  └── BM25Okapi (rank_bm25)  ──weight 0.1──▶  in-memory inverted index
                                               (thread-safe, dedup on batch add)
                    └──── weighted RRF sum ──▶ top-K deduplicated docs
```

No external embedding API is required — both dense and CLIP models run locally via HuggingFace `transformers`.

Tune weights via env vars: `RETRIEVER_DENSE_WEIGHT`, `RETRIEVER_CLIP_WEIGHT`, `RETRIEVER_BM25_WEIGHT`.

---

## Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18 + Vite + TypeScript, deployed on Vercel |
| **API** | FastAPI 0.115, Pydantic v2, OAuth2 Bearer (JWT HS256) |
| **Agent graph** | LangGraph 0.2 `StateGraph` with conditional routing |
| **LLM** | Groq `llama-3.3-70b-versatile` (planner, synthesizer, critic) via `langchain-groq` |
| **Dense retrieval** | `BAAI/bge-small-en-v1.5` (HuggingFace, local) → ChromaDB HTTP |
| **Multi-modal retrieval** | CLIP `openai/clip-vit-base-patch32` (HuggingFace, local) |
| **Sparse retrieval** | BM25Okapi (`rank-bm25`) in-memory |
| **Relational DB** | Supabase PostgreSQL via async SQLAlchemy 2.0 + asyncpg |
| **Cache / broker** | Redis 7 (aioredis + Celery) |
| **Object store** | MinIO (S3-compatible, `minio` Python SDK) |
| **Async workers** | Celery 5 (`ack_late=True`, beat scheduler) |
| **Migrations** | Alembic async |
| **Auth** | bcrypt passwords, JWT access + refresh tokens + GitHub OAuth2 |
| **Tracing** | OpenTelemetry SDK → OTLP gRPC exporter |
| **Metrics** | Prometheus + prometheus-fastapi-instrumentator |
| **Logging** | structlog (JSON in prod, ConsoleRenderer in dev) |
| **Containers** | Docker + Docker Compose v2 (EC2 deployment) |

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/Hridambiswas/helios.git
cd helios
cp .env.example .env
# Fill in GROQ_API_KEY and JWT_SECRET_KEY at minimum
# Optional: GITHUB_CLIENT_ID / GITHUB_CLIENT_SECRET for GitHub OAuth
# Optional: SUPABASE_DATABASE_URL to use Supabase instead of local Postgres
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
# Register (password must be 8+ chars, include upper, lower, digit)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","email":"alice@example.com","password":"Secret123"}'

# Get token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -d 'username=alice&password=Secret123' | jq -r .access_token)

# Run a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query": "Explain transformer attention in one paragraph"}'
```

Or use the live UI at [frontend-omega-blush-87.vercel.app](https://frontend-omega-blush-87.vercel.app) — type a question, the chat view opens automatically.

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

### 5. Frontend (React + Vite)

The frontend is a TypeScript React app deployed on Vercel.

**Chat interface features:**
- ChatGPT-style layout: collapsible sidebar + main chat area
- **Multi-turn memory** — conversation history is sent with every query; Helios remembers context across turns
- **Per-token streaming** — answers appear word-by-word with an inline blinking cursor; `Writing` step indicator during synthesis
- **Server-side persistence** — logged-in users' conversations sync to Supabase; accessible across devices
- Conversations fall back to `localStorage` when signed out (up to 50 sessions)
- **Retry indicator** — "Improving answer…" label shown when critic triggers a re-synthesis
- **Mobile-optimised** — bottom navigation bar, auto-collapsing sidebar, safe-area input padding
- **5 free queries** for guest users before sign-in is required
- Submitting a query from the landing page opens the chat view automatically
- User messages on the right, Helios responses on the left with live pipeline step indicators
- Follow-up question chips, copy button, latency, source count on each answer
- GitHub OAuth — one-click sign in via "Continue with GitHub"
- Upload panel with **chunk preview** and **test-retrieval search** per indexed document

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Set `VITE_API_URL` in `frontend/.env` to point at your backend.

### 6. WebSocket streaming

```javascript
const ws = new WebSocket(
  `ws://localhost:8000/ws/query?token=${TOKEN}`
);
ws.onmessage = (e) => console.log(JSON.parse(e.data));
// Send query with optional conversation history
ws.send(JSON.stringify({
  query: "What is BM25?",
  history: [
    { role: "user",      content: "Tell me about retrieval methods." },
    { role: "assistant", content: "Retrieval methods include dense, sparse, and hybrid approaches…" }
  ]
}));
// Events: planning → retrieving → synthesizing → token{…} × N → evaluating → done
// On critic fail with retry: planning → synthesizing → token{…} → evaluating → done
```

---

## Configuration

All settings are loaded from environment variables (or `.env`). See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key for all LLM calls |
| `JWT_SECRET_KEY` | — | **Required.** HS256 signing key (generate with `secrets.token_hex(32)`) |
| `SUPABASE_DATABASE_URL` | — | Supabase asyncpg DSN (falls back to local Postgres if unset) |
| `DATABASE_URL` | `postgresql+asyncpg://helios:@localhost/helios` | Local Postgres DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis DSN |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO host:port |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `GITHUB_CLIENT_ID` | — | GitHub OAuth App client ID |
| `GITHUB_CLIENT_SECRET` | — | GitHub OAuth App client secret |
| `OAUTH_FRONTEND_URL` | `https://frontend-omega-blush-87.vercel.app` | Frontend origin for OAuth redirect |
| `OAUTH_BACKEND_URL` | `https://helios-hridam.ddns.net` | Backend URL for building OAuth callback URI |
| `RETRIEVER_TOP_K` | `10` | Documents per retrieval path |
| `RETRIEVER_DENSE_WEIGHT` | `0.6` | BAAI/bge dense retrieval weight |
| `RETRIEVER_CLIP_WEIGHT` | `0.3` | CLIP retrieval weight |
| `RETRIEVER_BM25_WEIGHT` | `0.1` | BM25 sparse weight |
| `EXECUTOR_TIMEOUT_SECONDS` | `15` | Sandboxed execution timeout |
| `CRITIC_MIN_SCORE` | `0.5` | Minimum critic score to pass |
| `PLANNER_MAX_SUBTASKS` | `5` | Cap on decomposed subtasks |
| `GUEST_QUERY_LIMIT` | `5` | Free queries before auth required |
| `APP_ENV` | `development` | `development` or `production` |

---

## New API endpoints (v1.1)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/query` | optional | Run pipeline; accepts `history[]` for multi-turn context |
| `WS` | `/ws/query` | required | Streaming pipeline; accepts `{ query, history[] }` message |
| `GET` | `/api/v1/conversations` | required | List user's server-side conversations |
| `POST` | `/api/v1/conversations` | required | Create a new conversation |
| `GET` | `/api/v1/conversations/{id}` | required | Get conversation + messages |
| `POST` | `/api/v1/conversations/{id}/messages` | required | Append a message |
| `DELETE` | `/api/v1/conversations/{id}` | required | Delete conversation (cascade) |
| `GET` | `/api/v1/documents/{id}/chunks` | required | Paginated chunk text preview |
| `POST` | `/api/v1/documents/{id}/search` | required | Test retrieval scoped to document |

---

## Deployment

### Production setup (current)

| Component | Platform | URL |
|---|---|---|
| Frontend | Vercel (auto-deploy on push to `main`) | [frontend-omega-blush-87.vercel.app](https://frontend-omega-blush-87.vercel.app) |
| Backend API | AWS EC2 (Docker Compose) | [helios-hridam.ddns.net](https://helios-hridam.ddns.net) |
| Database | Supabase managed PostgreSQL | Connection via `SUPABASE_DATABASE_URL` |
| Redis / MinIO / ChromaDB | Self-hosted on EC2 | Docker Compose services |

### GitHub Actions CI/CD

Every push to `main` triggers:
1. `CI` — ruff lint, pyright type-check, pytest (130 tests)
2. `Deploy Backend to EC2` — SSH into EC2, inject secrets into `.env`, `docker compose up -d`
3. `Deploy Frontend to Vercel` — Vercel CLI build + deploy

### GitHub OAuth setup

1. Go to [github.com/settings/developers](https://github.com/settings/developers) → **New OAuth App**
2. Set **Authorization callback URL** to `https://your-backend/api/v1/auth/github/callback`
3. Add `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET` to your `.env` (or GitHub repo secrets as `GH_OAUTH_CLIENT_ID` / `GH_OAUTH_CLIENT_SECRET`)

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
| **Groundedness** | Critic LLM score ∈ [0,1] | ≥ 0.5 |
| **Faithfulness** | Critic LLM score ∈ [0,1] | ≥ 0.5 |
| **Completeness** | Critic LLM score ∈ [0,1] | ≥ 0.5 |
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
│   ├── __init__.py        # Exports all agent classes
│   ├── base.py            # Timing + Prometheus instrumentation mixin
│   ├── planner.py         # Groq query decomposition; context-aware via conversation history
│   ├── retriever.py       # Hybrid dense+CLIP+BM25 retrieval
│   ├── executor.py        # AST-guarded sandboxed Python runner
│   ├── synthesizer.py     # Grounded answer synthesis; per-token streaming; retry guidance
│   └── critic.py          # LLM-as-judge scoring; triggers retry on fail
├── api/
│   ├── __init__.py        # Exports routers and auth helpers
│   ├── auth.py            # bcrypt + JWT + refresh token rotation
│   ├── middleware.py      # Request-ID header, Redis rate limiter
│   ├── routes.py          # REST endpoints
│   ├── schemas.py         # Pydantic v2 request/response models
│   └── websocket.py       # JWT-authenticated streaming WebSocket
├── eval/
│   ├── __init__.py        # Exports scorers and harness
│   ├── questions.py       # 30-question bank
│   ├── scorers.py         # Keyword + critic score computation
│   ├── harness.py         # End-to-end eval runner with CSV/JSON reports
│   └── calibration.py     # ECE + overconfidence check
├── observability/
│   ├── __init__.py        # Exports setup_logging, setup_tracing, span
│   ├── metrics.py         # All Prometheus metric definitions
│   ├── tracing.py         # OTLP setup + span() context manager
│   └── logging_config.py  # structlog JSON/Console renderer
├── pipeline/
│   ├── __init__.py        # Exports run_pipeline, HeliosState
│   ├── run.py             # LangGraph StateGraph — 5 nodes, conditional routing
│   └── checkpointing.py   # Redis-backed per-session state snapshots
├── retrieval/
│   ├── __init__.py        # Exports vector_query, BM25Index, encode_text
│   ├── vector_store.py    # ChromaDB HTTP client wrapper
│   ├── clip_encoder.py    # CLIP text+image encoder (L2-normalised)
│   └── bm25_search.py     # Thread-safe BM25 in-memory index
├── scripts/
│   ├── ingest_demo.py     # Bulk directory ingest CLI
│   ├── run_eval.py        # Eval harness CLI wrapper
│   └── benchmark.py       # Agent latency benchmark
├── storage/
│   ├── __init__.py        # Exports session, models, CRUD helpers
│   ├── database.py        # Async SQLAlchemy engine + session factory
│   ├── models.py          # ORM: User, QueryRecord, Document, RefreshToken
│   ├── crud.py            # Reusable async CRUD helpers
│   ├── cache.py           # Redis async wrapper with TTL tiers
│   ├── object_store.py    # MinIO upload/download/presign/copy
│   └── migrations/        # Alembic async migration environment
├── workers/
│   ├── __init__.py        # Exports celery_app and task handles
│   ├── celery_app.py      # Celery app — broker, beat schedule, signals
│   ├── tasks.py           # run_pipeline_task, ingest_document_task
│   └── beat_tasks.py      # Periodic: token cleanup, BM25 stats
├── tests/
│   ├── conftest.py        # Shared fixtures
│   ├── test_agents.py     # Executor, scorer, BM25 unit tests
│   ├── test_api.py        # Health + auth route tests (mocked)
│   ├── test_pipeline.py   # LangGraph integration smoke tests
│   └── test_storage.py    # Cache + object store unit tests
├── frontend/              # React + Vite + TypeScript (deployed on Vercel)
│   ├── src/components/    # ChatSidebar, ChatView, AuthModal, Hero, etc.
│   ├── src/hooks/         # useAuth, useConversations, useToast, useDebounce
│   └── src/api/           # Axios client wrappers
├── graphs/                # Standalone visualisation scripts (matplotlib)
├── .github/workflows/     # CI (lint + typecheck + tests), deploy-backend, deploy-frontend
├── config.py              # Pydantic BaseSettings with lru_cache
├── main.py                # FastAPI app factory + lifespan hooks
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── pyrightconfig.json
├── pytest.ini
├── prometheus.yml
└── otel-config.yaml
```

---

## Observability

### Prometheus metrics

| Metric | Type | Labels |
|---|---|---|
| `helios_agent_latency_ms` | Histogram | `agent` |
| `helios_agent_errors_total` | Counter | `agent` |
| `helios_pipeline_latency_ms` | Histogram | — |
| `helios_pipeline_requests_total` | Counter | `status` (success/failed/critic_failed) |
| `helios_auth_failures_total` | Counter | `reason` (bad_credentials/inactive_user/bad_token) |
| `helios_brute_force_blocked_total` | Counter | `path` (/auth/login, /auth/register) |
| `helios_retrieval_docs_returned` | Histogram | `source` (dense/clip/bm25/merged) |
| `helios_retrieval_score` | Summary | `source` |
| `helios_critic_score` | Histogram | `dimension` (groundedness/faithfulness/completeness/overall) |
| `helios_critic_pass_total` | Counter | `result` (pass/fail) |
| `helios_active_websockets` | Gauge | — |
| `helios_celery_tasks_total` | Counter | `task_name`, `status` (sent/success/failure/retry) |

### Tracing

All agents and the pipeline entry point are instrumented with OTLP spans. Celery tasks propagate trace context via task headers, maintaining a single trace per request across sync and async execution paths.

### Structured logging

Every log line is emitted as JSON. Key fields:

| Field | Description |
|---|---|
| `event` | Human-readable message |
| `logger` | Module path (e.g. `helios.agents.synthesizer`) |
| `level` | `debug` / `info` / `warning` / `error` |
| `request_id` | UUID propagated from the incoming HTTP/WS request |
| `trace_id` | OTLP trace ID — correlates logs to spans in your tracing backend |
| `timestamp` | ISO-8601 UTC |

```json
{
  "event": "Execution done — success=True  stdout_len=42",
  "logger": "helios.agents.executor",
  "level": "info",
  "request_id": "4f3a2b1c",
  "trace_id": "00-abc123def456-01",
  "timestamp": "2026-04-26T10:30:00.000Z"
}
```

Set `LOG_FORMAT=text` in `.env` to switch to plain-text output for local development.

---

## Security

- **Rate limiting**: 60 requests / 60 seconds per user via Redis atomic `INCR`. Returns `429` with `Retry-After` and `X-RateLimit-*` headers. Exempt: `/health`, `/metrics`, `/docs`.
- **Brute-force protection**: Login and register endpoints are additionally guarded by `AuthBruteForceMiddleware` — blocks an IP after 5 attempts in 5 minutes.
- **Sandboxed execution**: User-submitted code passes AST validation before running. Only whitelisted top-level modules are importable (`math`, `numpy`, `pandas`, `scipy`, etc.). Forbidden builtins (`open`, `exec`, `eval`, `__import__`, …) are stripped from `__builtins__`. Execution runs in a daemon thread with a hard timeout.
- **JWT rotation**: Refresh tokens are stored as SHA-256 hashes in Supabase Postgres. Each refresh call atomically revokes the previous token and issues a new one.
- **GitHub OAuth**: Authorization-code flow with HMAC-signed `state` parameter for CSRF protection. Accounts linked by `oauth_id` first, email as fallback.
- **Non-root container**: The `Dockerfile` creates and switches to a `helios` user — no process runs as root inside the image.
- **Password policy**: Registration enforces minimum 8 characters, at least one uppercase, one lowercase, one digit.
- **Upload hardening**: Extension allowlist (`.txt .md .pdf .csv .json .rst`), 50 MB size cap, path traversal sanitization.

---

## Critic retry loop

When the Critic agent scores an answer below the passing threshold, the pipeline automatically re-runs the Synthesizer once with the critic's improvement suggestions injected as explicit fix instructions.

### Flow

```
Planner → Retriever → Executor → Synthesizer
                                       │
                                   Critic
                                       │
                          critic_passed?
                          ├── Yes  → done
                          └── No (retry_count < 1) → Synthesizer (retry)
                                                          │
                                                       Critic
                                                          │
                                                       → done (regardless of score)
```

The retry is capped at `_MAX_RETRIES = 1` so a failing critic never creates an infinite loop. The synthesizer receives the critic's `suggestions` list on its second pass and prepends them as fix instructions before regenerating the answer.

### WebSocket events

The WebSocket client sees an additional `retrying` event between the first and second synthesis passes:

```json
{ "event": "retrying", "data": { "attempt": 2 } }
```

The frontend displays "Improving answer…" in the pipeline step indicator during the retry.

### Tuning

To change the retry cap, edit `_MAX_RETRIES` at the top of `pipeline/run.py`. Setting it to `0` disables the retry loop entirely; `2` allows two re-synthesis passes.

---

## Document chunk feedback

The Upload panel now lets you inspect exactly how an ingested document was chunked and test retrieval quality against it before using it in a conversation.

### How it works

1. After upload, each document card in the Upload panel has an expand button (chevron).
2. Clicking it calls `GET /documents/{id}/chunks?limit=10` and renders the first 5 chunk previews — showing the raw text and character count per chunk.
3. A search input lets you type any query and click the search icon (or press Enter) to call `POST /documents/{id}/search` with `{"query": "..."}`.
4. The top matching chunks are returned ranked by cosine similarity score, shown as a percentage alongside the matched text.
5. This lets you verify that a particular document will surface the right chunks for a given question before building a conversation around it.

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/documents/{id}/chunks` | Paginated chunk previews (`limit`, `offset` query params) |
| `POST` | `/documents/{id}/search` | Test retrieval — returns top-`k` chunks ranked by score |

### Response shapes

```json
// GET /documents/{id}/chunks
{
  "document_id": "uuid",
  "filename": "paper.pdf",
  "total_chunks": 42,
  "chunks": [
    { "chunk_index": 0, "text": "...", "char_count": 512 }
  ]
}

// POST /documents/{id}/search
{
  "query": "what is CARLE?",
  "results": [
    { "chunk_index": 3, "text": "...", "score": 0.91, "source": "dense" }
  ]
}
```

---

## Server-side conversation persistence

From v1.1, logged-in users have their conversation history persisted in Supabase PostgreSQL so that chats survive page reloads and are accessible from any device.

### Data model

```
conversations
  id          UUID PK
  user_id     UUID FK → users.id  (cascade delete)
  title       VARCHAR(255)
  created_at  TIMESTAMPTZ
  updated_at  TIMESTAMPTZ

conversation_messages
  id               UUID PK
  conversation_id  UUID FK → conversations.id  (cascade delete)
  role             VARCHAR(16)   -- "user" | "assistant"
  content          TEXT
  created_at       TIMESTAMPTZ
```

Both tables are indexed on their parent FK columns (`user_id`, `conversation_id`) for fast list queries.

### REST endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/conversations` | List all conversations for the authenticated user |
| `POST` | `/api/v1/conversations` | Create a new conversation (body: `{"title": "..."}`) |
| `GET` | `/api/v1/conversations/{id}` | Get conversation with all messages |
| `POST` | `/api/v1/conversations/{id}/messages` | Append a message (`role`, `content`) |
| `DELETE` | `/api/v1/conversations/{id}` | Delete conversation and all its messages |

### Frontend sync

`useConversations(isLoggedIn)` in the frontend handles bi-directional sync:

1. On login — fetches server conversations and merges with any local-only chats (local ones get a `serverId` after the first sync).
2. On `newConversation()` — calls `POST /conversations` and stores the returned `serverId`.
3. On `selectConversation()` — lazy-loads messages via `GET /conversations/{id}` when the local message list is empty, avoiding N+1 fetches on the conversation list view.
4. On every assistant turn — calls `POST /conversations/{id}/messages` to append the persisted answer.
5. On logout — clears the `syncedRef` flag so the next login re-syncs from the server.

Guest users (not logged in) continue to use a local-only in-memory conversation store with no persistence.

---

## Mobile layout

The frontend is fully usable on iOS and Android browsers from v1.1.

### What changed

| Concern | Implementation |
|---------|----------------|
| Safe-area input bar | `padding-bottom: max(0.75rem, env(safe-area-inset-bottom))` on `.chat-input-bar` |
| Bottom navigation | `MobileBottomNav` fixed bar with Home / Chat / Upload / Sign-in tabs, `sm:hidden` |
| Sidebar | Auto-collapses at `window.innerWidth < 640` on mount; re-opens on desktop resize |
| User bubble | `w-full` on `xs`, `sm:max-w-xl` on larger breakpoints |
| Step labels | Hidden on `xs` via `.step-label { display: none }` to reduce pipeline-step noise |
| Viewport | `viewport-fit=cover` in `index.html` meta tag for edge-to-edge display on notched devices |

### Testing on device

Use Chrome DevTools device emulation or run a LAN preview from the Vite dev server:

```bash
cd frontend
npm run dev -- --host
# then open http://<your-lan-ip>:5173 on your phone
```

The Vercel deployment automatically serves over HTTPS, so PWA features and safe-area insets work on live devices too.

---

## Running tests

```bash
pip install -r requirements.txt

# Full suite (no live services required — all external deps are mocked)
pytest tests/ -v

# By category
pytest tests/test_pipeline.py -v           # end-to-end pipeline
pytest tests/test_security.py -v           # brute-force, JWT, upload hardening
pytest tests/test_rate_limit_headers.py -v # rate-limit headers on all rated routes
pytest tests/test_logout.py -v             # token revocation
pytest tests/test_async_query.py -v        # WebSocket streaming + Celery path

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

All LLM calls, Redis, ChromaDB, MinIO, and Postgres connections are replaced with
`unittest.mock` mocks, so the suite runs offline with no credentials.

---

## Author

**Hridam Biswas**  
B.Tech Computer Science, KIIT University  
[hridambiswas2005@gmail.com](mailto:hridambiswas2005@gmail.com) · [GitHub](https://github.com/hridambiswas) · [LinkedIn](https://linkedin.com/in/hridambiswas)

---

*Helios is open-source under the MIT License. Contributions, bug reports, and feature requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).*
