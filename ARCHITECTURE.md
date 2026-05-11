# Helios — System Architecture

## Overview

Helios uses a **LangGraph state machine** to orchestrate five specialized agents. Each agent reads from and writes to a shared typed state (`HeliosState`), enabling conditional branching without tight coupling.

## Agent Pipeline

```
Query Input
    │
    ▼
┌───────────────────────────────────────────────┐
│                  PlannerAgent                 │
│  - Classifies query: factual/analytical/code  │
│  - Sets requires_retrieval, requires_code     │
│  - Returns structured JSON plan               │
└───────────────┬───────────────┬───────────────┘
                │               │
         retrieval?           code only?
                │               │
                ▼               ▼
┌──────────────────┐    ┌──────────────┐
│  RetrieverAgent  │    │ ExecutorAgent│
│  - Dense (ONNX)  │    │ - Sandboxed  │
│  - BM25 sparse   │    │   Python     │
│  - DuckDuckGo    │    │   execution  │
│  - RRF merge     │    └──────┬───────┘
└──────┬───────────┘           │
       │                       │
       └──────────┬────────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │     SynthesizerAgent    │
    │  - Cites [D1],[W1] etc. │
    │  - Markdown answer      │
    │  - 2 follow-up Qs       │
    └─────────────┬───────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │       CriticAgent       │
    │  - Scores: groundedness │
    │    faithfulness,        │
    │    completeness         │
    │  - pass/fail threshold  │
    └─────────────────────────┘
```

## Retrieval Strategy

```
Query
  │
  ├──► Dense Path: FastEmbed (ONNX) → ChromaDB vector search
  │
  ├──► Sparse Path: BM25 index search
  │
  └──► Web Path: DuckDuckGo → [{title, url, snippet}]
       (always runs in parallel)

All three results → Reciprocal Rank Fusion (RRF) → Top-K merged docs
```

**Why ONNX/FastEmbed over PyTorch?**  
The EC2 t2.micro has 1 GB RAM. PyTorch base image is ~1.5 GB; CLIP model is ~600 MB. FastEmbed uses ONNX Runtime — the BAAI/bge-small-en-v1.5 model is ~120 MB and runs inference in ~20 ms.

## Database Architecture

```
┌─────────────────────────────────────────────────────┐
│  Supabase PostgreSQL (managed, ap-south-1 region)   │
│    - users                                          │
│    - refresh_tokens                                 │
│    - query_records                                  │
│    - documents (metadata)                           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Redis (EC2 local)                                  │
│    - Query result cache (5-min TTL)                 │
│    - Celery broker (async tasks)                    │
│    - Rate limit counters                            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  ChromaDB (EC2 local)                               │
│    - Document chunk embeddings (384-dim vectors)    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  MinIO (EC2 local, S3-compatible)                   │
│    - Raw uploaded document files                    │
└─────────────────────────────────────────────────────┘
```

## API Architecture

```
Internet
    │
    ▼
nginx (SSL termination, Let's Encrypt)
    │
    ▼
FastAPI (uvicorn, 1 worker)
    ├── /api/v1/auth/*    — JWT auth endpoints
    ├── /api/v1/query     — Sync pipeline execution
    ├── /api/v1/query/async — Celery task queue
    ├── /api/v1/ingest    — Document upload + indexing
    ├── /api/v1/health    — Health check
    └── /metrics          — Prometheus metrics
    │
    ├── WebSocket /ws/query/{user_id}  — Streaming pipeline events
    │
    └── Middleware stack:
        SecurityHeadersMiddleware
        AuthBruteForceMiddleware
        RequestIDMiddleware
        RateLimitMiddleware
        GatewayMiddleware (canary routing)
```

## Memory Budget (EC2 t2.micro — 1 GB RAM)

| Service | Limit | Actual |
|---------|-------|--------|
| API | 512 MB | ~300 MB |
| Celery Worker | 200 MB | ~150 MB |
| ChromaDB | 160 MB | ~80 MB |
| Redis | 80 MB | ~30 MB |
| MinIO | 96 MB | ~60 MB |
| Nginx | 64 MB | ~10 MB |
| Celery Beat | 64 MB | ~30 MB |
| PostgreSQL | 128 MB | ~80 MB |
| **Total** | **1304 MB** | **~740 MB** |

The API container stays under 512 MB because:
- ONNX runtime instead of PyTorch (saves ~800 MB)
- Lazy agent initialization (models load on first query, not startup)
- Single uvicorn worker (no per-process duplication)
