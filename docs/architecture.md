# Helios Architecture

## High-level overview

```
Browser (Vercel)
      │  HTTPS
      ▼
  nginx (EC2)
      │
      ▼
  FastAPI (uvicorn)
      │
      ├─► Celery worker  ──► Redis (task queue)
      │
      └─► LangGraph Pipeline
              │
              ├─ Planner  (Groq llama-3.3-70b)
              ├─ Retriever
              │    ├─ FastEmbed dense  ──► ChromaDB
              │    ├─ BM25 sparse
              │    └─ DuckDuckGo web search
              ├─ Executor  (sandboxed code runner)
              ├─ Synthesizer (Groq llama-3.3-70b)
              └─ Critic     (Groq llama-3.3-70b)
```

## Layers

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React 18 + Vite + Tailwind | UI, auth, query interface |
| Gateway | nginx (alpine) | TLS termination, rate limiting, reverse proxy |
| API | FastAPI + uvicorn | REST endpoints, auth, pipeline orchestration |
| Pipeline | LangGraph 0.2 | DAG-based multi-agent execution |
| LLM | Groq API (llama-3.3-70b-versatile) | Planning, synthesis, critique |
| Embeddings | FastEmbed ONNX (BAAI/bge-small-en-v1.5) | Dense retrieval vectors |
| Vector DB | ChromaDB | Dense similarity search |
| BM25 | rank-bm25 | Sparse keyword retrieval |
| Web search | DuckDuckGo (duckduckgo-search) | Live internet results |
| Task queue | Celery + Redis | Async document ingestion |
| Primary DB | Supabase PostgreSQL (asyncpg) | Users, queries, documents |
| Object store | MinIO | Raw uploaded files |
| Observability | OpenTelemetry + Prometheus | Traces, metrics |
| Deployment | Docker Compose on EC2 t2.micro | Single-node production |

## Pipeline DAG

```
       ┌─────────┐
       │ Planner │
       └────┬────┘
            │ route_after_planner
    ┌───────┼──────────┐
    │       │          │
    ▼       ▼          ▼
retriever  executor  synthesizer
    │       │          ▲
    │       └──────────┤
    │ route_after_retriever
    ├───► executor
    └───► synthesizer
              │
              ▼
           critic
              │
             END
```

## Memory budget (EC2 t2.micro — 1 GB)

| Service | Limit |
|---|---|
| api | 512 MB |
| worker | 200 MB |
| nginx | 64 MB |
| redis | 80 MB |
| minio | 96 MB |
| chroma | 160 MB |
| postgres (local fallback) | 128 MB |

## Security

- JWT (HS256) with 30-min access + 7-day refresh tokens
- bcrypt password hashing (cost factor 12)
- All traffic over TLS 1.2/1.3 (Let's Encrypt)
- `/metrics` endpoint blocked at nginx
- Auth endpoints rate-limited to 10 req/min
- Query endpoint rate-limited to 6 req/min
