# Helios — Distributed Multi-Modal Agentic GenAI Platform

[![Deploy Backend](https://github.com/Hridambiswas/helios/actions/workflows/deploy-backend.yml/badge.svg)](https://github.com/Hridambiswas/helios/actions/workflows/deploy-backend.yml)
[![Deploy Frontend](https://github.com/Hridambiswas/helios/actions/workflows/deploy-frontend.yml/badge.svg)](https://github.com/Hridambiswas/helios/actions/workflows/deploy-frontend.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)

> **Live Demo →** [https://frontend-omega-blush-87.vercel.app](https://frontend-omega-blush-87.vercel.app)

Helios is a full-stack agentic AI platform that answers any query by routing it through a multi-agent pipeline: planning, hybrid retrieval (local knowledge base + live web search), optional code execution, and a critic-evaluated synthesis. Built for real-world deployment on constrained hardware (EC2 t2.micro, 1 GB RAM).

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Branching Strategy](#branching-strategy)
- [Contributing](#contributing)

---

## Features

- **Multi-agent pipeline** — Planner → Retriever → Executor → Synthesizer → Critic
- **Hybrid retrieval** — Dense (ONNX FastEmbed) + BM25 sparse, merged via Reciprocal Rank Fusion
- **Live web search** — DuckDuckGo integration answers any question, even without uploaded docs
- **Source citations** — Every answer cites local documents `[D1]` and web sources `[W1]` with clickable links
- **Follow-up suggestions** — 2 contextual follow-up questions per answer, clickable in the UI
- **Conversational UI** — Markdown-rendered answers, streaming via WebSocket, animated pipeline progress
- **JWT auth** — Registration, login, refresh token rotation, direct bcrypt password hashing
- **Managed database** — Supabase PostgreSQL for user data (zero-ops, free tier)
- **Document ingestion** — Upload PDFs/text, chunk, embed, index into ChromaDB + BM25
- **Full observability** — Prometheus metrics, OpenTelemetry tracing, structured JSON logging
- **Production-ready** — Docker Compose, nginx + Let's Encrypt SSL, Celery workers, backpressure, circuit breakers, rate limiting

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Helios Platform                            │
│                                                                     │
│  ┌──────────────┐    ┌────────────────────────────────────────┐    │
│  │   Frontend   │    │            FastAPI Backend             │    │
│  │  React/Vite  │◄──►│  ┌─────────┐  ┌──────────────────┐   │    │
│  │   (Vercel)   │    │  │  Auth   │  │  LangGraph       │   │    │
│  └──────────────┘    │  │  JWT    │  │  Pipeline        │   │    │
│                      │  └─────────┘  │                  │   │    │
│                      │               │  1. Planner      │   │    │
│                      │  ┌─────────┐  │  2. Retriever ──►│Web│   │
│                      │  │ Celery  │  │  3. Executor     │   │   │
│                      │  │ Workers │  │  4. Synthesizer  │   │    │
│                      │  └─────────┘  │  5. Critic       │   │    │
│                      │               └──────────────────┘   │    │
│                      └────────────────────────────────────────┘    │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Supabase │  │  Redis   │  │ ChromaDB │  │      MinIO       │  │
│  │ Postgres │  │  Cache   │  │  Vectors │  │    Documents     │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Query Pipeline

```
User Query
    │
    ▼
┌──────────┐   ┌────────────────┐   ┌──────────┐   ┌─────────────┐   ┌────────┐
│ Planner  │──►│   Retriever    │──►│ Executor │──►│ Synthesizer │──►│ Critic │
│          │   │ Dense + BM25   │   │ (Python) │   │ + Citations │   │  Score │
│ Classify │   │ + DuckDuckGo   │   │          │   │ + Follow-ups│   │        │
└──────────┘   └────────────────┘   └──────────┘   └─────────────┘   └────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Framer Motion, react-markdown |
| **Backend** | FastAPI, Python 3.11, Uvicorn, Pydantic v2 |
| **AI / LLM** | Groq API (llama-3.3-70b-versatile), LangChain, LangGraph |
| **Embeddings** | FastEmbed ONNX — BAAI/bge-small-en-v1.5 (~120 MB, no PyTorch) |
| **Web Search** | DuckDuckGo Search (duckduckgo-search) |
| **Vector Store** | ChromaDB |
| **Database** | Supabase PostgreSQL, SQLAlchemy async, Alembic migrations |
| **Cache** | Redis |
| **Object Store** | MinIO (S3-compatible) |
| **Task Queue** | Celery + Redis broker |
| **Auth** | JWT (python-jose), bcrypt (direct, no passlib) |
| **Deployment** | Docker Compose, nginx, AWS EC2 t2.micro, Vercel |
| **CI/CD** | GitHub Actions |
| **Observability** | Prometheus, OpenTelemetry, structlog |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- [Groq API key](https://console.groq.com) (free tier available)

### 1. Clone

```bash
git clone https://github.com/Hridambiswas/helios.git
cd helios
git checkout develop
```

### 2. Configure

```bash
cp .env.example .env
# Fill in at minimum:
#   GROQ_API_KEY        — from console.groq.com
#   JWT_SECRET_KEY      — python -c "import secrets; print(secrets.token_hex(32))"
#   POSTGRES_PASSWORD   — any strong password
#   MINIO_ACCESS_KEY / MINIO_SECRET_KEY
```

### 3. Run

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Frontend | http://localhost:5173 |
| MinIO Console | http://localhost:9001 |

### 4. Run Tests

```bash
docker compose exec api pytest tests/ -v
```

---

## Environment Variables

See [`.env.example`](.env.example) for all variables with descriptions.

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | LLM inference (Groq) |
| `JWT_SECRET_KEY` | ✅ | Token signing secret |
| `POSTGRES_PASSWORD` | ✅ | Local PostgreSQL |
| `SUPABASE_DATABASE_URL` | Production | Managed DB (overrides local postgres) |
| `MINIO_ACCESS_KEY` | ✅ | Object storage |
| `CORS_ALLOWED_ORIGINS` | Production | Frontend origin(s) |

---

## API Reference

**Base URL:** `https://helios-hridam.ddns.net/api/v1`

### Authentication

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/register` | `{username, email, password}` | Create account |
| `POST` | `/auth/login` | form: `{username, password}` | Login → JWT tokens |
| `POST` | `/auth/refresh` | `{refresh_token}` | Rotate tokens |
| `POST` | `/auth/logout` | `{refresh_token}` | Revoke token |
| `GET` | `/auth/me` | — | Current user |

### Query

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/query` | `{query: string}` | Run pipeline → answer + sources + follow-ups |
| `POST` | `/query/async` | `{query: string}` | Async via Celery |
| `GET` | `/query/history` | — | Paginated history |

### Example

```bash
curl -X POST https://helios-hridam.ddns.net/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "difference between LLM and PyTorch"}'
```

```json
{
  "answer": "**LLMs** are trained models... [W1] ...\n\n---\n**Sources:**\n- [W1] [PyTorch docs](https://...)",
  "web_sources": [{"title": "...", "url": "https://...", "snippet": "..."}],
  "follow_up_questions": ["How do I fine-tune an LLM with PyTorch?", "..."],
  "critic_passed": true,
  "latency_ms": 3200
}
```

---

## Deployment

### Production Stack

- **Backend** — AWS EC2 t2.micro, Docker Compose, nginx reverse proxy, Let's Encrypt SSL
- **Frontend** — Vercel (auto-deploy on push to `main`)
- **Database** — Supabase free tier PostgreSQL (Mumbai region)

### GitHub Actions Secrets

| Secret | Description |
|--------|-------------|
| `EC2_HOST` | EC2 hostname |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | Private SSH key |
| `VERCEL_TOKEN_TWO` | Vercel deploy token |
| `VERCEL_ORG_ID` | Vercel org ID |
| `VERCEL_PROJECT_ID` | Vercel project ID |
| `SUPABASE_DATABASE_URL` | Supabase connection string |

### Deploy

Deployments are fully automated. Push to `main` triggers:
1. Backend rebuild on EC2 (Docker, ~15 min)
2. Frontend deploy to Vercel (~3 min)

---

## Branching Strategy

```
main        ← stable, production releases only
  │
  └── develop ← integration branch (base for all features)
        ├── feature/auth-hardening
        ├── feature/pipeline-improvements
        ├── feature/retrieval-web-search
        ├── feature/frontend-ux
        ├── feature/supabase-integration
        ├── feature/infra-optimization
        └── docs/project-documentation
```

**Rules:**
- `main` — merge via PR from `develop` only, no direct commits
- `develop` — integration, always deployable
- `feature/*` — one concern per branch, branched from `develop`
- `hotfix/*` — critical fixes branched from `main`, merged to both `main` and `develop`

---

## Contributing

1. Fork and clone
2. Branch from `develop`: `git checkout -b feature/your-feature develop`
3. Use [Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
4. Open a PR against `develop`

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Author

**Hridam Biswas** | B.Tech CSE @ KIIT | ML Researcher | IEEE Author

[![GitHub](https://img.shields.io/badge/GitHub-Hridambiswas-181717?logo=github)](https://github.com/Hridambiswas)

---

## License

MIT — see [LICENSE](LICENSE).
