# Contributing to Helios

Thank you for your interest in contributing. This document covers everything you
need to go from zero to a merged pull request.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local setup](#local-setup)
3. [Running tests](#running-tests)
4. [Project structure](#project-structure)
5. [What needs help](#what-needs-help)
6. [Pull request process](#pull-request-process)
7. [Code style](#code-style)

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Docker & Docker Compose | v2+ |
| Make | any |
| Groq API key | required for LLM agents (free tier available at console.groq.com) |

---

## Local setup

```bash
# 1. Fork & clone
git clone https://github.com/<your-fork>/helios.git
cd helios

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in environment variables
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY

# 5. Start all backing services (Postgres, Redis, ChromaDB, MinIO)
docker compose up -d postgres redis chroma minio

# 6. Run database migrations
make migrate

# 7. Start the API server
make dev
# → FastAPI running at http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

To also run Celery workers:

```bash
make worker       # starts the Celery worker pool
make beat         # starts the Celery beat scheduler (periodic tasks)
```

---

## Running tests

```bash
# All tests (unit + integration, requires no external services — uses mocks)
make test

# With coverage report
make test-cov

# Single file
pytest tests/test_pipeline.py -v
```

The test suite mocks all LLM calls and external services, so you do not need
real API keys or running containers to run tests.

---

## Project structure

```
helios/
├── agents/          # Five LangGraph agents (planner, retriever, executor, synthesizer, critic)
├── api/             # FastAPI routes, WebSocket, auth middleware, schemas
├── eval/            # Offline evaluation harness + LLM-as-judge scorers
├── frontend/        # React + Vite chat UI (deployed to Vercel)
├── graphs/          # Visualisation scripts for performance graphs (run standalone)
├── observability/   # OpenTelemetry tracing, Prometheus metrics, structured logging
├── pipeline/        # LangGraph StateGraph assembly + Redis checkpointing
├── retrieval/       # Hybrid retrieval: dense vectors, CLIP, BM25
├── scripts/         # Dev utilities (ingest demo, benchmark, migrations)
├── storage/         # SQLAlchemy models, async CRUD, Redis cache, MinIO object store
├── tests/           # Pytest test suite
└── workers/         # Celery app, task definitions, beat schedule
```

---

## What needs help

Look for issues labelled:

| Label | Meaning |
|-------|---------|
| `good first issue` | Small, well-scoped — great starting point |
| `help wanted` | Bigger tasks where extra hands are welcome |
| `bug` | Confirmed defects |
| `performance` | Latency or throughput improvements |

High-priority areas right now:

- **Eval coverage** — `eval/` has a 30-question harness; more diverse questions
  (multi-hop reasoning, table QA, code debugging) improve confidence in retrieval
  and generation quality.
- **Docker image size** — the current image bundles all ML model weights at build
  time; a multi-stage build with lazy model download would reduce the compressed
  image size significantly.
- **Conversation search** — full-text search across persisted conversations so
  users can find a past discussion by keyword without scrolling.
- **Streaming progress on retry** — when the critic triggers a second synthesis
  pass the `retrying` event is sent but the token stream restarts from zero;
  smoothly transitioning the streamed text in the UI would improve UX.
- **Multi-doc retrieval** — the `query_by_doc` helper scopes retrieval to a single
  document; a ranked merge across multiple documents would support "compare these
  two papers" queries.
- **PWA manifest** — adding `manifest.json` and a service-worker would let mobile
  users install the app and cache recent conversations offline.

---

## Pull request process

1. **Open an issue first** for any non-trivial change so we can align on approach
   before you invest time writing code.
2. **Branch naming** — `feat/<short-description>`, `fix/<short-description>`, or
   `chore/<short-description>`.
3. **Commits** — use conventional commits (`feat:`, `fix:`, `chore:`, `docs:`).
4. **Tests** — add or update tests for every behaviour change. PRs that drop
   coverage will not be merged.
5. **One concern per PR** — keep PRs focused. A PR that fixes a bug and refactors
   an unrelated module is harder to review and slower to merge.
6. Fill in the PR template fully — especially the *Test plan* section.

---

## Code style

```bash
# Lint (ruff)
make lint

# Format
make fmt

# Type-check
make typecheck
```

- Python 3.11+ syntax throughout (use `X | Y` unions, `match`, `tomllib`, etc.)
- All public functions need a one-line docstring.
- No `print()` in library code — use `logging`.
- Keep imports grouped: stdlib → third-party → local, separated by blank lines.
