# Local Development Guide

## Prerequisites

- Docker Desktop ≥ 4.x
- Python 3.11+
- Node.js 20+
- A [Groq API key](https://console.groq.com) (free tier is sufficient)

## 1 — Clone and configure

```bash
git clone https://github.com/Hridambiswas/helios.git
cd helios
cp .env.example .env
```

Open `.env` and set at minimum:

```
GROQ_API_KEY=gsk_...
SECRET_KEY=$(openssl rand -hex 32)
```

All other defaults work out of the box for local Docker.

## 2 — Start the backend

```bash
docker compose up --build -d
```

Services started: `api`, `worker`, `beat`, `postgres`, `redis`, `minio`, `chroma`.

The API is available at `http://localhost:8000`. Swagger UI: `http://localhost:8000/docs`.

## 3 — Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

## 4 — Run database migrations

Migrations run automatically on `api` startup. To run manually:

```bash
docker compose exec api alembic upgrade head
```

## 5 — Run tests

```bash
# Unit tests (no Docker needed)
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## 6 — Common commands

| Command | What it does |
|---|---|
| `make dev` | docker compose up --build |
| `make test` | pytest tests/ |
| `make lint` | ruff check . |
| `make format` | ruff format . |
| `make clean` | stop containers and remove volumes |
| `make logs` | tail api container logs |

## 7 — Environment variables

See `.env.example` for a full list. Key ones:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. LLM inference |
| `SECRET_KEY` | — | Required. JWT signing key |
| `DATABASE_URL` | `postgresql+asyncpg://helios:helios@postgres:5432/helios` | Local Postgres |
| `SUPABASE_DATABASE_URL` | `""` | Override to use Supabase |
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker + result backend |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model |
| `RETRIEVER_TOP_K` | `5` | Docs returned per retrieval |

## 8 — Adding a new agent

1. Create `agents/my_agent.py` extending `BaseAgent`
2. Implement `_run(self, state: dict) -> dict`
3. Register a node in `pipeline/run.py`
4. Add edges in `_build_graph()`
5. Add unit tests in `tests/test_my_agent_unit.py`
