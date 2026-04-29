.PHONY: up down build logs test test-cov lint fmt typecheck migrate migrate-gen migrate-down dev worker beat eval bench ingest

## ── Docker ───────────────────────────────────────────────────────────────────

up:
	docker compose up --build -d

down:
	docker compose down -v

build:
	docker compose build

logs:
	docker compose logs -f api worker

## ── Tests ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

## ── Lint ─────────────────────────────────────────────────────────────────────

lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	pyright --pythonversion 3.11 agents/ api/ pipeline/ retrieval/ workers/

## ── Database ─────────────────────────────────────────────────────────────────

migrate:
	alembic upgrade head

migrate-gen:
	alembic revision --autogenerate -m "$(msg)"

migrate-down:
	alembic downgrade -1

## ── Dev ──────────────────────────────────────────────────────────────────────

dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

worker:
	celery -A workers.celery_app worker --loglevel=info

beat:
	celery -A workers.celery_app beat --loglevel=info

## ── Eval & benchmarks ────────────────────────────────────────────────────────

eval:
	python scripts/run_eval.py

bench:
	python scripts/benchmark.py --n 20 --agent all

ingest:
	python scripts/ingest_demo.py --dir ./docs
