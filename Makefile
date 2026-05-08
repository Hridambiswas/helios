.PHONY: up down build logs restart status test test-cov lint fmt typecheck \
        migrate migrate-gen migrate-down dev worker beat \
        eval bench ingest ssl-setup frontend-dev frontend-build \
        backup shell-api shell-db prod-restart deploy

## ── Docker (local dev) ────────────────────────────────────────────────────────

up:
	docker compose up --build -d

down:
	docker compose down -v

build:
	docker compose build

logs:
	docker compose logs -f api worker

restart:
	docker compose restart api worker

status:
	docker compose ps

## ── Docker (production) ──────────────────────────────────────────────────────

prod-up:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

prod-build:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build --no-cache api worker beat

prod-logs:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api worker

prod-down:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml down

## ── Tests ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -x -q

## ── Lint / Format / Type ────────────────────────────────────────────────────

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

## ── Dev servers ──────────────────────────────────────────────────────────────

dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

worker:
	celery -A workers.celery_app worker --loglevel=info --concurrency=2

beat:
	celery -A workers.celery_app beat --loglevel=info

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

## ── SSL setup (EC2 only) ─────────────────────────────────────────────────────

ssl-setup:
	bash scripts/setup_ssl.sh

## ── Eval & benchmarks ────────────────────────────────────────────────────────

eval:
	python scripts/run_eval.py

bench:
	python scripts/benchmark.py --n 20 --agent all

ingest:
	python scripts/ingest_demo.py --dir ./docs

## ── Backup ───────────────────────────────────────────────────────────────────

backup:
	bash scripts/backup.sh ./backups

## ── Shell helpers ────────────────────────────────────────────────────────────

shell-api:
	docker compose exec api /bin/bash

shell-db:
	docker compose exec postgres psql -U helios -d helios

## ── Production helpers ───────────────────────────────────────────────────────

prod-restart:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml restart api worker

deploy:
	bash scripts/deploy.sh

## ── Help ─────────────────────────────────────────────────────────────────────

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | grep -v '^.PHONY' | awk -F: '{print $$1}' | sort
