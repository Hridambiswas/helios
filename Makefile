.PHONY: dev build up down logs test lint format clean

# ── Development ───────────────────────────────────────────────────────────────

dev:
	docker compose up --build

build:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build api worker beat

up:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

down:
	docker compose down

logs:
	docker compose logs -f api

# ── Quality ───────────────────────────────────────────────────────────────────

test:
	docker compose exec api pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

# ── Maintenance ───────────────────────────────────────────────────────────────

clean:
	docker system prune -f
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

migrate:
	docker compose exec api alembic upgrade head

shell:
	docker compose exec api python3 -i -c "from config import cfg; print('Helios shell')"
