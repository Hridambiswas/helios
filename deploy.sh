#!/bin/bash
set -e

echo "==> Pulling latest code..."
git pull origin main

echo "==> Building images..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml build --pull

echo "==> Starting services..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo "==> Waiting for DB..."
sleep 5

echo "==> Running migrations..."
docker compose exec api alembic upgrade head

echo "==> Done. Helios is live on port 80."
docker compose ps
