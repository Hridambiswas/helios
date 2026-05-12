#!/bin/bash
set -e

# ── Swap (run once on fresh EC2 t2.micro) ────────────────────────────────────
if [ ! -f /swapfile ]; then
  echo "==> Setting up 2GB swap..."
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# ── Docker ────────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "==> Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
  echo "==> Docker installed. Re-run this script (new group needs fresh shell)."
  exit 0
fi

# ── Deploy ────────────────────────────────────────────────────────────────────
echo "==> Pulling latest code..."
git pull origin main

echo "==> Building images..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml build --pull

echo "==> Starting services..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

echo "==> Waiting for DB to be ready..."
timeout 60 bash -c 'until docker compose exec postgres pg_isready -U helios; do sleep 2; done'

echo "==> Running migrations..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec api alembic upgrade head

echo ""
echo "==> Done! Helios is live on port 80."
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
