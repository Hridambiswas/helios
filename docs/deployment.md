# Deployment Guide

## Production Stack

nginx (SSL) → FastAPI → Celery → Redis/Postgres/MinIO/ChromaDB

## EC2 Initial Setup

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu
git clone https://github.com/Hridambiswas/helios.git && cd helios
cp .env.example .env && nano .env
sudo certbot certonly --standalone -d your-domain.com
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## CI/CD Flow

Push to `main` triggers GitHub Actions which:
1. `docker system prune -a -f` (free disk)
2. `git pull origin main`
3. Rebuild + restart API, worker, beat
4. Inject `SUPABASE_DATABASE_URL` from secrets

## Memory Optimization on t2.micro (1 GB)

- FastEmbed ONNX instead of PyTorch (saves ~800 MB)
- Lazy agent init — models load on first query, not startup
- mem_limit per service in docker-compose.prod.yml
- Single uvicorn worker

## Required GitHub Secrets

EC2_HOST, EC2_USER, EC2_SSH_KEY, VERCEL_TOKEN_TWO, VERCEL_ORG_ID, VERCEL_PROJECT_ID, SUPABASE_DATABASE_URL
