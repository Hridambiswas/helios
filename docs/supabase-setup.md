# Supabase Setup Guide

Helios uses Supabase PostgreSQL as the managed database in production.

## Create Project

1. Go to https://supabase.com → New Project
2. Choose region closest to your EC2 (e.g., ap-south-1 for Mumbai)

## Get Connection String

Dashboard → **Connect** → **Session mode** → copy URI.

Add `+asyncpg` to the scheme for SQLAlchemy async support:
```
postgresql+asyncpg://postgres.PROJECT_REF:PASSWORD@HOST:5432/postgres
```

## GitHub Secret

Name: `SUPABASE_DATABASE_URL`
Value: the asyncpg connection string above

The deploy workflow injects this into the EC2 `.env` file automatically.

## Auto-Migration

Alembic runs `upgrade head` on container startup — all tables are created automatically in Supabase.

## Free Tier Limits

| Resource | Limit |
|----------|-------|
| Storage | 500 MB |
| Connections | 60 (Helios uses pool_size=5) |
| Bandwidth | 5 GB/month |

## SSL

Helios automatically sets `ssl=True` in asyncpg when `SUPABASE_DATABASE_URL` is present.
