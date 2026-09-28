#!/usr/bin/env bash
# start.sh — HF Space container entrypoint.
#
# HF Spaces run one container per Space, and only inbound port 7860 is
# exposed. This script brings up Redis (in-memory, listening on
# 127.0.0.1:6379) alongside the FastAPI process because:
#   - Rate-limit + brute-force middleware call Redis on every request.
#   - HF blocks outbound port 6379, so an external Redis (Upstash-native,
#     etc.) would not work here.
#
# Postgres is NOT started locally — SUPABASE_DATABASE_URL routes writes
# to Supabase (port 5432 IS on the HF allow-list).
#
# Trapped signals: SIGTERM propagates to both children so HF can shut the
# Space down cleanly during redeploys.

set -euo pipefail

log() { printf '\033[1;36m[start]\033[0m %s\n' "$*"; }

# Ensure the Chroma root exists and is writable — CHROMA_PATH default is
# /tmp/chroma which is always writable for UID 1000.
mkdir -p "${CHROMA_PATH:-/tmp/chroma}"

# ── Redis (bind localhost only) ────────────────────────────────────────
log "starting redis-server on 127.0.0.1:6379 (in-memory)"
redis-server \
    --bind 127.0.0.1 \
    --port 6379 \
    --daemonize no \
    --maxmemory 128mb \
    --maxmemory-policy allkeys-lru \
    --save "" \
    --appendonly no \
    --loglevel notice &
REDIS_PID=$!

cleanup() {
    log "shutdown signal received — stopping redis (pid $REDIS_PID)"
    kill -TERM "$REDIS_PID" 2>/dev/null || true
    wait "$REDIS_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

# Wait for Redis to accept connections before uvicorn starts. Without
# this, the middleware fires on the first request and 500s.
for i in 1 2 3 4 5 6 7 8 9 10; do
    if redis-cli -h 127.0.0.1 -p 6379 ping 2>/dev/null | grep -q PONG; then
        log "redis ready (attempt $i)"
        break
    fi
    log "waiting for redis (attempt $i)"
    sleep 1
done

# ── FastAPI (uvicorn) ─────────────────────────────────────────────────
log "starting uvicorn on 0.0.0.0:${APP_PORT:-7860}"
exec uvicorn main:app \
    --host "${APP_HOST:-0.0.0.0}" \
    --port "${APP_PORT:-7860}" \
    --workers 1 \
    --log-level "$(printf '%s' "${LOG_LEVEL:-INFO}" | tr '[:upper:]' '[:lower:]')"
