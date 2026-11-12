#!/usr/bin/env bash
# start_helios.command — one-click local Helios demo for macOS.
#
# Double-click from Finder (or run from Terminal). Launches the native
# infra services (Postgres 16 + Redis via brew services, MinIO on :9000,
# ChromaDB on :8001), the FastAPI API, the Celery worker, the Vite dev
# server, and `caffeinate` so the Mac never sleeps mid-demo. Every
# process writes its log and pid to demo/logs/<name>.{log,pid}. Polls
# the API health endpoint, then opens http://localhost:3000 in the
# default browser.
#
# No Docker. Uses brew services + the main checkout's python venv +
# a local node_modules in the worktree.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/demo/logs"
PID_DIR="$LOG_DIR"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
MINIO_DATA_DIR="$SCRIPT_DIR/demo/minio-data"
CHROMA_DATA_DIR="$SCRIPT_DIR/demo/chroma-data"
# The worktree reuses the main checkout's virtualenv — it holds
# gigabytes of heavy ML dependencies (torch, chromadb, sentence-transformers)
# that we don't want to install twice.
VENV_PY="/Users/hridambiswas/helios/backend/.venv/bin/python"
VENV_BIN="/Users/hridambiswas/helios/backend/.venv/bin"

mkdir -p "$LOG_DIR" "$MINIO_DATA_DIR" "$CHROMA_DATA_DIR"

echo "Helios local demo starting…"
echo "  script dir  : $SCRIPT_DIR"
echo "  log dir     : $LOG_DIR"
echo "  venv python : $VENV_PY"

# ── Pre-flight: ports must be free ────────────────────────────────────────
echo ""
echo "[1/8] Checking required ports…"
# Only the ports the demo itself will *open*. 5432 (Postgres) and 6379
# (Redis) are served by brew services that are already listening — we
# WANT them to be busy, and we check them below.
port_busy=0
for port in 3000 8000 9000 9001 8001; do
  if ! "$SCRIPT_DIR/scripts/port_check.sh" "$port"; then
    port_busy=1
  fi
done
if [ "$port_busy" -ne 0 ]; then
  echo ""
  echo "❌ One or more ports are busy. Run ./stop_helios.command or quit"
  echo "   whatever is using them, then try again."
  exit 1
fi

# ── Pre-flight: brew services (Postgres + Redis) ──────────────────────────
echo ""
echo "[2/8] Checking brew services (Postgres 16, Redis)…"
if ! command -v brew >/dev/null 2>&1; then
  echo "❌ Homebrew is not installed. Install from https://brew.sh and retry."
  exit 1
fi

brew_svc_status() {
  # Prints "started" / "stopped" / "none" for a service name.
  brew services list | awk -v s="$1" '$1==s {print $2}'
}

for svc in postgresql@16 redis; do
  st="$(brew_svc_status "$svc")"
  if [ "$st" != "started" ]; then
    echo "  starting $svc (was: ${st:-missing})…"
    brew services start "$svc" >/dev/null
  else
    echo "  $svc already started."
  fi
done

# Wait until both services actually accept connections.
echo "  waiting for Postgres (5432)…"
for i in $(seq 1 60); do
  if (echo > /dev/tcp/127.0.0.1/5432) >/dev/null 2>&1; then
    echo "  Postgres is up."
    break
  fi
  sleep 1
done
echo "  waiting for Redis (6379)…"
for i in $(seq 1 60); do
  if (echo > /dev/tcp/127.0.0.1/6379) >/dev/null 2>&1; then
    echo "  Redis is up."
    break
  fi
  sleep 1
done

# ── MinIO ────────────────────────────────────────────────────────────────
echo ""
echo "[3/8] Starting MinIO on :9000…"
if ! command -v minio >/dev/null 2>&1; then
  echo "❌ minio CLI not found. Run: brew install minio"
  exit 1
fi
export MINIO_ROOT_USER="minio_local_dev"
export MINIO_ROOT_PASSWORD="minio_local_dev_secret"
nohup minio server --address ":9000" --console-address ":9001" \
  "$MINIO_DATA_DIR" > "$LOG_DIR/minio.log" 2>&1 &
echo $! > "$PID_DIR/minio.pid"
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:9000/minio/health/ready >/dev/null 2>&1; then
    echo "  MinIO ready (pid $(cat "$PID_DIR/minio.pid"))."
    break
  fi
  sleep 1
done
if ! curl -sf http://127.0.0.1:9000/minio/health/ready >/dev/null 2>&1; then
  echo "❌ MinIO did not become ready in 60s. See $LOG_DIR/minio.log"
  exit 1
fi

# ── ChromaDB ─────────────────────────────────────────────────────────────
echo ""
echo "[4/8] Starting ChromaDB on :8001…"
if [ ! -x "$VENV_BIN/chroma" ]; then
  echo "❌ chroma CLI not found in venv. Run: $VENV_BIN/pip install chromadb"
  exit 1
fi
cd "$BACKEND_DIR"
nohup "$VENV_BIN/chroma" run --host 127.0.0.1 --port 8001 \
  --path "$CHROMA_DATA_DIR" > "$LOG_DIR/chroma.log" 2>&1 &
echo $! > "$PID_DIR/chroma.pid"
cd "$SCRIPT_DIR"
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8001/api/v2/heartbeat >/dev/null 2>&1 \
     || curl -sf http://127.0.0.1:8001/api/v1/heartbeat >/dev/null 2>&1; then
    echo "  ChromaDB ready (pid $(cat "$PID_DIR/chroma.pid"))."
    break
  fi
  sleep 1
done
if ! (curl -sf http://127.0.0.1:8001/api/v2/heartbeat >/dev/null 2>&1 \
      || curl -sf http://127.0.0.1:8001/api/v1/heartbeat >/dev/null 2>&1); then
  echo "❌ ChromaDB did not become ready in 60s. See $LOG_DIR/chroma.log"
  exit 1
fi

# ── Backend env file ─────────────────────────────────────────────────────
echo ""
echo "[5/8] Preparing backend/.env…"
if [ ! -f "$BACKEND_DIR/.env" ]; then
  if [ ! -f "$BACKEND_DIR/.env.local" ]; then
    echo "❌ No backend/.env.local found. Copy backend/.env.local.example,"
    echo "   paste your Groq and Gemini keys, and retry."
    exit 1
  fi
  # Prepend a marker line so switch_to_production.command can tell the
  # demo-created file apart from one the user wrote by hand.
  {
    echo "# created-by: start_helios.command"
    cat "$BACKEND_DIR/.env.local"
  } > "$BACKEND_DIR/.env"
  echo "  copied backend/.env.local -> backend/.env (marked)"
else
  echo "  backend/.env already present — leaving it alone."
fi

# ── Caffeinate so the Mac doesn't sleep during the demo ──────────────────
echo ""
echo "[6/8] Starting caffeinate (keep Mac awake)…"
nohup caffeinate -dims > "$LOG_DIR/caffeinate.log" 2>&1 &
echo $! > "$PID_DIR/caffeinate.pid"
echo "  caffeinate pid $(cat "$PID_DIR/caffeinate.pid")."

# ── API + Celery worker ──────────────────────────────────────────────────
echo ""
echo "[7/8] Starting API (uvicorn :8000) and Celery worker…"
cd "$BACKEND_DIR"
nohup "$VENV_BIN/uvicorn" main:app --host 0.0.0.0 --port 8000 \
  > "$LOG_DIR/api.log" 2>&1 &
echo $! > "$PID_DIR/api.pid"
nohup "$VENV_BIN/celery" -A workers.celery_app worker --loglevel=info \
  > "$LOG_DIR/worker.log" 2>&1 &
echo $! > "$PID_DIR/worker.pid"
cd "$SCRIPT_DIR"

# Poll /health for up to 120s.
echo "  waiting for http://localhost:8000/api/v1/health …"
api_ok=0
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then
    api_ok=1
    echo "  API healthy after ${i}s (pid $(cat "$PID_DIR/api.pid"))."
    break
  fi
  sleep 1
done
if [ "$api_ok" -ne 1 ]; then
  echo "❌ API did not become healthy in 120s. See $LOG_DIR/api.log"
  exit 1
fi

# ── Frontend ─────────────────────────────────────────────────────────────
echo ""
echo "[8/8] Starting Vite dev server on :3000…"
cd "$FRONTEND_DIR"
if [ ! -d node_modules ]; then
  echo "  running npm ci (first run)…"
  npm ci --silent >> "$LOG_DIR/frontend-install.log" 2>&1
fi
nohup npm run dev -- --host 127.0.0.1 --port 3000 --strictPort \
  > "$LOG_DIR/frontend.log" 2>&1 &
echo $! > "$PID_DIR/frontend.pid"
cd "$SCRIPT_DIR"

# Wait for Vite to accept connections.
for i in $(seq 1 60); do
  if (echo > /dev/tcp/127.0.0.1/3000) >/dev/null 2>&1; then
    echo "  Vite ready after ${i}s (pid $(cat "$PID_DIR/frontend.pid"))."
    break
  fi
  sleep 1
done

echo ""
echo "✅ Helios is running."
echo "    http://localhost:3000   (frontend)"
echo "    http://localhost:8000   (API, /api/v1/health /docs)"
echo "    Logs: $LOG_DIR"
echo "    Stop with ./stop_helios.command"

open http://localhost:3000 || true
