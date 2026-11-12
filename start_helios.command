#!/usr/bin/env bash
# start_helios.command — one-click local Helios demo for macOS.
#
# Double-click from Finder (or run from Terminal). Launches the Docker
# infra services (Postgres, Redis, MinIO, ChromaDB), the FastAPI API,
# the Celery worker, the Vite dev server, and `caffeinate` so the Mac
# never sleeps mid-demo. Every process writes its log and pid to
# demo/logs/<name>.{log,pid}. Polls the API health endpoint, then opens
# http://localhost:3000 in the default browser.
#
# Intentionally written as pure bash + the already-installed tooling
# (docker, python venv, node) so there is nothing to install first.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/demo/logs"
PID_DIR="$LOG_DIR"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
# The worktree reuses the main checkout's virtualenv — it holds
# gigabytes of heavy ML dependencies (torch, chromadb, sentence-transformers)
# that we don't want to install twice.
VENV_PY="/Users/hridambiswas/helios/backend/.venv/bin/python"

mkdir -p "$LOG_DIR"

echo "Helios local demo starting…"
echo "  script dir  : $SCRIPT_DIR"
echo "  log dir     : $LOG_DIR"
echo "  venv python : $VENV_PY"

# ── Pre-flight: ports must be free ────────────────────────────────────────
echo ""
echo "[1/8] Checking required ports…"
port_busy=0
for port in 3000 8000 5432 6379 9000 9001 8001; do
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

# ── Pre-flight: Docker Desktop must be running ────────────────────────────
echo ""
echo "[2/8] Checking Docker Desktop…"
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is not running. Starting it now…"
  open -a Docker
  for i in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then
      echo "Docker Desktop is up."
      break
    fi
    sleep 2
  done
  if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker Desktop did not start within 120s. Open it manually and retry."
    exit 1
  fi
else
  echo "Docker Desktop already running."
fi
