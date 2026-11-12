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
