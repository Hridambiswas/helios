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
