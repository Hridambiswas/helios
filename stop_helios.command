#!/usr/bin/env bash
# stop_helios.command — stops everything start_helios.command launched.
#
# Kills the processes recorded in demo/logs/*.pid (minio, chroma, api,
# worker, frontend, caffeinate), then verifies the ports are free. Leaves
# the brew-services (postgresql@16, redis) running — those are long-lived
# and shared with other local work; start_helios will just reconnect to
# them next time.

set -u  # not -e: we want to continue even if a kill fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/demo/logs"

echo "Helios local demo stopping…"

stop_pid_file() {
  local name="$1"
  local pid_file="$LOG_DIR/${name}.pid"
  if [ ! -f "$pid_file" ]; then
    echo "  $name: no pid file"
    return
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "  $name: pid $pid not running"
    rm -f "$pid_file"
    return
  fi
  echo "  $name: killing pid $pid …"
  kill "$pid" 2>/dev/null || true
  for i in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "  $name: still alive, SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

# Stop in reverse-of-start order so children exit before their parents.
stop_pid_file frontend
stop_pid_file worker
stop_pid_file api
stop_pid_file caffeinate
stop_pid_file chroma
stop_pid_file minio

# Sometimes `npm run dev` spawns a child Vite that outlives npm.
# Catch any stray node pointed at port 3000.
stray_vite="$(lsof -ti tcp:3000 2>/dev/null || true)"
if [ -n "$stray_vite" ]; then
  echo "  stray vite on :3000 — killing $stray_vite"
  kill $stray_vite 2>/dev/null || true
fi

echo ""
echo "Port status after stop:"
for port in 3000 8000 9000 9001 8001; do
  if (echo > /dev/tcp/127.0.0.1/$port) >/dev/null 2>&1; then
    echo "  :$port still has a listener"
  else
    echo "  :$port free"
  fi
done

echo ""
echo "✓ Stopped demo processes."
echo "  Postgres and Redis (brew services) are left running on purpose."
echo "  Stop them with 'brew services stop postgresql@16 redis' if you want."
