#!/usr/bin/env bash
# port_check.sh — exit 0 if the given TCP port is free, 1 if it is in use.
# Usage: scripts/port_check.sh <port>
# Prints a short, human-readable message on both outcomes so the start
# script can show the professor what is wrong.

set -u

if [ $# -ne 1 ]; then
  echo "usage: $(basename "$0") <port>" >&2
  exit 2
fi

port="$1"

if ! [[ "$port" =~ ^[0-9]+$ ]]; then
  echo "port must be an integer, got: $port" >&2
  exit 2
fi

# lsof is on macOS by default. -nP keeps it fast (no DNS / name lookups).
if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
  owner=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null \
    | awk 'NR==2 {print $1" (pid "$2")"}')
  echo "port $port is busy: $owner"
  exit 1
fi

echo "port $port is free"
exit 0
