#!/usr/bin/env bash
# health_check.sh — poll an HTTP URL until it returns 200, or time out.
# Usage: scripts/health_check.sh <url> [timeout_seconds] [label]
# Exit 0 once the URL returns 2xx. Exit 1 on timeout. Prints progress
# lines every attempt so the demo operator can see it waking up.

set -u

url=${1:-}
timeout=${2:-120}
label=${3:-endpoint}

if [ -z "$url" ]; then
  echo "usage: $(basename "$0") <url> [timeout_seconds] [label]" >&2
  exit 2
fi

start=$(date +%s)
attempt=0
while :; do
  attempt=$((attempt + 1))
  code=$(curl -o /dev/null -s -w "%{http_code}" --max-time 3 "$url" || echo "000")
  if [ "$code" = "200" ] || [ "$code" = "204" ]; then
    echo "$label: OK (HTTP $code) after ${attempt} attempt(s)"
    exit 0
  fi
  now=$(date +%s)
  if [ $((now - start)) -ge "$timeout" ]; then
    echo "$label: TIMEOUT after ${timeout}s (last HTTP: $code)" >&2
    exit 1
  fi
  echo "$label: waiting… (attempt ${attempt}, last HTTP: $code)"
  sleep 2
done
