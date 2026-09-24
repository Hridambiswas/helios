#!/usr/bin/env bash
# Quick post-deploy check — run from your laptop after DNS + SSL are up.
# Fails loud if any of the critical endpoints don't respond as expected.

set -euo pipefail

HOST="${1:-helios-hridam.duckdns.org}"
BASE="https://${HOST}"

check() {
  local label="$1" url="$2" expect="$3"
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' -m 10 "$url" || echo "000")
  printf '  %-30s %s → HTTP %s' "$label" "$url" "$code"
  if [ "$code" = "$expect" ]; then
    printf '  \033[32mOK\033[0m\n'
  else
    printf '  \033[31mFAIL (expected %s)\033[0m\n' "$expect"
    exit 1
  fi
}

echo "verifying $BASE"
check "nginx health"          "$BASE/nginx-health"           200
check "API root"              "$BASE/api/v1/stats"           200
check "docs (public)"         "$BASE/docs"                   200
check "metrics (should 403)"  "$BASE/metrics"                403

echo "all checks passed"
