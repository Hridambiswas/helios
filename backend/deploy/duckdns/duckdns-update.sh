#!/usr/bin/env bash
# Update the DuckDNS A record for $DUCKDNS_DOMAIN to point at this host.
# DuckDNS auto-detects the source IP when the `ip=` query param is empty.

set -euo pipefail

: "${DUCKDNS_DOMAIN:?DUCKDNS_DOMAIN not set}"
: "${DUCKDNS_TOKEN:?DUCKDNS_TOKEN not set}"

resp=$(curl -fsSL "https://www.duckdns.org/update?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=")

case "$resp" in
  OK)   echo "duckdns: OK ($(date -u +%FT%TZ))" ;;
  KO)   echo "duckdns: rejected — check DUCKDNS_TOKEN / DUCKDNS_DOMAIN" >&2; exit 1 ;;
  *)    echo "duckdns: unexpected response: $resp" >&2; exit 1 ;;
esac
