#!/usr/bin/env bash
# After first-deploy.sh succeeds, rotate GitHub Actions secrets so:
#   1. future backend deploys go to the DO droplet (not the dead EC2)
#   2. the frontend, on next rebuild, points at the DuckDNS API
#
# Run from the local repo root:
#   DO_HOST=<floating-ip> ./backend/deploy/digitalocean/rotate-secrets.sh
#
# Optional:
#   REPO             default Hridambiswas/helios
#   DUCKDNS_DOMAIN   default helios-hridam (used for VITE_API_URL host)
#
# NOTE: the deploy-backend.yml workflow uses secrets named DO_HOST, DO_USER,
# DO_SSH_KEY. DO_USER + DO_SSH_KEY are pre-set already; this script only sets
# DO_HOST + VITE_API_URL.

set -euo pipefail

: "${DO_HOST:?DO_HOST not set — pass the droplet's Floating IP}"

REPO="${REPO:-Hridambiswas/helios}"
DUCKDNS_DOMAIN="${DUCKDNS_DOMAIN:-helios-hridam}"
API_URL="https://${DUCKDNS_DOMAIN}.duckdns.org"

echo "→ setting DO_HOST = $DO_HOST"
gh secret set DO_HOST --repo "$REPO" --body "$DO_HOST"

echo "→ updating VITE_API_URL = $API_URL"
gh secret set VITE_API_URL --repo "$REPO" --body "$API_URL"

echo "→ triggering deploy-frontend workflow (fresh build with new API URL)"
gh workflow run deploy-frontend.yml --repo "$REPO" --ref main

echo
echo "secrets rotated. old EC2_* + ORACLE_* secrets are still present for"
echo "rollback; delete them once you've confirmed the browser stops showing"
echo "'request failed':"
echo "  for s in EC2_HOST EC2_USER EC2_SSH_KEY ORACLE_HOST ORACLE_USER ORACLE_SSH_KEY; do"
echo "    gh secret delete \$s --repo $REPO 2>/dev/null || true"
echo "  done"
