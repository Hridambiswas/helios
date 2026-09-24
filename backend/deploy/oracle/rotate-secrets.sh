#!/usr/bin/env bash
# After first-deploy.sh succeeds, rotate the GitHub Actions secrets so:
#   1. future backend deploys go to Oracle (not the dead EC2)
#   2. the frontend, on next rebuild, points at the DuckDNS API
#
# Run from the local repo root:
#   ORACLE_HOST=<ip> ./backend/deploy/oracle/rotate-secrets.sh
#
# Optional:
#   REPO             default Hridambiswas/helios
#   DUCKDNS_DOMAIN   default helios-hridam (used for VITE_API_URL host)

set -euo pipefail

: "${ORACLE_HOST:?ORACLE_HOST not set — pass the VM's reserved public IP}"

REPO="${REPO:-Hridambiswas/helios}"
DUCKDNS_DOMAIN="${DUCKDNS_DOMAIN:-helios-hridam}"
API_URL="https://${DUCKDNS_DOMAIN}.duckdns.org"

echo "→ setting ORACLE_HOST = $ORACLE_HOST"
gh secret set ORACLE_HOST --repo "$REPO" --body "$ORACLE_HOST"

echo "→ updating VITE_API_URL = $API_URL"
gh secret set VITE_API_URL --repo "$REPO" --body "$API_URL"

echo "→ triggering deploy-frontend workflow (fresh build with new API URL)"
gh workflow run deploy-frontend.yml --repo "$REPO" --ref main

echo
echo "secrets rotated. old EC2_* secrets are still present for rollback;"
echo "delete them once you've confirmed the browser stops showing 'request failed':"
echo "  gh secret delete EC2_HOST --repo $REPO"
echo "  gh secret delete EC2_USER --repo $REPO"
echo "  gh secret delete EC2_SSH_KEY --repo $REPO"
