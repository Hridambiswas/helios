#!/usr/bin/env bash
# After first-deploy.sh succeeds, rotate GitHub Actions secrets so:
#   1. future backend deploys reach the VM (VM_HOST / VM_USER / VM_SSH_KEY)
#   2. the frontend, on next rebuild, points at the DuckDNS API URL
#
# Run from the local repo root:
#   VM_HOST=<static-ip> ./backend/deploy/vm/rotate-secrets.sh
#
# Optional:
#   REPO             default Hridambiswas/helios
#   DUCKDNS_DOMAIN   default helios-hridam (used for VITE_API_URL host)
#   VM_USER          if set, updates VM_USER secret (default: azureuser)
#
# NOTE: VM_SSH_KEY must be set manually via
#   gh secret set VM_SSH_KEY --repo Hridambiswas/helios < path/to/private_key
# so its content is never printed to the terminal by this script.

set -euo pipefail

: "${VM_HOST:?VM_HOST not set - pass the VM static public IP}"

REPO="${REPO:-Hridambiswas/helios}"
DUCKDNS_DOMAIN="${DUCKDNS_DOMAIN:-helios-hridam}"
VM_USER="${VM_USER:-azureuser}"
API_URL="https://${DUCKDNS_DOMAIN}.duckdns.org"

echo "→ setting VM_HOST = $VM_HOST"
gh secret set VM_HOST --repo "$REPO" --body "$VM_HOST"

echo "→ setting VM_USER = $VM_USER"
gh secret set VM_USER --repo "$REPO" --body "$VM_USER"

echo "→ updating VITE_API_URL = $API_URL"
gh secret set VITE_API_URL --repo "$REPO" --body "$API_URL"

echo "→ triggering deploy-frontend workflow on fix/request-failed"
gh workflow run deploy-frontend.yml --repo "$REPO" --ref fix/request-failed

echo
echo "VM_SSH_KEY was NOT touched by this script. If you haven't set it yet:"
echo "  gh secret set VM_SSH_KEY --repo $REPO < /path/to/ssh/private_key"
echo
echo "Old DO_* / EC2_* / ORACLE_* secrets are still present for rollback."
echo "Once /health is green through the browser, delete them:"
echo "  for s in DO_HOST DO_USER DO_SSH_KEY EC2_HOST EC2_USER EC2_SSH_KEY \\"
echo "          ORACLE_HOST ORACLE_USER ORACLE_SSH_KEY; do"
echo "    gh secret delete \$s --repo $REPO 2>/dev/null || true"
echo "  done"
