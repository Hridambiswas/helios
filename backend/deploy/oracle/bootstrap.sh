#!/usr/bin/env bash
# Bootstrap an Oracle Cloud Always-Free ARM64 Ubuntu 22.04 VM for Helios.
# Idempotent — safe to re-run.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "bootstrap.sh must be run as root (use sudo)" >&2
  exit 1
fi

REPO_URL="https://github.com/Hridambiswas/helios.git"
CHECKOUT_DIR="/home/ubuntu/helios"

log() { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }

log "starting Oracle bootstrap on $(hostname) — $(date -u +%FT%TZ)"

log "step 1/8: apt update"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

log "step 2/8: install base packages"
apt-get install -y \
  ca-certificates curl gnupg lsb-release \
  git jq netfilter-persistent iptables-persistent
