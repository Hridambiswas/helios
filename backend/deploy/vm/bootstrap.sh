#!/usr/bin/env bash
# Bootstrap a stock Ubuntu 22.04 x86_64 VM for Helios.
# Idempotent — safe to re-run.
#
# Designed for cloud VMs whose SSH login is a sudo-enabled non-root user
# (Azure → azureuser, GCP → ubuntu, Oracle Cloud → ubuntu). Escalates
# through sudo instead of requiring direct root SSH, which Azure disables
# by default.
#
# Env vars (all optional):
#   REPO_URL       upstream repo (default: https://github.com/Hridambiswas/helios.git)
#   DEPLOY_BRANCH  branch to check out (default: fix/request-failed)
#   CHECKOUT_DIR   where to clone (default: $HOME/helios)
#
# Usage on the VM:
#   bash bootstrap.sh              # never as root, never with sudo bash

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Hridambiswas/helios.git}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-fix/request-failed}"
CHECKOUT_DIR="${CHECKOUT_DIR:-$HOME/helios}"

log() { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then
  echo "bootstrap.sh: do NOT run as root — run as the VM SSH user (e.g. azureuser)." >&2
  echo "It will sudo where needed." >&2
  exit 1
fi

if ! sudo -n true 2>/dev/null; then
  log "priming sudo (you may be prompted for a password)"
  sudo -v
fi

log "starting VM bootstrap on $(hostname) — $(date -u +%FT%TZ) — user=$USER"

log "step 1/6: apt update + base packages"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y \
  ca-certificates curl gnupg lsb-release \
  git jq ufw

log "step 2/6: add Docker apt repo"
sudo install -m 0755 -d /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update -y

log "step 3/6: install docker-ce + compose plugin"
sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
  sudo usermod -aG docker "$USER"
  log "  → added $USER to docker group (new shells will pick it up)"
fi

log "step 4/6: ufw allow 22, 80, 443"
sudo ufw --force default deny incoming
sudo ufw --force default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

log "step 5/6: install certbot (snap route — matches Let's Encrypt docs)"
if ! command -v certbot >/dev/null 2>&1; then
  sudo apt-get install -y snapd
  sudo snap install core
  sudo snap refresh core
  sudo snap install --classic certbot
  sudo ln -sf /snap/bin/certbot /usr/bin/certbot
fi

log "step 6/6: clone repo on branch '$DEPLOY_BRANCH'"
if [ ! -d "$CHECKOUT_DIR/.git" ]; then
  git clone --branch "$DEPLOY_BRANCH" "$REPO_URL" "$CHECKOUT_DIR"
else
  log "  → repo already present, syncing $CHECKOUT_DIR to origin/$DEPLOY_BRANCH"
  git -C "$CHECKOUT_DIR" fetch origin "$DEPLOY_BRANCH"
  git -C "$CHECKOUT_DIR" checkout "$DEPLOY_BRANCH"
  git -C "$CHECKOUT_DIR" reset --hard "origin/$DEPLOY_BRANCH"
fi

log "bootstrap complete."
cat <<EOF

next steps (done for you by first-deploy.sh):
  1. populate  ${CHECKOUT_DIR}/backend/.env
  2. populate  /etc/helios/duckdns.env
  3. run       sudo bash ${CHECKOUT_DIR}/backend/deploy/duckdns/install.sh
  4. wait for DNS, then run certbot standalone for the DuckDNS host
  5. cd ${CHECKOUT_DIR}/backend && make prod-up
EOF
