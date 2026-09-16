#!/usr/bin/env bash
# Bootstrap a DigitalOcean Ubuntu 22.04 droplet for Helios.
# Idempotent — safe to re-run.
#
# DO's stock Ubuntu image ships with:
#   - ufw disabled (no Oracle-style iptables surprise)
#   - only the `root` user by default
#
# This script creates a `helios` user + installs everything.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "bootstrap.sh must be run as root" >&2
  exit 1
fi

REPO_URL="https://github.com/Hridambiswas/helios.git"
DEPLOY_USER="helios"
CHECKOUT_DIR="/home/${DEPLOY_USER}/helios"

log() { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }

log "starting DigitalOcean bootstrap on $(hostname) — $(date -u +%FT%TZ)"

log "step 1/7: apt update + base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y
apt-get install -y \
  ca-certificates curl gnupg lsb-release \
  git jq ufw

log "step 2/7: add Docker apt repo"
install -m 0755 -d /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
fi
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -y

log "step 3/7: install docker-ce + compose plugin"
apt-get install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker

log "step 4/7: create ${DEPLOY_USER} user + copy root's SSH keys"
if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
  adduser --disabled-password --gecos "" "$DEPLOY_USER"
  usermod -aG docker "$DEPLOY_USER"
  # Give the helios user the same SSH keys the droplet was created with.
  mkdir -p "/home/${DEPLOY_USER}/.ssh"
  cp -a /root/.ssh/authorized_keys "/home/${DEPLOY_USER}/.ssh/authorized_keys"
  chown -R "${DEPLOY_USER}:${DEPLOY_USER}" "/home/${DEPLOY_USER}/.ssh"
  chmod 700 "/home/${DEPLOY_USER}/.ssh"
  chmod 600 "/home/${DEPLOY_USER}/.ssh/authorized_keys"
fi

log "step 5/7: ufw allow 22, 80, 443"
ufw --force default deny incoming
ufw --force default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

log "step 6/7: install certbot (snap route — matches Let's Encrypt docs)"
if ! command -v certbot >/dev/null 2>&1; then
  apt-get install -y snapd
  snap install core
  snap refresh core
  snap install --classic certbot
  ln -sf /snap/bin/certbot /usr/bin/certbot
fi

log "step 7/7: clone repo (if missing)"
if [ ! -d "$CHECKOUT_DIR/.git" ]; then
  sudo -u "$DEPLOY_USER" git clone "$REPO_URL" "$CHECKOUT_DIR"
else
  log "  → repo already present, skipping clone"
fi

log "bootstrap complete."
cat <<EOF

next steps (do these manually, once):
  1. populate  ${CHECKOUT_DIR}/backend/.env  (see backend/.env.example)
  2. populate  /etc/helios/duckdns.env      (see backend/deploy/duckdns/.env.example)
  3. run       sudo bash ${CHECKOUT_DIR}/backend/deploy/duckdns/install.sh
  4. wait ~5min for DNS to propagate, then:
     sudo certbot certonly --standalone \\
       -d \${DUCKDNS_DOMAIN}.duckdns.org \\
       --agree-tos -m hridambiswas2005@gmail.com --non-interactive
  5. cd ${CHECKOUT_DIR}/backend
     make prod-up
EOF
