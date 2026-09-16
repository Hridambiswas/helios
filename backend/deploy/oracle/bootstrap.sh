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

log "step 3/8: add Docker apt repo"
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

log "step 4/8: install docker-ce + compose plugin"
apt-get install -y \
  docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker

log "step 5/8: add ubuntu to docker group"
if ! id -nG ubuntu | grep -qw docker; then
  usermod -aG docker ubuntu
  log "  → ubuntu added to docker group (log out + back in to pick up)"
fi

log "step 6/8: open ports 80 + 443 in iptables (Oracle default policy blocks)"
for port in 80 443; do
  if ! iptables -C INPUT -m state --state NEW -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
    iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$port" -j ACCEPT
  fi
done
netfilter-persistent save

log "step 7/8: install certbot (snap route — smaller than apt version)"
if ! command -v certbot >/dev/null 2>&1; then
  apt-get install -y snapd
  snap install core
  snap refresh core
  snap install --classic certbot
  ln -sf /snap/bin/certbot /usr/bin/certbot
fi
