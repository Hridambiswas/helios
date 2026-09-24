#!/usr/bin/env bash
# One-shot first-time deploy for a DigitalOcean droplet.
#
# Run from your LAPTOP after you have:
#   1. Created the Ubuntu 22.04 droplet with your SSH key attached.
#   2. Attached a Floating (reserved) IP.
#   3. Registered helios-hridam.duckdns.org on https://www.duckdns.org.
#
# Usage:
#   DO_HOST=<floating-ip>         \
#   DUCKDNS_TOKEN=<uuid>          \
#   GROQ_API_KEY=<gsk_...>        \
#   ./first-deploy.sh
#
# Optional env vars:
#   SSH_KEY          path to private key (default ~/.ssh/id_ed25519_helios_oracle)
#   DUCKDNS_DOMAIN   subdomain, no .duckdns.org suffix (default helios-hridam)
#   ACME_EMAIL       Let's Encrypt account email (default hridambiswas2005@gmail.com)
#   JWT_SECRET_KEY   auto-generated with openssl rand -hex 32 if unset

set -euo pipefail

: "${DO_HOST:?DO_HOST not set — pass the droplet's Floating IP}"
: "${DUCKDNS_TOKEN:?DUCKDNS_TOKEN not set — grab it from https://www.duckdns.org}"
: "${GROQ_API_KEY:?GROQ_API_KEY not set — Groq console → API keys}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_helios_oracle}"
DUCKDNS_DOMAIN="${DUCKDNS_DOMAIN:-helios-hridam}"
ACME_EMAIL="${ACME_EMAIL:-hridambiswas2005@gmail.com}"
JWT_SECRET_KEY="${JWT_SECRET_KEY:-$(openssl rand -hex 32)}"
FULL_DOMAIN="${DUCKDNS_DOMAIN}.duckdns.org"

# DO droplets are root-by-default; bootstrap creates a `helios` user later.
SSH_ROOT="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 root@$DO_HOST"
SSH_HELIOS="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 helios@$DO_HOST"
SCP_ROOT="scp -i $SSH_KEY -o StrictHostKeyChecking=accept-new"

log() { printf '\n\033[1;36m[first-deploy]\033[0m %s\n' "$*"; }

log "sanity: can we reach $DO_HOST as root?"
$SSH_ROOT 'echo "ssh ok — $(hostname) — $(uname -m)"'

log "1/8 uploading bootstrap.sh"
$SCP_ROOT "$(dirname "$0")/bootstrap.sh" "root@$DO_HOST:/tmp/bootstrap.sh"

log "2/8 running bootstrap.sh (installs docker, ufw, creates helios user)"
$SSH_ROOT 'bash /tmp/bootstrap.sh'

log "3/8 seeding /etc/helios/duckdns.env"
$SSH_ROOT "mkdir -p /etc/helios && tee /etc/helios/duckdns.env >/dev/null" <<EOF
DUCKDNS_DOMAIN=${DUCKDNS_DOMAIN}
DUCKDNS_TOKEN=${DUCKDNS_TOKEN}
EOF
$SSH_ROOT 'chmod 600 /etc/helios/duckdns.env'

log "4/8 installing DuckDNS systemd timer + first ping"
$SSH_ROOT 'bash /home/helios/helios/backend/deploy/duckdns/install.sh'
$SSH_ROOT 'systemctl start duckdns.service && sleep 3 && journalctl -u duckdns.service -n 5 --no-pager'

log "5/8 waiting for DNS to propagate ($FULL_DOMAIN → $DO_HOST)"
for i in $(seq 1 30); do
  resolved=$(dig +short "$FULL_DOMAIN" @8.8.8.8 | tail -1)
  if [ "$resolved" = "$DO_HOST" ]; then
    log "  → resolved after ${i}0s"
    break
  fi
  printf '.'
  sleep 10
done
if [ "$resolved" != "$DO_HOST" ]; then
  echo "DNS still not resolving to $DO_HOST (got '$resolved') — aborting."
  echo "Check DuckDNS token, then visit https://www.duckdns.org to force update."
  exit 1
fi

log "6/8 issuing Let's Encrypt cert for $FULL_DOMAIN (standalone HTTP-01)"
$SSH_ROOT "certbot certonly --standalone -d $FULL_DOMAIN --agree-tos -m $ACME_EMAIL --non-interactive"

log "7/8 seeding backend/.env with GROQ + JWT (deploy workflow appends the rest)"
$SSH_ROOT "tee /home/helios/helios/backend/.env >/dev/null" <<EOF
GROQ_API_KEY=${GROQ_API_KEY}
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=60
JWT_REFRESH_EXPIRY_DAYS=7
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=production
LOG_LEVEL=INFO
LOG_FORMAT=json
PUBLIC_HOSTNAME=${FULL_DOMAIN}
CORS_ALLOWED_ORIGINS=["https://helios-hridam.vercel.app","https://frontend-omega-blush-87.vercel.app"]
OAUTH_FRONTEND_URL=https://helios-hridam.vercel.app
OAUTH_BACKEND_CALLBACK=https://${FULL_DOMAIN}/api/v1/auth/github/callback
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=$(openssl rand -hex 16)
MINIO_BUCKET=helios-docs
MINIO_SECURE=false
CHROMA_HOST=chroma
CHROMA_PORT=8000
CHROMA_COLLECTION=helios
EOF
$SSH_ROOT 'chown helios:helios /home/helios/helios/backend/.env && chmod 600 /home/helios/helios/backend/.env'

log "8/8 bringing the stack up (docker compose prod, x86_64)"
$SSH_HELIOS 'cd /home/helios/helios/backend && make prod-up'

log "waiting 45s for containers to warm up..."
sleep 45

log "verify"
"$(dirname "$0")/verify.sh" "$FULL_DOMAIN"

cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║  first-deploy complete                                            ║
╠══════════════════════════════════════════════════════════════════╣
║  next: run rotate-secrets.sh so GitHub Actions can push future    ║
║        deploys, and the frontend rebuild picks up the new API URL:║
║                                                                    ║
║    DO_HOST=$DO_HOST \\
║      backend/deploy/digitalocean/rotate-secrets.sh                 ║
╚══════════════════════════════════════════════════════════════════╝
EOF
