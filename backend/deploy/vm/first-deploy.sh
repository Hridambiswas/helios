#!/usr/bin/env bash
# One-shot first-time deploy for a generic Ubuntu VM (Azure, GCP, Oracle).
#
# Run from your LAPTOP after you have:
#   1. Created an Ubuntu 22.04 x86_64 VM (>= 4 GB RAM recommended).
#   2. Given it a static / reserved public IP.
#   3. Opened ports 22, 80, 443 in the cloud provider's firewall
#      (Azure Network Security Group, GCP firewall rules, Oracle security list).
#   4. Registered helios-hridam.duckdns.org on https://www.duckdns.org.
#
# Usage (Azure example — VM_USER defaults to azureuser):
#   VM_HOST=<static-public-ip>    \
#   DUCKDNS_TOKEN=<uuid>          \
#   GROQ_API_KEY=<gsk_...>        \
#   GEMINI_API_KEY=<AIza...>      \
#   ./first-deploy.sh
#
# Env vars:
#   VM_HOST            required — VM's public IP or DNS name
#   VM_USER            SSH login user (default: azureuser)
#   SSH_KEY            private key (default: ~/.ssh/id_ed25519)
#   DUCKDNS_TOKEN      required — DuckDNS account token
#   DUCKDNS_DOMAIN     DuckDNS subdomain (default: helios-hridam)
#   GROQ_API_KEY       required — primary LLM
#   GEMINI_API_KEY     required — Gemini verifier (or set VERIFIER_ENABLED=false)
#   VERIFIER_ENABLED   set to "false" to skip GEMINI_API_KEY (loud warning)
#   ACME_EMAIL         Let's Encrypt email (default: hridambiswas2005@gmail.com)
#   JWT_SECRET_KEY     auto-generated with openssl rand -hex 32 if unset

set -euo pipefail

: "${VM_HOST:?VM_HOST not set - pass the VM static public IP or DNS name}"
: "${DUCKDNS_TOKEN:?DUCKDNS_TOKEN not set — grab it from https://www.duckdns.org}"
: "${GROQ_API_KEY:?GROQ_API_KEY not set — Groq console → API keys}"

VM_USER="${VM_USER:-azureuser}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
DUCKDNS_DOMAIN="${DUCKDNS_DOMAIN:-helios-hridam}"
ACME_EMAIL="${ACME_EMAIL:-hridambiswas2005@gmail.com}"
JWT_SECRET_KEY="${JWT_SECRET_KEY:-$(openssl rand -hex 32)}"
VERIFIER_ENABLED="${VERIFIER_ENABLED:-true}"
FULL_DOMAIN="${DUCKDNS_DOMAIN}.duckdns.org"
CHECKOUT_DIR="/home/${VM_USER}/helios"

if [ "$VERIFIER_ENABLED" = "true" ]; then
  : "${GEMINI_API_KEY:?GEMINI_API_KEY not set — Google AI Studio → API keys. Or run with VERIFIER_ENABLED=false to disable the Gemini verifier.}"
else
  echo "" >&2
  echo "  ******************************************************************" >&2
  echo "  WARNING: VERIFIER_ENABLED=false — Gemini verifier will be SKIPPED." >&2
  echo "  Answers will not be cross-checked. Only use this for local dev or" >&2
  echo "  when you have not yet obtained a GEMINI_API_KEY." >&2
  echo "  ******************************************************************" >&2
  echo "" >&2
  GEMINI_API_KEY="${GEMINI_API_KEY:-}"
fi

SSH_BASE="ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ${VM_USER}@${VM_HOST}"
SCP_BASE="scp -i $SSH_KEY -o StrictHostKeyChecking=accept-new"

log() { printf '\n\033[1;36m[first-deploy]\033[0m %s\n' "$*"; }

log "sanity: can we reach $VM_HOST as $VM_USER?"
$SSH_BASE 'echo "ssh ok — $(hostname) — $(uname -m)"'

log "1/8 uploading bootstrap.sh"
$SCP_BASE "$(dirname "$0")/bootstrap.sh" "${VM_USER}@${VM_HOST}:/tmp/bootstrap.sh"

log "2/8 running bootstrap.sh (installs docker, ufw, clones repo)"
$SSH_BASE 'bash /tmp/bootstrap.sh'

log "3/8 seeding /etc/helios/duckdns.env"
# Never echo secret values to the terminal — pipe through ssh into a
# root-owned file with 600 perms.
$SSH_BASE "sudo mkdir -p /etc/helios && sudo tee /etc/helios/duckdns.env >/dev/null && sudo chmod 600 /etc/helios/duckdns.env" <<EOF
DUCKDNS_DOMAIN=${DUCKDNS_DOMAIN}
DUCKDNS_TOKEN=${DUCKDNS_TOKEN}
EOF

log "4/8 installing DuckDNS systemd timer + first ping"
$SSH_BASE "sudo bash ${CHECKOUT_DIR}/backend/deploy/duckdns/install.sh"
$SSH_BASE 'sudo systemctl start duckdns.service && sleep 3 && sudo journalctl -u duckdns.service -n 5 --no-pager'

log "5/8 waiting for DNS to propagate ($FULL_DOMAIN → $VM_HOST)"
resolved=""
for i in $(seq 1 30); do
  resolved=$(dig +short "$FULL_DOMAIN" @8.8.8.8 | tail -1)
  if [ "$resolved" = "$VM_HOST" ]; then
    log "  → resolved after ~$((i*10))s"
    break
  fi
  printf '.'
  sleep 10
done
if [ "$resolved" != "$VM_HOST" ]; then
  echo "DNS still not resolving to $VM_HOST (got '$resolved') — aborting."
  echo "Check DUCKDNS_TOKEN, then visit https://www.duckdns.org to force update."
  exit 1
fi

log "6/8 issuing Let's Encrypt cert for $FULL_DOMAIN (standalone HTTP-01)"
$SSH_BASE "sudo certbot certonly --standalone -d $FULL_DOMAIN --agree-tos -m $ACME_EMAIL --non-interactive"

log "7/8 seeding ${CHECKOUT_DIR}/backend/.env (deploy workflow appends OAuth + Supabase)"
# GEMINI_API_KEY may legitimately be empty when VERIFIER_ENABLED=false.
# The heredoc is piped straight into a root-owned tee — secrets never
# reach the terminal.
$SSH_BASE "sudo tee ${CHECKOUT_DIR}/backend/.env >/dev/null" <<EOF
GROQ_API_KEY=${GROQ_API_KEY}
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_API_KEY=${GEMINI_API_KEY}
GEMINI_MODEL=gemini-2.5-flash
VERIFIER_ENABLED=${VERIFIER_ENABLED}
VERIFIER_MIN_SCORE=0.5
VERIFIER_TIMEOUT_SECONDS=20
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
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
OAUTH_BACKEND_URL=https://${FULL_DOMAIN}
POSTGRES_PASSWORD=$(openssl rand -hex 24)
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
$SSH_BASE "sudo chown ${VM_USER}:${VM_USER} ${CHECKOUT_DIR}/backend/.env && sudo chmod 600 ${CHECKOUT_DIR}/backend/.env"

log "8/8 bringing the stack up (docker compose prod, x86_64)"
# First run may need sudo because the docker group is not active in this
# shell yet. Subsequent SSH sessions get it for free.
$SSH_BASE "cd ${CHECKOUT_DIR}/backend && (sudo docker compose -p helios -f docker-compose.yml -f docker-compose.prod.yml up -d --build)"

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
║    VM_HOST=$VM_HOST \\                                              ║
║      backend/deploy/vm/rotate-secrets.sh                          ║
╚══════════════════════════════════════════════════════════════════╝
EOF
