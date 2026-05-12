#!/bin/bash
# One-time HTTPS setup for Helios on AWS EC2
# Usage: DOMAIN=your.domain.com EMAIL=you@example.com bash scripts/setup_ssl.sh
set -e

DOMAIN="${DOMAIN:-helios-hridam.ddns.net}"
EMAIL="${EMAIL:?Set EMAIL=you@example.com before running this script}"
HELIOS_DIR="${HELIOS_DIR:-/home/ubuntu/helios}"

echo "==> [1/4] Installing certbot..."
sudo apt-get update -qq
sudo apt-get install -y certbot

echo "==> [2/4] Stopping nginx to free port 80 for certificate challenge..."
cd "$HELIOS_DIR"
docker compose -f docker-compose.prod.yml stop nginx

echo "==> [3/4] Obtaining Let's Encrypt certificate for $DOMAIN..."
sudo certbot certonly \
    --standalone \
    -d "$DOMAIN" \
    --non-interactive \
    --agree-tos \
    -m "$EMAIL"

echo "==> [4/4] Switching to SSL nginx config and restarting..."
cp "$HELIOS_DIR/nginx/nginx-ssl.conf" "$HELIOS_DIR/nginx/nginx.conf"
docker compose -f docker-compose.prod.yml up -d

echo ""
echo "SSL setup complete. HTTPS is live at https://$DOMAIN"
echo ""
echo "Next steps:"
echo "  1. Update Vercel env var VITE_API_URL to https://$DOMAIN"
echo "  2. Redeploy the frontend (push any commit or trigger redeploy in Vercel dashboard)"
