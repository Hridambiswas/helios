#!/usr/bin/env bash
# Run by certbot's post-hook after a successful renewal.
# Reloads the nginx container so the new cert is picked up without downtime.

set -euo pipefail

CONTAINER="${HELIOS_NGINX_CONTAINER:-helios-nginx-1}"

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker exec "$CONTAINER" nginx -s reload
  echo "certbot post-hook: reloaded nginx in $CONTAINER"
else
  echo "certbot post-hook: nginx container $CONTAINER not running; skipped reload"
fi
