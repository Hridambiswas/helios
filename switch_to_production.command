#!/usr/bin/env bash
# switch_to_production.command — restore the hosted setup.
#
# start_helios.command creates backend/.env from backend/.env.local on
# first run and marks the file with a header "# created-by: start_helios.command".
# This script removes that env file ONLY if it carries that marker,
# so a hand-written backend/.env is never clobbered.
#
# For "full capacity" (hosted backend + Vercel frontend), see DEMO_LOCAL.md.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ENV="$SCRIPT_DIR/backend/.env"

echo "Switching Helios back to the hosted setup…"

if [ ! -f "$BACKEND_ENV" ]; then
  echo "  backend/.env already absent — nothing to do."
  exit 0
fi

first_line="$(head -n 1 "$BACKEND_ENV")"
if [ "$first_line" = "# created-by: start_helios.command" ]; then
  rm -f "$BACKEND_ENV"
  echo "  removed demo-created backend/.env"
else
  echo "  backend/.env was NOT created by the demo (first line: '$first_line')."
  echo "  Leaving it alone. If you want to switch anyway:"
  echo "    rm \"$BACKEND_ENV\""
fi

echo ""
echo "Hosted setup ('full capacity'):"
echo "  - Frontend  : https://helios-hridam.vercel.app (served by Vercel)"
echo "  - Backend   : set VITE_API_URL to the hosted backend URL at build time"
echo "                (GitHub Actions: .github/workflows/deploy-frontend.yml)"
echo "  - Backend env lives on the host VM; nothing to do locally."
echo ""
echo "This script does NOT touch hosting — only the local backend/.env."
