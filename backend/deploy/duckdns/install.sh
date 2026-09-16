#!/usr/bin/env bash
# Install the DuckDNS updater on an Ubuntu host.
# Run once after /etc/helios/duckdns.env is populated.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "install.sh must be run as root (use sudo)" >&2
  exit 1
fi

if [ ! -f /etc/helios/duckdns.env ]; then
  echo "missing /etc/helios/duckdns.env — copy .env.example there first" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

install -m 0755 "$HERE/duckdns-update.sh" /usr/local/bin/helios-duckdns-update.sh
install -m 0644 "$HERE/duckdns.service"   /etc/systemd/system/duckdns.service
install -m 0644 "$HERE/duckdns.timer"     /etc/systemd/system/duckdns.timer

systemctl daemon-reload
systemctl enable --now duckdns.timer

echo "duckdns.timer installed — first run in 1 min, then every 5 min"
systemctl status --no-pager duckdns.timer
