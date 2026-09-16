# DuckDNS updater

Keeps the DuckDNS A record for `${DUCKDNS_DOMAIN}.duckdns.org` pointed at this
host's current public IP. Runs every 5 minutes via a systemd timer.

Even though the Oracle Cloud Always-Free VM has a reserved public IP, this
updater is defensive: if the IP is ever detached and reattached, or if the VM
is rebuilt, the record catches up automatically.

## Files

| File              | Purpose                                          |
| ----------------- | ------------------------------------------------ |
| `duckdns-update.sh` | The actual `curl` to DuckDNS's update endpoint |
| `duckdns.service` | systemd unit that runs the script one-shot      |
| `duckdns.timer`   | systemd timer that fires the service every 5 min |
| `install.sh`      | Copies units into `/etc/systemd/system` + enables the timer |
| `.env.example`    | Template for `/etc/helios/duckdns.env`          |
