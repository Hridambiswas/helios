# Migration notes

Backend is being migrated **off AWS EC2** (dynamic IP + expired No-IP DDNS) to
a **DigitalOcean droplet** funded by the GitHub Student Developer Pack's $200
credit, with **DuckDNS** replacing No-IP.

## Cut-over cheat sheet

After provisioning the DO droplet (Ubuntu 22.04 + attached Floating IP + SSH
key installed) and registering `helios-hridam.duckdns.org` on DuckDNS:

```bash
DO_HOST=<floating-ip> DUCKDNS_TOKEN=<token> GROQ_API_KEY=<key> \
  backend/deploy/digitalocean/first-deploy.sh

DO_HOST=<floating-ip> \
  backend/deploy/digitalocean/rotate-secrets.sh
```

Details: [`backend/deploy/digitalocean/README.md`](backend/deploy/digitalocean/README.md).

## Alternate target (kept for posterity)

An Oracle Cloud Always-Free ARM path is also in-tree at
[`backend/deploy/oracle/`](backend/deploy/oracle/) — the original migration
target, still fully wired up if you want to switch to it later. Both paths
share the same DuckDNS scripts (`backend/deploy/duckdns/`), nginx SSL config,
and Let's Encrypt flow.

## Why this migration

- `helios-hridam.ddns.net` no longer resolves (No-IP free tier expired).
- `3.110.161.146` no longer belongs to us (AWS reassigned the public IP after
  the instance stopped without an Elastic IP).
- Frontend surfaced both as `request failed` on every call.

Full write-up (with rollback plan): [`docs/migration/oracle-arm.md`](docs/migration/oracle-arm.md).
