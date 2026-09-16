# DigitalOcean droplet deploy

Idempotent bootstrap + one-shot deploy for Helios on a DigitalOcean Ubuntu
22.04 droplet (Regular Intel, x86_64), funded by the GitHub Student Developer
Pack's $200 credit.

## Why DigitalOcean

- No card required at signup (student credit covers it).
- Reserved Floating IP is free while attached — pins the API URL forever.
- BLR1 region (Bangalore) → low latency for the target audience.
- Docker + apt work the same as EC2 — the whole compose stack drops in.

## TL;DR — full deploy in two commands

Once the droplet is up (Ubuntu 22.04 + reserved Floating IP + SSH key attached)
and you've registered `helios-hridam.duckdns.org`:

```bash
DO_HOST=<floating-ip> \
DUCKDNS_TOKEN=<duckdns-token> \
GROQ_API_KEY=<gsk_...> \
  backend/deploy/digitalocean/first-deploy.sh

DO_HOST=<floating-ip> \
  backend/deploy/digitalocean/rotate-secrets.sh
```

## Files

| File              | Purpose                                              |
| ----------------- | ---------------------------------------------------- |
| `bootstrap.sh`    | Installs docker + certbot, opens firewall, clones repo |
| `first-deploy.sh` | End-to-end orchestrator (run from your laptop)       |
| `rotate-secrets.sh` | Sets DO_HOST + updates VITE_API_URL in GH Actions  |
| `verify.sh`       | Post-deploy smoke test (`/nginx-health`, `/api/v1/stats`, `/docs`, `/metrics`) |
