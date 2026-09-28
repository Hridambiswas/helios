# Generic VM deploy (Azure, GCP, Oracle)

Idempotent bootstrap + one-shot deploy for Helios on any stock Ubuntu 22.04
x86_64 VM whose SSH login is a sudo-enabled non-root user (Azure →
`azureuser`, GCP → `ubuntu`, Oracle Cloud → `ubuntu`).

The DigitalOcean tooling in `../digitalocean/` was tied to root-SSH. Azure
disables root SSH by default; these scripts escalate through `sudo` instead.
The DigitalOcean scripts are preserved for anyone using DO — this directory
is an addition, not a replacement.

## TL;DR — full deploy in two commands

Once the VM is up (Ubuntu 22.04, static public IP, SSH key attached, ports
22/80/443 opened in the cloud firewall) and you've registered
`helios-hridam.duckdns.org`:

```bash
VM_HOST=<static-ip>                    \
DUCKDNS_TOKEN=<duckdns-token>          \
GROQ_API_KEY=<gsk_...>                 \
GEMINI_API_KEY=<AIza...>               \
  backend/deploy/vm/first-deploy.sh

VM_HOST=<static-ip>                    \
  backend/deploy/vm/rotate-secrets.sh
```

If you don't yet have a Gemini key, disable the verifier explicitly:

```bash
VERIFIER_ENABLED=false                 \
VM_HOST=<static-ip>                    \
DUCKDNS_TOKEN=<duckdns-token>          \
GROQ_API_KEY=<gsk_...>                 \
  backend/deploy/vm/first-deploy.sh
```

## Azure networking (must-do before first-deploy)

Ubuntu's `ufw` alone is not enough on Azure — the VM sits behind a
Network Security Group (NSG) that blocks inbound by default. Open the
same three ports at both layers:

1. In the Azure portal → VM → Networking → **Add inbound port rule**, or
   `az network nsg rule create ...`:
   - allow `TCP 22` (SSH — usually already there via the "SSH" default rule)
   - allow `TCP 80` (Let's Encrypt HTTP-01 challenge + nginx redirect)
   - allow `TCP 443` (HTTPS)
2. `bootstrap.sh` runs `ufw allow 22/80/443` on the guest — the two
   layers must agree, or requests are dropped at the NSG even though
   ufw is open.

Same idea for GCP (firewall rules under VPC networks) and Oracle Cloud
(security lists under the VCN's subnet).

## Deploying without pushing to main

The GitHub Actions backend workflow (`.github/workflows/deploy-backend.yml`)
triggers on pushes to `fix/request-failed` and `main`, plus manual
`workflow_dispatch`. To deploy the current `fix/request-failed` tip
without a merge:

```bash
gh workflow run deploy-backend.yml \
  --repo Hridambiswas/helios \
  --ref fix/request-failed
```

The frontend workflow (`.github/workflows/deploy-frontend.yml`) currently
triggers only on pushes to `main` under `frontend/**`, but it also
declares `workflow_dispatch:`, which — per GitHub's docs — ignores the
push `branches`/`paths` filter and can dispatch any ref that contains
the workflow file. To rebuild the Vercel bundle from `fix/request-failed`:

```bash
gh workflow run deploy-frontend.yml \
  --repo Hridambiswas/helios \
  --ref fix/request-failed
```

`rotate-secrets.sh` runs the same command automatically after updating
`VITE_API_URL`.

## Files

| File                | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| `bootstrap.sh`      | Installs docker + certbot, opens ufw, clones repo (runs on the VM as the SSH user, sudos where needed) |
| `first-deploy.sh`   | End-to-end orchestrator (run from your laptop)             |
| `rotate-secrets.sh` | Sets VM_HOST / VM_USER + updates VITE_API_URL in GH Actions |
| `verify.sh`         | Post-deploy smoke test (`/nginx-health`, `/api/v1/health`, `/api/v1/stats`, `/docs`, `/metrics`) |
