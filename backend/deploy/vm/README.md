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

## Files

| File                | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| `bootstrap.sh`      | Installs docker + certbot, opens ufw, clones repo (runs on the VM as the SSH user, sudos where needed) |
| `first-deploy.sh`   | End-to-end orchestrator (run from your laptop)             |
| `rotate-secrets.sh` | Sets VM_HOST / VM_USER + updates VITE_API_URL in GH Actions |
| `verify.sh`         | Post-deploy smoke test (`/nginx-health`, `/api/v1/health`, `/api/v1/stats`, `/docs`, `/metrics`) |
