# 002: Make the deploy tooling work on an Azure Ubuntu VM. Code only, no infra actions.

Context: DigitalOcean is no longer in the GitHub Student Pack. The human chose Azure for
Students ($100 credit). The target is one Ubuntu 22.04 x86_64 VM (4GB RAM, Central India),
static public IP, DuckDNS name `helios-hridam.duckdns.org`. The human will create the VM,
DuckDNS entry and keys himself. Do NOT ssh anywhere, call any cloud API, or set any
GitHub secret in this prompt.

Only start this after 001_report.md exists. If 001 found something that contradicts the
"no backend reachable" hypothesis, write 002_report.md saying BLOCKED with why, and stop.

## Read first, then change
Read end to end before editing: `backend/deploy/digitalocean/*`, `backend/deploy/duckdns/*`,
`backend/docker-compose.yml`, `docker-compose.prod.yml`, `backend/nginx/*`,
`backend/config.py`, `backend/.env.production.example`, `.github/workflows/deploy-*.yml`.
Write a short "current flow" section in the report before any change.

## Known problems to confirm and fix (one commit each)
1. Azure VMs have no root SSH. The scripts ssh/scp as `root@$DO_HOST`. Make a generic
   `backend/deploy/vm/` (copy, don't delete digitalocean/) that uses `$VM_USER`
   (default `azureuser`) with sudo, `$VM_HOST`, `$SSH_KEY`.
2. `bootstrap.sh` clones the repo's default branch (main), which lacks this work. Add
   `$DEPLOY_BRANCH` (default `fix/request-failed`) and check it out.
3. `first-deploy.sh` only requires GROQ_API_KEY. Also require GEMINI_API_KEY (or set
   VERIFIER_ENABLED=false explicitly with a loud warning), and write both into the server
   .env without echoing them.
4. CORS: first-deploy writes `CORS_ALLOWED_ORIGINS` as a JSON list; `config.py`
   `cors_origins_list` may expect comma separated. Verify the parser with a unit test and
   make the written format match. Origins must include https://helios-hridam.vercel.app.
5. Azure networking: document in the README that ports 22, 80, 443 must be opened in the
   VM's Network Security Group, in addition to ufw.
6. `rotate-secrets.sh`: rename vars to VM_*, keep `gh secret set VITE_API_URL` to
   https://helios-hridam.duckdns.org. Backend deploy workflow uses DO_* secrets; make it
   read VM_HOST / VM_USER / VM_SSH_KEY instead, and trigger on `fix/request-failed` plus
   workflow_dispatch (not only main).
7. Frontend deploy workflow only runs on push to main. Confirm `workflow_dispatch` can be
   run with `--ref fix/request-failed`, and document the exact command.
8. `backend/config.py` default `oauth_backend_url` still points at ddns.net. Change to
   the duckdns host.

## Also check (report, do not fix unless trivial)
- Memory budget of the compose stack against 4GB (sum of mem limits, CLIP model size).
  Recommend swap size if needed.
- Anything else in the deploy path that would fail on a fresh VM.

## Tests
`cd backend && pytest -q` must pass. Add tests for the CORS parsing. `bash -n` every
shell script you touched, and `shellcheck` if available.

## Report
Current flow, each problem confirmed or refuted with evidence, what changed (commit
hashes), remaining risks, and the exact command sequence the human will run once the VM
exists, with placeholders for secrets. Push to `fix/request-failed`. Never main.
