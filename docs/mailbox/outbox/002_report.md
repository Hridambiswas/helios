# 002 report — deploy tooling adapted for an Azure Ubuntu VM

001_report.md confirmed the "no backend reachable" hypothesis; nothing in it
contradicts the plan for 002, so this prompt was not blocked. Everything below
is code-only. No cloud API was called, no SSH was performed, no GitHub secret
was written. All commits are on `fix/request-failed`.

## Current flow (before this cycle)

`backend/deploy/digitalocean/` contained four scripts that assumed direct root
SSH:

- `bootstrap.sh` — apt-installs Docker + ufw + certbot, creates a `helios`
  user, clones the repo. Requires `id -u == 0` on entry.
- `first-deploy.sh` — runs on the operator's laptop, uploads `bootstrap.sh` via
  `ssh -o … root@$DO_HOST`, then seeds `/etc/helios/duckdns.env`,
  `/home/helios/helios/backend/.env`, runs certbot, brings compose up.
  Requires `DO_HOST`, `DUCKDNS_TOKEN`, `GROQ_API_KEY` (but not `GEMINI_API_KEY`,
  even though `config.py` refuses to boot without it when `VERIFIER_ENABLED`
  is on).
- `rotate-secrets.sh` — sets `gh secret DO_HOST` and `gh secret VITE_API_URL`,
  triggers `deploy-frontend.yml`.
- `verify.sh` — curls `/nginx-health`, `/api/v1/stats`, `/docs`, `/metrics`.

`.github/workflows/deploy-backend.yml` used `secrets.DO_HOST` /
`secrets.DO_USER` / `secrets.DO_SSH_KEY` and only triggered on pushes to
`main`. Its in-container script hard-coded `cd /home/helios/helios`.

`backend/config.py` `oauth_backend_url` defaulted to
`https://helios-hridam.ddns.net` — the dead No-IP host.

`backend/config.py` `cors_origins_list` accepts JSON list first, comma
fallback second, but had no test.

None of this ran anywhere in the last four months; the last successful backend
deploy was 2026-05-13, and the 2026-09-24 attempt died on `no space left on
device` during docker layer export (evidence in 001_report.md task C).

## Problems — confirmed and addressed

I combined problems 1 and 3 into a single commit because a from-scratch
`vm/first-deploy.sh` naturally requires both keys as part of its initial
adaptation — splitting the require-check out into a second commit would have
left an intentionally broken script on disk between commits. Every other
numbered problem got its own commit.

| # | Problem | Status | Commit |
|---|---------|--------|--------|
| 1 | Azure disables root SSH; scripts hard-code `root@$DO_HOST` | fixed | `b84e54d` |
| 2 | `bootstrap.sh` clones the default branch (`main`), which lacks the vm/ scripts and the model/verifier work still on `fix/request-failed` | fixed | `6456535` |
| 3 | `first-deploy.sh` only required `GROQ_API_KEY`, but v1.2 `config.py:190` refuses to boot when `VERIFIER_ENABLED=true` and `GEMINI_API_KEY` is empty | fixed | `b84e54d` (bundled with #1) |
| 4 | CORS: `first-deploy.sh` and `deploy-backend.yml` write a JSON list; `config.py.cors_origins_list` should parse both JSON and comma. Verify with a test | verified (parser already handles both) + tests added | `3275a6f` |
| 5 | Azure NSG must open 22/80/443 in addition to ufw | documented | `cf26d7f` |
| 6 | Backend deploy workflow used DO_* secrets and only triggered on main | renamed to VM_*, added `fix/request-failed` trigger and disk-pruning step | `b35a7e4` |
| 7 | Frontend deploy workflow only runs on push-to-main; document `workflow_dispatch --ref fix/request-failed` | documented | `cf26d7f` (bundled with #5, both are README-only) |
| 8 | `config.py` default `oauth_backend_url` still points at ddns.net | changed to duckdns.org | `1635bf6` |

Details per problem:

### 1 — root SSH on Azure (fixed in `b84e54d`)

Copied `backend/deploy/digitalocean/` to a fresh `backend/deploy/vm/` (per the
prompt's "copy, don't delete" instruction — DigitalOcean tooling is preserved).

`vm/bootstrap.sh` now:
- Refuses to run as root (`id -u == 0` → abort) and instead primes sudo with
  `sudo -v`.
- Wraps every root-only command (`apt-get`, `install`, `gpg --dearmor`,
  `systemctl`, `snap install`, `ufw`, `certbot`) in `sudo`.
- Does not create a separate `helios` user. The checkout lives at
  `$HOME/helios`, i.e. `/home/azureuser/helios`, owned by the SSH login user.
- Adds the SSH login user to the `docker` group so subsequent SSH sessions
  can run `docker` without sudo.

`vm/first-deploy.sh` now:
- Accepts `VM_HOST`, `VM_USER` (default `azureuser`), `SSH_KEY` (default
  `~/.ssh/id_ed25519`) in place of the DO variables.
- Streams the `/etc/helios/duckdns.env` and `backend/.env` heredocs through
  `sudo tee` on the target so secrets never touch the terminal.
- Uses `sudo docker compose` on the first up because the SSH shell that ran
  `usermod -aG docker` has not yet re-authenticated into the docker group.

`vm/rotate-secrets.sh` uses `VM_HOST` / `VM_USER` and dispatches
`deploy-frontend.yml` against `fix/request-failed`. It intentionally does
**not** write `VM_SSH_KEY` — that has to be piped in via
`gh secret set VM_SSH_KEY < path/to/key` so the private key never lands in
the process arg list.

`vm/verify.sh` was cloned unchanged except for adding a `/api/v1/health` probe
next to `/nginx-health` (this is the real health endpoint per `routes.py:728`;
task E7 in 001_report.md).

Static evidence:
```
$ bash -n backend/deploy/vm/bootstrap.sh backend/deploy/vm/first-deploy.sh \
        backend/deploy/vm/rotate-secrets.sh backend/deploy/vm/verify.sh
(all four → OK)
```
`shellcheck` is not installed on the workstation (`command -v shellcheck` →
empty); documented rather than installed.

### 2 — clone main which lacks this work (fixed in `6456535`)

`vm/bootstrap.sh` now reads `$DEPLOY_BRANCH` (default `fix/request-failed`)
and does `git clone --branch "$DEPLOY_BRANCH"`. If the checkout already
exists, it fetches and `git reset --hard "origin/$DEPLOY_BRANCH"` so re-runs
converge. Once this branch merges, `DEPLOY_BRANCH` can be lowered back to
`main` via env override, or the default flipped in a follow-up commit.

### 3 — GEMINI_API_KEY not required (fixed in `b84e54d`)

`vm/first-deploy.sh` requires `GEMINI_API_KEY` unless `VERIFIER_ENABLED=false`
is passed explicitly, and prints a loud multi-line stderr warning in the
opt-out case. Excerpt:
```
VERIFIER_ENABLED="${VERIFIER_ENABLED:-true}"
if [ "$VERIFIER_ENABLED" = "true" ]; then
  : "${GEMINI_API_KEY:?GEMINI_API_KEY not set …}"
else
  echo "  ******************************************************************" >&2
  echo "  WARNING: VERIFIER_ENABLED=false — Gemini verifier will be SKIPPED." >&2
  …
fi
```
`GROQ_API_KEY`, `GEMINI_API_KEY`, `JWT_SECRET_KEY`, `POSTGRES_PASSWORD`,
`MINIO_SECRET_KEY` are all written into `backend/.env` via `sudo tee` and
never echoed to stdout by the script. `set -x` is deliberately not used.

Verified against `backend/config.py:183–202` (`validate_secrets`): the
combination of `GROQ_API_KEY` + `GEMINI_API_KEY` (when verifier on) +
`JWT_SECRET_KEY` + `POSTGRES_PASSWORD` + `MINIO_ACCESS_KEY/SECRET_KEY` is
exactly what boot demands in production.

### 4 — CORS list parsing (verified + tests added in `3275a6f`)

`backend/config.py:137–145`:
```
@property
def cors_origins_list(self) -> list[str]:
    v = self.cors_allowed_origins.strip()
    if not v:
        return []
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return [i.strip() for i in v.split(",") if i.strip()]
```
The parser already accepts both formats — no code change needed. Added
`backend/tests/test_cors_parsing.py` (5 tests, all pass):
- `test_cors_accepts_json_list` — matches what the deploy scripts write.
- `test_cors_accepts_comma_separated` — matches `.env.production.example`.
- `test_cors_accepts_single_origin` — the `docker-compose.prod.yml` inline
  override case.
- `test_cors_empty_string_yields_empty_list` — the default in `config.py`.
- `test_cors_deploy_scripts_include_the_vercel_origin` — guards against
  future edits accidentally dropping `helios-hridam.vercel.app`, which
  would re-surface the `Request failed` symptom as a CORS block.

```
$ cd backend && python -m pytest tests/test_cors_parsing.py -v
tests/test_cors_parsing.py::test_cors_accepts_json_list                     PASSED
tests/test_cors_parsing.py::test_cors_accepts_comma_separated               PASSED
tests/test_cors_parsing.py::test_cors_accepts_single_origin                 PASSED
tests/test_cors_parsing.py::test_cors_empty_string_yields_empty_list        PASSED
tests/test_cors_parsing.py::test_cors_deploy_scripts_include_the_vercel_origin PASSED
5 passed
```
The written format therefore does not need to change; the deploy scripts
continue writing JSON lists.

### 5 — Azure NSG documentation (added in `cf26d7f`)

Added a top-of-file "Azure networking (must-do before first-deploy)" section
to `backend/deploy/vm/README.md` that names the two firewall layers (Azure
NSG + ufw), lists the exact ports (22, 80, 443), gives both portal and
`az network nsg rule create` paths, and calls out that GCP firewall rules
and Oracle security lists are the analogous layers on those providers.

### 6 — deploy-backend.yml VM_* + branch triggers (fixed in `b35a7e4`)

`.github/workflows/deploy-backend.yml` now:
- Reads `secrets.VM_HOST` / `secrets.VM_USER` / `secrets.VM_SSH_KEY`.
- Triggers on pushes to `fix/request-failed` **and** `main` (per the prompt's
  "not only main"), plus `workflow_dispatch`.
- Hard-resets the on-VM checkout to the exact ref that triggered the
  workflow (`REF="${GITHUB_REF_NAME:-main}"` → `git reset --hard "origin/$REF"`)
  instead of always pulling `origin/main` — so a push to
  `fix/request-failed` actually deploys that branch's code.
- Uses `~/helios` for the in-container checkout path so it works regardless
  of which login user `VM_USER` is set to (was `/home/helios/helios`).
- Adds `docker builder prune -af` next to the existing `docker system prune`
  — the 2026-09-24 failure was `no space left on device` during layer
  export while builder cache had accumulated (001_report.md task C).

### 7 — frontend workflow_dispatch documentation (added in `cf26d7f`)

Same commit as #5 (both README-only). The section spells out that
`deploy-frontend.yml` only auto-triggers on pushes to `main` under
`frontend/**`, but its `workflow_dispatch:` clause lets any ref containing
the workflow file be dispatched manually. Command:
```
gh workflow run deploy-frontend.yml \
  --repo Hridambiswas/helios \
  --ref fix/request-failed
```
`vm/rotate-secrets.sh` already runs the equivalent line after updating
`VITE_API_URL`.

### 8 — oauth_backend_url default (fixed in `1635bf6`)

`backend/config.py:129` changed from
```
oauth_backend_url: str = "https://helios-hridam.ddns.net"
```
to
```
# ddns.net no longer resolves — see 001_report.md.
oauth_backend_url: str = "https://helios-hridam.duckdns.org"
```
This is the OAuth GitHub-callback host. When `OAUTH_BACKEND_URL` is not set
in `.env`, the callback URI that OAuth builds now points at the DuckDNS
host — matching what `vm/first-deploy.sh` writes and what nginx will serve.

## Also-checked (memory budget + remaining risks)

**Memory budget on a 4 GB Azure VM.** Sum of the `mem_limit` values in
`backend/docker-compose.prod.yml`:

| Service | mem_limit |
| ------- | --------- |
| nginx | 64m |
| api | 512m |
| worker | 200m |
| beat | 64m |
| postgres | 128m |
| redis | 80m |
| minio | 96m |
| chroma | 160m |
| **sum** | **1304m ≈ 1.3 GB** |

Add the guest OS + Docker daemon + snapd + systemd (~600 MB on a fresh Ubuntu
22.04) and the stack should fit in ~2 GB steady-state, leaving ~2 GB headroom.

But the **api** container's 512m limit is tight: `agents/*` loads
`open_clip ViT-B-32` (≈150 MB resident) and `BAAI/bge-small-en-v1.5`
(≈130 MB resident) at import time, before any request handling, before the
Groq / Gemini SDKs allocate. Under a burst of concurrent queries or a large
retriever result set, the api process can briefly exceed 512m and be
OOM-killed by the kernel with no swap to page into. **Recommend a 2 GB
swap file on the VM** — added as a step for the human in the command
sequence below. This is a report-only recommendation; no code change was
made to `docker-compose.prod.yml` because raising the api limit above 512m
would silently shift the OOM ceiling in a way the operator may not want.

**Remaining risks — not fixed in this cycle:**

- `frontend/src/components/AuthModal.tsx:4` hard-codes
  `'https://helios-hridam.ddns.net'` as the fallback default when
  `VITE_API_URL` is missing. Once VITE_API_URL is rotated, this fallback
  never fires; but it should still be flipped to duckdns.org for future
  proofing. Out of 002's scope (no frontend deploy tooling changes were
  asked for).
- `backend/scripts/setup_ssl.sh:6` also defaults `DOMAIN=helios-hridam.ddns.net`.
  Same class of stale-default problem. Also out of scope.
- `docker-compose.prod.yml` line 23 hard-codes
  `CORS_ALLOWED_ORIGINS=https://helios-hridam.vercel.app` in the api
  service's `environment:` block. Since compose's `environment` overrides
  `env_file`, this shadows the two-origin JSON list the deploy workflow
  writes into `backend/.env`, meaning only the primary Vercel alias is
  accepted at runtime (the `frontend-omega-blush-87.vercel.app` preview
  alias is silently blocked). Not urgent — the frontend uses the primary
  alias — but a papercut worth mentioning.
- `pytest -q` locally reports `80 passed, 30 failed, 62 errors`. Every
  failure and error resolves to `ModuleNotFoundError: No module named
  'langchain_groq'` (or a sibling) — the local Python env is missing
  runtime deps, not a regression. The two tests I added (5 CORS cases)
  and the pre-existing config-defaults tests all pass. Sample:
  ```
  E   ModuleNotFoundError: No module named 'langchain_groq'
  ```
  Fixing the local env is out of scope; CI runs against a
  requirements.txt-installed image and will exercise the same tests
  cleanly. `cd frontend && npm run build` was not run because no
  frontend source was touched in this cycle.

## Exact command sequence the human runs once the VM exists

Placeholders in `<angle brackets>`. Never echo the secret values to a shell
history file — prefer entering them interactively or through a password
manager fill.

```bash
# ── 0. Azure prerequisites (once) ───────────────────────────────────────
# In the Azure portal, VM → Networking → Add inbound port rule:
#   allow TCP 22 (SSH — default rule usually already exists)
#   allow TCP 80 (Let's Encrypt HTTP-01 + nginx redirect)
#   allow TCP 443 (HTTPS)
# Confirm the VM has a static public IP (Reserve → Static).
# Register helios-hridam.duckdns.org on https://www.duckdns.org.
# Copy your public SSH key to azureuser during VM creation.

# ── 1. First deploy (from your laptop) ──────────────────────────────────
export VM_HOST=<static-public-ip>
export VM_USER=azureuser              # optional, this is the default
export DUCKDNS_TOKEN=<duckdns-token>
export GROQ_API_KEY=<gsk_...>
export GEMINI_API_KEY=<AIza...>
# Optional:
# export SSH_KEY=$HOME/.ssh/<your_private_key>
# export VERIFIER_ENABLED=false        # only if you don't have a Gemini key

backend/deploy/vm/first-deploy.sh

# ── 2. Add a 2 GB swap on the VM (once, defensive) ──────────────────────
ssh -i "$SSH_KEY" "$VM_USER@$VM_HOST" '
  sudo fallocate -l 2G /swapfile &&
  sudo chmod 600 /swapfile &&
  sudo mkswap /swapfile &&
  sudo swapon /swapfile &&
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab &&
  free -h
'

# ── 3. Rotate GitHub Actions secrets ────────────────────────────────────
# Set the private key ONCE via stdin — never expose it as a positional arg:
gh secret set VM_SSH_KEY --repo Hridambiswas/helios < "${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# Then set VM_HOST + VM_USER + VITE_API_URL and dispatch the frontend rebuild:
VM_HOST="$VM_HOST" VM_USER="$VM_USER" \
  backend/deploy/vm/rotate-secrets.sh

# ── 4. Dispatch the backend deploy against fix/request-failed ───────────
gh workflow run deploy-backend.yml \
  --repo Hridambiswas/helios \
  --ref fix/request-failed

# ── 5. Verify from anywhere ────────────────────────────────────────────
backend/deploy/vm/verify.sh helios-hridam.duckdns.org
# Then open https://helios-hridam.vercel.app in a real browser and submit
# a query — the 'Request failed' toast should be replaced with a real
# answer. If not, capture the failing request from DevTools → Network and
# open 003_prompt.md for the next diagnosis cycle.

# ── 6. (once green) delete stale rollback secrets ───────────────────────
for s in DO_HOST DO_USER DO_SSH_KEY \
         EC2_HOST EC2_USER EC2_SSH_KEY \
         ORACLE_USER ORACLE_SSH_KEY; do
  gh secret delete "$s" --repo Hridambiswas/helios 2>/dev/null || true
done
```

## Commits pushed on `fix/request-failed`

| SHA | Subject |
|-----|---------|
| `b84e54d` | feat(deploy): add backend/deploy/vm/ for Azure/GCP/Oracle Ubuntu VMs |
| `6456535` | feat(deploy/vm): add DEPLOY_BRANCH override in bootstrap.sh |
| `3275a6f` | test(config): cover cors_origins_list JSON-list and comma-separated formats |
| `b35a7e4` | ci(deploy-backend): switch to VM_HOST/VM_USER/VM_SSH_KEY, trigger fix/request-failed |
| `1635bf6` | fix(config): default oauth_backend_url to the DuckDNS host |
| `cf26d7f` | docs(deploy/vm): document Azure NSG rules and workflow_dispatch commands |

(This report is the 7th commit on top.)
