# 005 report — HF Space Parts C/D BLOCKED (billing) — no infra touched

**Status: BLOCKED at Part C step 1.** Docker Spaces on the free `cpu-basic`
tier now require a Hugging Face **PRO subscription** ($9/mo). The
`create_repo(... space_sdk="docker")` call fails immediately with
HTTP 402 Payment Required. Nothing was created on Hugging Face, no
Space secret or variable was set, no push happened, no GitHub Actions
secret was rotated, no deploy-frontend workflow was dispatched. This
is a **billing decision that only the human can make**, per the CLAUDE.md
rule about infra-facing blockers.

The design 003 recommended (option c, HF cpu-basic + Supabase over 5432)
was correct in 2023 when radames' forum thread was written, but HF has
since paywalled Docker Spaces on the free tier. 003_report.md's
recommendation should be re-read with that new constraint in mind.

## Evidence

### Local secrets — all four present

Never printed; only lengths and last-4 chars shown.

```
$ for f in .secrets/{huggingface_token,groq_api_key,gemini_api_key,supabase_database_url}; do
    [ -f "$f" ] && printf '  PRESENT  %s  bytes=%d  last4=****%s\n' \
      "$f" "$(wc -c < "$f" | tr -d ' ')" "$(tail -c 5 "$f" | tr -d '\n')"
  done
  PRESENT  .secrets/huggingface_token  bytes=38  last4=****acZo
  PRESENT  .secrets/groq_api_key  bytes=57  last4=****DB1X
  PRESENT  .secrets/gemini_api_key  bytes=54  last4=****nAsA
  PRESENT  .secrets/supabase_database_url  bytes=115  last4=****gres
```

`bytes=38` for `huggingface_token` is `hf_` + 34 chars + trailing newline,
matching a standard HF user token.

### HF token identity + scope

```
$ HF_TOKEN=$(cat .secrets/huggingface_token | tr -d '\n\r') \
    backend/.venv/bin/python -c '
from huggingface_hub import HfApi; import os
who = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print(f"user={who[\"name\"]} type={who[\"type\"]}",
      f"token_role={who[\"auth\"][\"accessToken\"][\"role\"]}")'
user=Hridam type=user token_role=write
```

Token is valid, scoped `write`, owned by `Hridam` — matches
003_prompt.md's director update.

### The blocker — HF-side 402

```
$ backend/.venv/bin/python -c '
from huggingface_hub import create_repo; import os
create_repo("Hridam/helios", repo_type="space", space_sdk="docker",
            exist_ok=True, token=os.environ["HF_TOKEN"])'
```

Trimmed response body (verbatim):

```
huggingface_hub.errors.HfHubHTTPError: Client error '402 Payment Required'
for url 'https://huggingface.co/api/repos/create'
(Request ID: Root=1-6aba7975-19ea527e10b0337d00d514ea;
 180c828f-a7da-40ea-b88a-00b25cf4770c)

Static Spaces are free for everyone, but hosting Gradio and Docker
Spaces on free cpu-basic requires a PRO subscription. Subscribe at
https://huggingface.co/pro
```

The 402 is emitted by HF's `/api/repos/create` before the request
touches any hardware or storage. That means:

- **No Space exists** at `huggingface.co/spaces/Hridam/helios` after
  this run. No cleanup required.
- **No secrets or variables were sent to HF.** The four keys under
  `.secrets/` and the freshly-generated `JWT_SECRET_KEY` never left this
  machine's memory. The Python process exited before any
  `HfApi.add_space_secret(...)` call.
- **No git push to a Space remote** happened.

## What was NOT done (all deferred to a follow-up cycle)

Per the mailbox protocol's stop-clean rule on infra blockers, none of
the Part C or Part D steps ran:

- (C.1) Space creation — attempted, blocked.
- (C.1) Space secrets / variables — never called.
- (C.2) Space tree export + `git push hf main` — never called.
- (C.2) `get_space_runtime` polling — never called (no Space to poll).
- (C.3) `/api/v1/health` DB egress check — no Space to hit.
- (C.4) Smoke test (health, guest query, WebSocket, CORS preflight) —
  no live URL to hit.
- (D.1) `gh secret set VITE_API_URL` — deliberately not run. Rotating
  it now would break the current production frontend without any
  backend to point at.
- (D.2) `gh workflow run deploy-frontend.yml --ref feat/hf-space-deploy`
  — same reason, not run.
- (D.3) Live bundle verification — no new build.
- (D.4) `tests/smoke/test_live.py` — not authored this cycle. Nothing
  to smoke-test.

## Decisions the human needs to make

Three paths. Pick one, then a follow-up cycle can pick up the work
without re-running Part A.

### Path 1 — Subscribe to HF PRO ($9/mo)

Cheapest cash outlay if you want to keep the HF architecture unchanged.
`feat/hf-space-deploy` is ready to push as-is; a repeat of 005 will
succeed at `create_repo` and can walk through Parts C and D to
completion.

Trade-offs:
- Recurring cost.
- Still on ephemeral disk — Chroma vectors and any uploaded blobs
  vanish on every Space restart.
- HF's DDoS rules and health-check timeouts still apply; a cold-start
  that exceeds `startup_duration_timeout` (default 30 min) flips the
  Space to unhealthy.

### Path 2 — Ship on the Azure VM instead

`backend/deploy/vm/` from 002 is already complete and tested (bash -n
clean, README covers Azure NSG + workflow_dispatch). The human's
$100 Azure-for-Students credit funds cpu-basic-equivalent VMs for
months.

Trade-offs:
- Human provisions the VM manually (portal or `az` CLI).
- 4 GB VM needs the 2 GB swap file from 002_report.md's step 2.
- Real disk means Chroma and MinIO vectors/blobs persist.
- Requires DNS choice (DuckDNS vs static IP) — 002 defaulted to
  DuckDNS. All the code paths in `feat/hf-space-deploy` still work
  (HF-specific env vars just get ignored when running on a VM).

### Path 3 — Free CPU-only alternative to HF

Options that still fit the sync-Groq-plus-embedded-Chroma shape:
- **Render** — free tier still exists (`backend/deploy/render.yaml`
  already present). 750 h/mo free but the instance sleeps aggressively;
  Supabase over 5432 works.
- **Fly.io** — free tier is 3 shared-CPU VMs, WebSocket-friendly,
  Postgres addon available.
- **Railway** — hobby $5 credit / mo. No free tier since 2023.

Trade-offs: each provider needs new deploy tooling. `feat/hf-space-deploy`
would need to move again.

**Recommendation:** if Path 2 is realistic (Azure account already
in hand, willing to provision the VM), that's the cheapest and most
durable option. It uses code that's already merged-ready. Path 1 is
the fastest if you're happy to pay $9/mo and prefer HF's ergonomics.

## Rules-compliance checklist

- `git add -A` — not used this cycle. `git add` was called with
  explicit paths only. `frontend/node_modules` is not staged.
- Main branch — untouched. Nothing pushed anywhere except the
  `docs/mailbox/outbox/005_report.md` about to be committed to
  `fix/request-failed`.
- Secrets — none printed, none committed. The last-4 fingerprints
  above are the entire footprint the report has of the four secret
  files.

## Exact text the human should read next

**No live site to test yet.** Until one of the three paths above is
chosen and a follow-up cycle completes:

- `https://helios-hridam.vercel.app` still shows the "Cannot reach the
  Helios API at helios-hridam.ddns.net" message (thanks to the
  `humanReadableError` helper from 004 commit `91ed266`, once the
  Vercel bundle is rebuilt from `fix/request-failed` — which cannot
  happen until the backend is reachable, per Part D).
- The live bundle at
  `https://helios-hridam.vercel.app/assets/index-f7syGmmD.js` still
  contains the old `ddns.net` URL. That is unchanged from
  001_report.md.

**When the human has picked a path**, the next cycle should:

1. If Path 1: purchase HF PRO, then re-run **005** verbatim (this
   report will move to `outbox/005_report.md.old` — actually leave
   it in place, per the "never touch inbox files" rule; a new prompt
   should reference this SHA).
2. If Path 2: send a new prompt naming the Azure VM's public IP and
   confirming DUCKDNS_TOKEN + GROQ_API_KEY + GEMINI_API_KEY are ready
   to inject. The next cycle runs `backend/deploy/vm/first-deploy.sh`
   and then `rotate-secrets.sh`, per 002_report.md's command sequence.
3. If Path 3: send a new prompt naming the provider. `feat/hf-space-deploy`
   already has embedded Chroma, MINIO_ENABLED flag, OTel no-op, and
   the Supabase branch cherry-picks — the provider-specific bits are
   a Dockerfile / start.sh sibling to `backend/deploy/hf-space/`, plus
   a provider deploy CLI step.

Nothing on `feat/hf-space-deploy` needs to be reverted for any of the
three paths.
