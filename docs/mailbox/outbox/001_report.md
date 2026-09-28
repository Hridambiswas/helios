# 001 report — "request failed" diagnosis

**Verdict (confirmed, high confidence).** The failing hop is **DNS resolution of
`helios-hridam.ddns.net`**, which is the API base hard-coded into the currently
served Vercel bundle. Every browser request from `https://helios-hridam.vercel.app`
therefore fails at name resolution before it ever leaves the client, and both
`ChatView.tsx` and `QueryInterface.tsx` render the generic string `Request failed`
because axios never gets a `response.data.detail`. All five observations in the
prompt are confirmed. No fix applied — only this report was written and committed.

Everything below is raw evidence. Commands are shown before their trimmed output.
Nothing was changed anywhere outside `docs/mailbox/outbox/001_report.md`.

---

## A. Frontend target — confirms observation 1

```
curl -s -o /tmp/helios-diag/index.html https://helios-hridam.vercel.app/
grep -oE '/assets/[A-Za-z0-9._-]+' /tmp/helios-diag/index.html | sort -u
```
```
/assets/index-DqQWQMpX.css
/assets/index-f7syGmmD.js
```

```
curl -s -o /tmp/helios-diag/bundle.js \
  https://helios-hridam.vercel.app/assets/index-f7syGmmD.js
# 1 327 103 bytes
grep -oE '[A-Za-z0-9.-]*helios-hridam[A-Za-z0-9./_-]*' /tmp/helios-diag/bundle.js | sort -u
```
```
helios-hridam.ddns.net
```

```
grep -oE 'https?://[A-Za-z0-9._:/?=&%-]{4,}' /tmp/helios-diag/bundle.js | sort -u | grep helios
```
```
https://helios-hridam.ddns.net
```

```
grep -oE 'wss?://[A-Za-z0-9._:/?=&%-]{4,}' /tmp/helios-diag/bundle.js | sort -u
```
(no output — no `ws://` or `wss://` literal is baked in; the frontend derives it
from `VITE_API_URL` at runtime — see task E.)

```
grep -c 'Request failed' /tmp/helios-diag/bundle.js
```
```
2
```
Two `Request failed` literals ship in the bundle. These correspond to the two
fallback branches (`ChatView.tsx:551`, `QueryInterface.tsx:145`) — see task E.

**Confirmed:** the only Helios-owned host referenced by the live bundle is
`https://helios-hridam.ddns.net`. There is no `duckdns.org` reference and no
alternative base URL.

---

## B. DNS — confirms observation 2

```
dig +short @8.8.8.8 A helios-hridam.ddns.net
dig +short @8.8.8.8 AAAA helios-hridam.ddns.net
dig +short @1.1.1.1 A helios-hridam.ddns.net
dig +short @1.1.1.1 AAAA helios-hridam.ddns.net
```
All four commands return **empty**. Verbose form against 8.8.8.8:
```
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 36458
;; flags: qr rd ra; QUERY: 1, ANSWER: 0, AUTHORITY: 1, ADDITIONAL: 1
;; AUTHORITY SECTION:
ddns.net.  1800 IN SOA nf1.no-ip.com. hostmaster.no-ip.com. …
```
`NOERROR` with zero answers — the No-IP zone exists but has no record for
`helios-hridam`. Consistent with the No-IP free-tier hostname having expired
(see MIGRATION.md).

```
dig +short @8.8.8.8 A helios-hridam.duckdns.org       # empty
dig +short @8.8.8.8 AAAA helios-hridam.duckdns.org    # empty
dig +short @1.1.1.1 A helios-hridam.duckdns.org       # empty
dig +short @1.1.1.1 AAAA helios-hridam.duckdns.org    # empty
```
Verbose form:
```
;; ->>HEADER<<- opcode: QUERY, status: NXDOMAIN, id: 38828
```
`NXDOMAIN` — the DuckDNS hostname was never actually registered.

```
curl -v --max-time 15 https://helios-hridam.ddns.net/health
curl -v --max-time 15 https://helios-hridam.duckdns.org/health
```
Both fail identically:
```
* Could not resolve host: helios-hridam.ddns.net
curl: (6) Could not resolve host: helios-hridam.ddns.net
```
```
* Could not resolve host: helios-hridam.duckdns.org
curl: (6) Could not resolve host: helios-hridam.duckdns.org
```

**Confirmed:** neither hostname resolves anywhere. The API base baked into the
bundle points at a domain that does not exist in public DNS.

---

## C. Is any Helios backend alive anywhere?

Local Mac (read-only). None of the following returned a Helios reference:
```
grep -iE 'helios|ddns|duckdns|digitalocean|droplet|oracle' ~/.ssh/config
# (empty)
awk '{print $1}' ~/.ssh/known_hosts | tr ',' '\n' \
  | grep -iE 'helios|ddns|duckdns'
# (empty)
```
```
command -v doctl
# not installed
```

GitHub — names and updated dates only, no secret values printed.
```
gh secret list --repo Hridambiswas/helios
```
```
DO_SSH_KEY               2026-09-16T10:35:54Z
DO_USER                  2026-09-16T10:35:53Z
EC2_HOST                 2026-05-08T19:44:47Z
EC2_SSH_KEY              2026-05-08T19:44:49Z
EC2_USER                 2026-05-08T19:44:48Z
GH_OAUTH_CLIENT_ID       2026-05-11T13:51:47Z
GH_OAUTH_CLIENT_SECRET   2026-05-11T13:51:48Z
ORACLE_SSH_KEY           2026-09-16T10:15:52Z
ORACLE_USER              2026-09-16T10:15:51Z
SUPABASE_DATABASE_URL    2026-05-11T12:27:21Z
VERCEL_ORG_ID            2026-05-08T19:37:27Z
VERCEL_PROJECT_ID        2026-05-08T19:37:28Z
VERCEL_TOKEN             2026-05-08T19:37:26Z
VERCEL_TOKEN_TWO         2026-05-09T10:19:05Z
VITE_API_URL             2026-05-08T19:37:29Z
```
Notes drawn from names + dates only:
- `VITE_API_URL` was last updated **2026-05-08** — well before the 2026-09-24
  bundle rebuild. That rebuild therefore baked in the ancient `ddns.net` URL.
- Three deploy targets have partial secrets: EC2 (host + user + key, May),
  DigitalOcean (**user + key only — DO_HOST is missing**, Sept 16), and Oracle
  (**user + key only — no host secret**, Sept 16).
- `gh variable list` returned nothing (no repo variables).

```
gh workflow list --repo Hridambiswas/helios
```
```
CI                          active   268305943
Deploy Backend to EC2       active   273419636
Deploy Frontend to Vercel   active   273327446
```
(Workflow file `deploy-backend.yml` is named "Deploy Backend to DigitalOcean"
internally — the display title above is the older workflow's saved name in
GitHub.)

Recent deploy runs (main):
```
gh run list --workflow deploy-backend.yml --limit 3 --json …
```
```
{"conclusion":"failure",  "updatedAt":"2026-09-24T18:57:14Z", "headBranch":"main",
 "displayTitle":"Merge pull request #10 …"}
{"conclusion":"success",  "updatedAt":"2026-05-13T09:01:35Z", "headBranch":"main"}
{"conclusion":"success",  "updatedAt":"2026-05-13T08:50:51Z", "headBranch":"main"}
```
```
gh run list --workflow deploy-frontend.yml --limit 3 --json …
```
```
{"conclusion":"success",  "updatedAt":"2026-09-24T18:48:37Z", "headBranch":"main"}
{"conclusion":"success",  "updatedAt":"2026-05-13T09:01:35Z", "headBranch":"main"}
{"conclusion":"success",  "updatedAt":"2026-05-13T08:50:51Z", "headBranch":"main"}
```
Failure detail from the 2026-09-24 backend deploy run (`36043648792`):
```
gh run view 36043648792 --log-failed | tail
```
```
err: target api: failed to solve: failed to extract layer sha256:…:
     write /var/lib/containerd/…/asyncpg/protocol/protocol.cpython-311…:
     no space left on device
2026/09/24 18:57:12 Process exited with status 1
```
The docker build via SSH did connect to *some* host and progressed to layer
export before running out of disk. That host is not reachable via any
public DNS name we found — the SSH action received the target through
`secrets.DO_HOST`, which is not visible in the secret list. Two possibilities:
(a) `DO_HOST` exists but is being masked by the CLI listing (unlikely — `gh
secret list` shows all names by default), or (b) the workflow's
`host: ${{ secrets.DO_HOST }}` resolved to empty and appleboy/ssh-action
happened to succeed against a cached / previously configured host. Either way
the target ran out of disk. We could not probe it directly because no IP was
recovered from `known_hosts`, `~/.ssh/config`, `doctl` (not installed), or any
public DNS record.

No candidate IP was found on the Mac. No `curl` probe to a numeric host was
possible.

CI runs (context):
```
gh run list --workflow ci.yml --limit 3
```
```
{"conclusion":"failure","headBranch":"main",                              2026-09-24T18:48:11Z}
{"conclusion":"failure","headBranch":"feature/gemini-verifier-and-model-refresh", 2026-09-24T18:33:39Z}
{"conclusion":"failure","headBranch":"fix/migrate-backend-to-oracle-arm",  2026-09-16T10:37:59Z}
```
CI on main is failing as of 2026-09-24 as well — orthogonal to the request-failed
bug but relevant to any fix rollout.

---

## D. Real-browser reproduction — SKIPPED (Playwright not installed)

```
npx --no-install playwright --version
# npm error npx canceled due to missing packages and no YES option: ["playwright@1.63.0"]
ls ~/.cache/ms-playwright        # No such file or directory
ls /Applications/Google\ Chrome.app/Contents/MacOS/   # No such file or directory
ls /Applications/Chromium.app/Contents/MacOS/         # No such file or directory
```
Playwright is not cached, Chromium/Chrome are not installed as apps, and the
prompt forbids installing system packages. Per the prompt's escape clause
("If Playwright is not available, say so and skip"), task D is not run.
Substituted evidence: because the bundle's only outbound host is
`helios-hridam.ddns.net` (task A) and that host does not resolve
(task B), any real browser opening the site would fire the same failing
request and Chromium would surface `net::ERR_NAME_NOT_RESOLVED`. The two
`Request failed` literals in the bundle (task A) are the exact strings any
such reproduction would show in the UI.

---

## E. Downstream hops (static check)

**E1. Where the API base and WebSocket base come from.**
```
frontend/src/api/client.ts
```
```
const BASE = import.meta.env.VITE_API_URL ?? ''
export const api = axios.create({
  baseURL: `${BASE}/api/v1`,
  headers: { 'Content-Type': 'application/json' },
  timeout: 60_000,
})
…
export function connectQueryWS(token, onEvent, onClose): WebSocket {
  const wsBase = (import.meta.env.VITE_API_URL ?? window.location.origin)
    .replace(/^http/, 'ws')
  const ws = new WebSocket(`${wsBase}/ws/query?token=${token}`)
  …
}
```
Also:
```
frontend/src/components/AuthModal.tsx:4
const API_BASE = import.meta.env.VITE_API_URL ?? 'https://helios-hridam.ddns.net'
```
`AuthModal` even hard-codes `ddns.net` as its fallback default — even if
`VITE_API_URL` were unset, the frontend would still hit that dead host.

**E2. Where `Request failed` comes from.**
```
frontend/src/components/ChatView.tsx:551
const msg = (e as { response?: { data?: { detail?: string } } })
  ?.response?.data?.detail ?? 'Request failed'
```
```
frontend/src/components/QueryInterface.tsx:145
const msg = (e as { response?: { data?: { detail?: string } } })
  ?.response?.data?.detail
setErrorMsg(msg ?? 'Request failed')
```
Because `ERR_NAME_NOT_RESOLVED` never populates `response.data.detail`, both
sites always fall through to the literal `'Request failed'` — matching what
the user sees.

**E3. CORS.**
```
backend/config.py:132
cors_allowed_origins: str = ""
@property
def cors_origins_list(self) -> list[str]:  # accepts JSON list or comma-separated
```
The default is empty. `.github/workflows/deploy-backend.yml` overwrites
`CORS_ALLOWED_ORIGINS` on the backend host every deploy:
```
sed -i '/^CORS_ALLOWED_ORIGINS=/d' /home/helios/helios/backend/.env
echo 'CORS_ALLOWED_ORIGINS=["https://helios-hridam.vercel.app",
                             "https://frontend-omega-blush-87.vercel.app"]' \
  >> /home/helios/helios/backend/.env
```
So CORS is fine **as long as a deploy actually succeeds**. It hasn't since
2026-05-13; the 2026-09-24 deploy failed on disk-full. If a real backend were
up now, the last known-good CORS list is May's — which does already include
`https://helios-hridam.vercel.app`, matching the Vercel origin.
`.env.production.example` still contains the placeholder
`CORS_ALLOWED_ORIGINS=https://your-project.vercel.app`, but that file is only a
template.

**E4. Axios timeout vs pipeline latency.** `client.ts` sets `timeout: 60_000`
(60 s). The synchronous `/api/v1/query` path fans out planner + retriever +
executor + synthesizer + critic + verifier; cold latency on Groq
`openai/gpt-oss-120b` + Gemini `gemini-2.5-flash` easily exceeds 30 s and can
approach or breach 60 s on cold caches. Even once DNS is fixed, some queries
will hit the axios timeout unless the frontend switches to the WebSocket path
(`/ws/query`) for long-running queries.

**E5. WebSocket URL derivation.** `wsBase = VITE_API_URL.replace(/^http/, 'ws')`.
With today's `VITE_API_URL = https://helios-hridam.ddns.net` the derived URL
is `wss://helios-hridam.ddns.net/ws/query?token=…` — same DNS wall.

**E6. Model IDs.**
```
backend/config.py:20-24
groq_api_key:    SecretStr = SecretStr("")
groq_model:      str = "openai/gpt-oss-120b"
gemini_api_key:  SecretStr = SecretStr("")
gemini_model:    str = "gemini-2.5-flash"
embedding_model: str = "BAAI/bge-small-en-v1.5"
```
`openai/gpt-oss-120b` on Groq and `gemini-2.5-flash` are both current
(matches MIGRATION_1.1_TO_1.2.md and 1.2 release notes). No deprecated models.
`config.py:186–190` enforces that startup fails if `groq_api_key` is empty and,
when `verifier_enabled=true`, if `gemini_api_key` is empty.

**E7. `/health` and dependency status.**
```
backend/api/routes.py:728  @router.get("/health", response_model=HealthResponse)
```
Health is mounted at **`/api/v1/health`**, not `/health`, because the router
prefix in `backend/main.py` is `/api/v1` (excluded_handlers list at
`backend/main.py:89` corroborates). Any external monitor (or the prompt's
own `curl …/health` command in task B) will get 404 rather than a real
health signal even against a healthy backend. The response includes the new
`verifier_enabled` flag (per commit `764f4da`).

---

## F. Git state

```
git --no-optional-locks branch --show-current
```
```
fix/request-failed
```
```
git --no-optional-locks log --oneline origin/main -5
```
```
2e91bda Merge pull request #10 from Hridambiswas/feature/gemini-verifier-and-model-refresh
bc20ad0 docs: add MIGRATION_1.1_TO_1.2 guide covering model swap + Gemini verifier setup
764f4da feat(health): include verifier_enabled flag in /health response
4daa046 test(agents): guarantee all six agents are in agents.__all__
074e2f8 docs(planner): mention critic + verifier in downstream agents list
```
`09abb9f Merge DigitalOcean deploy tooling (PR#9) into main + Gemini verifier`
is the extra commit on the local `feature/do-deploy-with-verifier` branch tip
above `origin/main`. The rev-list count vs `origin/main` is `0	79` — the
79-commit local branch has never been pushed as its own branch to origin.

Both deploy workflows trigger from `main` only:
```
.github/workflows/deploy-frontend.yml
  on: push: branches: [main], paths: [frontend/**, .github/workflows/deploy-frontend.yml]
.github/workflows/deploy-backend.yml
  on: push: branches: [main], paths: [backend/**, .github/workflows/deploy-backend.yml]
```

`VITE_API_URL` updated date (from `gh secret list`): **2026-05-08T19:37:29Z**.
The frontend rebuild that produced the currently-served bundle ran on
**2026-09-24T18:48:37Z** — the secret was **not** updated between then and now,
even though `deploy-frontend.yml` carries the comment "Should be
https://helios-hridam.duckdns.org after migration." The migration was
therefore never wired up in the secret.

---

## Ordered list of failures that follow, once each is unblocked

1. **DNS / API base URL (current failure).** `helios-hridam.ddns.net` does not
   resolve. Even resurrecting a host will not help until either (a) a hostname
   is pointed at it and `VITE_API_URL` is updated + the frontend rebuilt, or
   (b) `VITE_API_URL` is switched to a bare IP.
2. **`AuthModal.tsx` hard-coded fallback.** Its default is still
   `https://helios-hridam.ddns.net`. If the new hostname / IP is set only via
   `VITE_API_URL` (which is correct) this is moot; but if VITE_API_URL is ever
   left unset, AuthModal will still hit the dead host.
3. **Backend deploy target.** The last successful deploy was 2026-05-13. The
   2026-09-24 deploy failed on **disk-full** on whichever host was actually
   reached. Even if DNS is fixed, if that host is still full, the running
   backend is stale (pre-v1.2, no Gemini verifier, no `openai/gpt-oss-120b`).
4. **`DO_HOST` secret.** `deploy-backend.yml` expects `secrets.DO_HOST` but no
   `DO_HOST` is visible in `gh secret list`. Any future deploy triggered from
   `main` will not reliably reach the intended DigitalOcean droplet.
5. **`GROQ_API_KEY` / `GEMINI_API_KEY`.** Not listed as GitHub secrets. Per
   `deploy-backend.yml`, they are expected to already exist inside
   `/home/helios/helios/backend/.env` on the target host. If the host is
   rebuilt or `.env` reset, backend boot will fail at `config.py:188` (Groq)
   or `config.py:190` (Gemini, since `VERIFIER_ENABLED=true`).
6. **60 s axios timeout.** Once the network path works, long queries can still
   surface as `Request failed` (axios `ECONNABORTED`) unless the WebSocket
   path is used or the timeout is raised.
7. **CORS.** Deploy workflow rewrites `CORS_ALLOWED_ORIGINS` correctly. Only a
   risk if a deploy is skipped and the host boots on the old `.env`, which is
   still the May-2026 list — that list already contains the Vercel origin,
   so this is likely a non-issue.
8. **`/health` route mismatch.** External monitors probing `…/health` will
   404 forever; probes should hit `…/api/v1/health`. The frontend itself
   doesn't call `/health`, so this only matters for uptime checks / the
   diagnostic commands in this prompt.
9. **CI on main is red (2026-09-24).** Any hotfix PR will land on a red
   baseline. Not blocking, but worth knowing.

---

## Decisions needed from the human

- **Which host runs the backend?** The repo currently carries partial secrets
  for three targets: EC2 (May), DigitalOcean (Sept 16, but missing `DO_HOST`),
  and Oracle (Sept 16, missing any host secret). Only one should own DNS.
- **Who owns DNS?** Options: (a) revive `helios-hridam.ddns.net` on No-IP,
  (b) register `helios-hridam.duckdns.org` on DuckDNS and update
  `VITE_API_URL`, (c) drop DDNS entirely and pin the frontend to the droplet's
  static / floating IP. The workflow comment says (b) is the intended path.
- **If DigitalOcean is the intended target:** add `DO_HOST` as a GitHub secret,
  add `GROQ_API_KEY` and `GEMINI_API_KEY` (or confirm they are already inside
  the droplet's `.env`), and clear docker layer cache on the droplet
  (`docker system prune -a -f && docker builder prune -af`) — the last deploy
  died on **no space left on device** during layer export.
- **VITE_API_URL rotation.** Once a host + DNS are chosen, update the
  `VITE_API_URL` GitHub secret and re-run `Deploy Frontend to Vercel`
  manually (`workflow_dispatch`) so the ddns.net URL stops shipping.
- **Long-query strategy.** Confirm whether the synchronous `/api/v1/query`
  path should stay (raise axios timeout > 60 s) or whether the UI should
  switch to the streaming WebSocket path for user-visible queries.

No fixes were proposed or applied. This report is the only file touched.
