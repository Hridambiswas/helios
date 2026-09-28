# 001: Diagnose "request failed" end to end. CHANGE NOTHING.

Pure diagnosis. No code, config, secret or infra changes. Only the report file is committed.

## What the director already observed (verify, do not trust)
1. The live bundle https://helios-hridam.vercel.app/assets/index-*.js hardcodes the API base
   `https://helios-hridam.ddns.net` (VITE_API_URL at build time). Last modified 24 Sep 2026.
2. Public DNS (dns.google, cloudflare-dns) returns NO A/AAAA record for
   `helios-hridam.ddns.net` and NXDOMAIN for `helios-hridam.duckdns.org`.
3. MIGRATION.md itself says ddns.net no longer resolves and the old EC2 IP is gone.
   The DigitalOcean + DuckDNS migration lives on `feature/do-deploy-with-verifier`
   (79 commits ahead of origin/main, never merged), so it was likely never deployed.
4. `ChatView.tsx:551` and `QueryInterface.tsx:145` fall back to the literal string
   'Request failed' when there is no `response.data.detail`, i.e. every network error
   (DNS, CORS, timeout, TLS) looks identical in the UI.
5. `backend/config.py`: `cors_allowed_origins` defaults to "" and production uses
   `cors_origins_list`, so a prod backend with no CORS env would also block the browser.
Hypothesis: the UI never reaches any backend. Backend rebuilds could not have changed the symptom.

## Tasks (collect raw evidence for each)
A. Frontend target: download the live index.html and bundle, grep the API base and
   the WebSocket base. Confirm or refute observation 1.
B. DNS: `dig +short A` and `AAAA` for both hostnames, against 8.8.8.8 and 1.1.1.1.
   `curl -v --max-time 15 https://<host>/health` for both. Paste output.
C. Is ANY Helios backend alive anywhere? Search the Mac, read only, for evidence of the
   current host: `~/.ssh/config`, `~/.ssh/known_hosts` (hostnames only), `doctl`
   (`doctl compute droplet list` if installed and authed), `gh secret list` and
   `gh variable list` on Hridambiswas/helios (names and updated dates only),
   `gh run list --limit 20` for both deploy workflows with conclusions and branches.
   If a candidate IP is found, `curl -v -k --max-time 15 https://<ip>/health` and
   `http://<ip>/health`. Never print secret values.
D. Reproduce in a real browser: run a headless Playwright/Chromium script (npx playwright
   is fine, use the installed Chromium if present) that opens the Vercel site, submits one
   question, and records the failing network request: URL, error text
   (e.g. net::ERR_NAME_NOT_RESOLVED), and console errors. If Playwright is not
   available, say so and skip; do not install system packages.
E. Downstream hops, static check only (so we know what fails next once DNS is fixed):
   CORS origins in `backend/.env.production.example` and deploy scripts vs the Vercel
   domain; axios timeout (60s) vs expected pipeline latency; WebSocket URL derivation;
   model IDs in config (`groq_model`, `gemini_model`, embedding) and whether any are
   known deprecated; whether `/health` exposes dependency status (Redis, Postgres,
   Celery, Chroma).
F. Git state: current branch, `git --no-optional-locks log --oneline origin/main -5`,
   which branch the frontend and backend deploy workflows trigger from, and whether
   the VITE_API_URL secret was updated after 24 Sep (from gh secret list dates).

## Report format (docs/mailbox/outbox/001_report.md)
- Verdict line: the exact failing hop, confirmed or refuted, with confidence.
- Evidence per task A to F (commands and trimmed raw output).
- Ordered list of every hop that will fail after the first one is fixed.
- Decisions needed from the human (for example: which host to run the backend on,
  who owns DNS), stated plainly. Do not propose or perform fixes yet.

Commit only the report on `fix/request-failed`, push that branch. Never touch main.
