# 003 report — Hugging Face Spaces feasibility for the Helios backend

Design + spike prompt. Nothing was pushed to any HF-related branch, no
Space was created, no GitHub secret was touched. `feat/hf-space-deploy`
was not created because the optional spike is not attempted this cycle
(justification in §Spike). This report was written on `fix/request-failed`.

001_report.md ruled the backend unreachable, so 002 wired up an Azure
path; that path is still uncertain because verification depends on the
human provisioning the VM. This report evaluates HF Spaces as the backup
host on the assumption the Azure path may fail or be delayed.

Uncertainty is called out inline as **[uncertain]**.

---

## §Hard constraint — HF Spaces outbound ports

Source: HF forum thread https://discuss.huggingface.co/t/open-port-for-space-to-connect-to-postgresql/29938
(the same link the director referenced), authored/replied to by user
`radames`, who identifies as HF staff.

Verbatim from the thread:

- **2023-01-18** (radames): *"currently only 80 and 443"* are open for
  outbound connections from Spaces.
- **2023-01-20** (radames): *"we've changed the rules and we'll enable
  5432, 27017 in addition to 80, 443."*
- **2023-05-04/05** (radames): 9200, 9243, 45454 opened, then 9243
  confirmed.

Consolidated allow-list as of that thread (still the most recent public
statement I found — HF's own docs do not enumerate outbound ports):

| Port | Purpose | Status |
| ---- | ------- | ------ |
| 80   | HTTP | open |
| 443  | HTTPS (Groq, Gemini, DuckDuckGo, S3-over-HTTPS) | open |
| 5432 | PostgreSQL | **open** |
| 9200/9243 | Elasticsearch | open |
| 27017 | MongoDB | open |
| 45454 | Papertrail | open |

**Not on the list — and therefore assumed blocked outbound:**

- **6379 — Redis.** No open port for Redis-native protocol. Redis Cloud
  over TLS on 6379 fails; over 443 needs a bespoke gateway (Upstash
  offers a REST-over-HTTPS API — see below).
- **6543 — Supabase transaction pooler.** The direct Postgres port
  (5432) IS open, so we can use Supabase's session-mode connection
  string; the pooler port is not needed.
- **Anything else outbound** (SMTP 25/465, Redis 6380, MQ 5672, ssh 22
  outbound). All assumed blocked.

**[uncertain]** — the forum thread is dated 2023. HF's docs do not
publish an authoritative current list, and I could not find a newer
staff statement. Treat 5432 as "very likely open" (multiple later
threads confirm PG-over-5432 working from Spaces) and 6379 as "not
found in any public allow-list" rather than "verified blocked."

---

## 1. Hard vs optional dependencies on the /api/v1/query path

Traced from `backend/main.py:29-51`, `backend/api/routes.py:141-238`
(sync `/query` handler), `backend/api/websocket.py:52-199` (streaming
path), and `backend/pipeline/run.py`.

| Service | Where it is touched on /query | Hard? |
| ------- | ----------------------------- | ----- |
| **PostgreSQL** | `main.py:43 create_tables()`, then per-request `routes.py:155-160` (write `QueryRecord` on start) and `routes.py:181-195` (update on completion) — **only inside `if current_user:`**. Not touched for guest queries. Also touched by every auth-required route (`/auth/*`, `/conversations`, `/query/history`). | Hard for authenticated flow; skippable for pure guest flow. |
| **Redis** | `routes.py:167 cache_get`, `routes.py:236 cache_set`, plus middleware: `api/middleware.py:47 RateLimitMiddleware`, `api/security.py AuthBruteForceMiddleware`, and `routes.py:58 _backpressure_guard` → `active_pipeline()` (all Redis-backed). `storage/cache.py:39-56` swallows Redis exceptions per call, but the middleware ping-list is not tolerant. | Hard. |
| **ChromaDB (HTTP client)** | `routes.py:39 upsert_batch/chroma_delete_batch` (ingest side) and, via the retriever, on every query. `retrieval/vector_store.py:16-38` opens `chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)` — HTTP to a separate Chroma container. | Hard for retrieval; **can be swapped** for `chromadb.PersistentClient` (embedded, same-process, no port). |
| **MinIO** | `routes.py:37 minio_ping/upload/delete`. Called on `/ingest` and `/health`. Not called on `/query`. `main.py:44 ensure_bucket()` at startup — would crash the app if MinIO is unreachable. | Hard at startup; not needed for /query steady-state. |
| **Celery worker + beat** | `routes.py:258 run_pipeline_task.delay` is on `/query/async` and `workers/beat_tasks.py`. Not touched by sync `/query` (which uses `asyncio.to_thread(run_pipeline, …)` at `routes.py:174`). | Optional for /query. Required for /query/async and periodic health tasks. |
| **OTel collector** | `observability/tracing.py setup_tracing()` at `main.py:35`. Exporter target `cfg.otel_exporter_otlp_endpoint` = `http://localhost:4317` by default; when unreachable, the OTLP exporter drops spans silently. | Fully optional. |
| **Prometheus** | Only scrapes `/metrics`; scrape target lives outside the container. `prometheus_fastapi_instrumentator` in `main.py:86` runs in-process regardless — it just exposes `/metrics`. | Fully optional. |

**Groq, Gemini, DuckDuckGo, Hugging Face Hub** — all reached over HTTPS (port
443), so they work inside a Space.

## 2. Can each optional dependency be disabled by config today?

| Service | Toggle today | Minimal change needed |
| ------- | ------------ | ---------------------- |
| **MinIO** | No env toggle. `main.py:44 ensure_bucket()` would raise at startup, and `/health` would always report `minio=False`. | Wrap the startup call in `try: ensure_bucket(); except: logger.warning(...)`, plus guard `routes.py:37 minio_ping` calls in `/health`. Small (~15 LOC). Ingest endpoint would 5xx but /query would work. |
| **Celery worker + beat** | Effectively yes: sync `/query` doesn't hit Celery, and `/query/async` is only invoked by clients that call it. There's no `CELERY_ENABLED` flag, but nothing forces the worker/beat processes to run. | None. Just don't start those two processes inside the Space. |
| **OTel** | Partially. Setting `OTEL_EXPORTER_OTLP_ENDPOINT` to an unreachable URL is silent-safe: the OTLP exporter buffers and drops. There's no explicit `OTEL_DISABLED` env. | None strictly required; a follow-up commit could no-op `setup_tracing()` if the env var is empty (`observability/tracing.py`). |
| **Chroma HTTP server** | No — `retrieval/vector_store.py:16-25` hard-codes `chromadb.HttpClient(host=..., port=...)`. | ~20 LOC: swap to `chromadb.PersistentClient(path="/tmp/chroma")`. Data ephemeral on Space restart. |
| **Postgres** | Partial — guest queries bypass it, but `main.py:43 create_tables()` and every auth route need it. | None if we keep an external Postgres reachable over port 5432 (e.g. Supabase — see §3(c)). |
| **Redis** | No. Rate-limit middleware calls Redis directly and raises if the ping fails; brute-force middleware likewise. `storage/cache.py` per-call try/except only covers cache misses, not middleware. | Either embed Redis inside the Space (see §3(a)) or feature-gate the two middlewares behind a `REDIS_URL` presence check (~30 LOC). |

## 3. Proposed single-container architecture

The Space is one Docker container (`sdk: docker`), listening on `app_port:
7860` — the only inbound port HF exposes (`spaces-sdks-docker` docs,
verbatim: *"you can also change the default exposed port 7860 by setting
`app_port: 7860`"*).

Three options evaluated, plus the recommended hybrid.

### (a) Everything embedded — Postgres + Redis + Chroma + FastAPI + Celery-in-thread under supervisord

Layout:
- `supervisord` starts:
  - `postgres` (PGDATA on `/tmp/pgdata` — HF ephemeral, `/data` not
    available at build time and only on paid Spaces)
  - `redis-server` (in-memory only)
  - `uvicorn main:app --host 0.0.0.0 --port 7860`
  - (no Celery worker; sync `/query` only)
- `retrieval/vector_store.py` swapped to `chromadb.PersistentClient(path="/tmp/chroma")`.
- MinIO stubbed out with a local-fs adapter or ingest disabled.

**What breaks / what's lost on restart:**
- **All user accounts** (`users` table), **conversation history**
  (`conversations`, `messages`), **query history** (`query_records`), and
  **ingested documents** (`document_records` + Chroma vectors + MinIO
  blobs) reset to empty on every restart. Free-tier Spaces sleep after
  ~48h idle and restart on the next request.
- OAuth GitHub sign-ins still work per request but the resulting user row
  vanishes on the next restart.
- The 5-minute guest-query cache in Redis obviously also resets.
- Alembic migrations run against a fresh PG on every cold start, adding
  ~5–15 s to boot.

**Memory estimate (against 16 GB ceiling on cpu-basic):**
- Ubuntu base + supervisord + logging: ~200 MB
- postgres 16-alpine (default shared_buffers 128 MB, connections 20):
  ~250 MB
- redis (maxmemory 128 MB): ~150 MB peak
- Python 3.11 + FastAPI + SQLAlchemy + Celery client libs: ~200 MB
- BAAI/bge-small-en-v1.5 (fastembed, on CPU): ~150 MB resident
- CLIP ViT-B-32 (torch + transformers, CPU): ~600 MB resident
- Chroma PersistentClient + duckdb + working set (small collections): ~250 MB
- Groq/Gemini SDK request buffers: ~50 MB
- **Sum steady-state: ~1.85 GB. Burst under a query: ~2.5 GB. 16 GB
  ceiling → ample headroom.**

**Cold start time [uncertain]:**
- Container pull + start: HF caches the built image, so this is fast (a
  few seconds after build).
- On first request after a restart:
  - postgres initdb (if PGDATA blank): ~5–8 s
  - alembic upgrade head: ~5–10 s
  - Redis start: ~1 s
  - FastEmbed model warm-up (already cached in image per
    `Dockerfile.prod:15`): ~1–2 s
  - CLIP model load into RAM: ~4–8 s (already cached in image if
    preloaded)
  - **Total to first successful /query: ~25–40 s**. Well inside HF's
    default 30-minute startup timeout.

### (b) SQLite instead of Postgres

Storage layout is the same as (a) except `postgres` → SQLite file at
`/tmp/helios.sqlite`. Downsides:

- `storage/database.py:35-45` picks `postgres+asyncpg` or Supabase.
  Switching to SQLite requires `sqlite+aiosqlite` and rewriting
  `_effective_database_url()` at minimum.
- Alembic migrations were authored for PostgreSQL. `storage/models.py`
  (not read this cycle but implied by the schema shape) may use
  `JSONB`, `ARRAY`, `UUID` PG-native types — those need `sqlite_json1`
  substitutions per column. **[uncertain]** — did not verify each
  column type, but a fair-quality refactor typically takes 100+ LOC and
  test churn.
- Data still ephemeral (SQLite file on `/tmp`).
- Memory drop vs (a): ~250 MB (no Postgres). Cold start drop: ~10–15 s.
- Overall a lot of one-off effort for a marginal win, given (c) exists.

### (c) External Supabase over 5432 (the recommended cut)

**5432 is on the HF outbound allow-list.** `backend/config.py:44-45`
already has `supabase_database_url`, and `storage/database.py`
`_effective_database_url()` (read this cycle) already routes to Supabase
when set, mapping `postgresql://` to `postgresql+asyncpg://` and forcing
SSL. `origin/feature/supabase-integration` (last commit `a4b9326`) adds
`wait_for_db` retry logic and `connection_info()` for `/health` — worth
cherry-picking if we go this route.

Recommend: **Supabase for Postgres over 5432, everything else (Redis
in-mem, Chroma embedded, models in-process) inside the Space container.**

- Data persistence: users, history, and ingested-doc metadata all
  survive Space restarts. Only the **Chroma vectors and any file blobs**
  are lost on restart — acceptable if we re-ingest on demand or ship
  a periodic reindex task.
- Memory: ~1.6 GB steady, ~2.3 GB burst (no local postgres). Well under
  16 GB.
- Cold start: no initdb, no alembic against fresh DB. First /query
  in ~10–15 s.
- Ingest still degraded: MinIO isn't reachable, so file uploads either
  live on `/tmp` (lost on restart) or need an external S3 (over 443 —
  fine). Either way — call `/ingest` disabled for cycle-1 and turn it
  on later.

**Recommendation: option (c).** Both (a) and (b) throw away user data on
restart, which is unnecessary since Supabase-over-5432 is available.

## 4. Executor sandbox — safe inside a Space

`agents/executor.py:1-90` (read this cycle):

- No docker-in-docker: the sandbox is pure-Python (`ast.parse`, walked
  by `_ImportGuard`, then executed in-process with a whitelisted
  builtins dict and stdout capture via `contextlib.redirect_stdout`).
- No process fork: `_run` runs the compiled AST on a Python thread with
  a soft timeout (`threading.Thread` + a `_TimeoutError` bailout).
- No filesystem writes required — no `open`/`exec` are exposed because
  they're in `_FORBIDDEN_BUILTINS`.
- `_ALLOWED_IMPORTS` covers numpy/pandas/scipy but these are heavy —
  they need to be installed inside the Space's Python env at build time
  (i.e. carried through `requirements.txt`). That's already how
  `Dockerfile.prod` builds.

Runs fine as UID 1000 (no capabilities needed). No further code changes
required for the Space.

## 5. Space port, WebSockets, and URL

- `app_port` defaults to `7860` (HF docs verbatim); set explicitly in the
  Space `README.md` YAML.
- Container listens on `0.0.0.0:7860`. HF's fronting proxy terminates
  TLS on `https://<user>-<space>.hf.space` and forwards to the app.
- Public URL for `Hridam/helios` with lowercased user handle:
  **`https://hridam-helios.hf.space`** (confirmed in the director's
  update at the bottom of 003_prompt.md).
- **WebSockets:** HF's proxy supports WebSocket upgrades — Gradio-SDK
  Spaces (and many custom Docker Spaces) run WebSocket streams through
  it and I could not find a documented port/path restriction. **[low
  uncertain]** — HF's docs do not explicitly say "WebSockets are
  supported" in the `spaces-sdks-docker` page I read, but the Gradio
  precedent is strong evidence they work. The frontend already sets
  `wsBase = VITE_API_URL.replace(/^http/, 'ws')`, so
  `wss://hridam-helios.hf.space/ws/query?token=…` should just work once
  `VITE_API_URL=https://hridam-helios.hf.space`.

Nginx (per `docker-compose.prod.yml`, TLS-terminating) is **not needed
inside the Space** — HF's proxy already terminates TLS. Drop nginx in
the Space Dockerfile.

## 6. CORS and VITE_API_URL — exact values

- Space env / Dockerfile ENV:
  ```
  CORS_ALLOWED_ORIGINS=["https://helios-hridam.vercel.app","https://frontend-omega-blush-87.vercel.app"]
  ```
  Same JSON-list format the deploy workflow writes today (verified by
  the CORS test added in 002 commit `3275a6f`).
- GitHub Actions secret (used by the frontend build):
  ```
  VITE_API_URL=https://hridam-helios.hf.space
  ```
- Frontend WebSocket URL is derived automatically from `VITE_API_URL`
  (`frontend/src/api/client.ts:141`), no separate secret needed.
- `OAUTH_BACKEND_URL` for OAuth callback:
  `https://hridam-helios.hf.space` (overrides the duckdns default set in
  002 commit `1635bf6`).
- GitHub OAuth app settings on the human's side: add
  `https://hridam-helios.hf.space/auth/github/callback` to the
  "Authorization callback URL" list on the OAuth app config page.

## 7. Space secrets (names only)

Set these in the Space's Settings → Secrets tab (never in the
Dockerfile). Non-secret runtime values (LOG_LEVEL, APP_ENV, CORS list)
go in Settings → Variables, or bake into the Dockerfile ENV.

Secrets:
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `JWT_SECRET_KEY`
- `SUPABASE_DATABASE_URL`
- `GITHUB_CLIENT_ID`
- `GITHUB_CLIENT_SECRET`

Variables (Settings → Variables, not secrets — they end up in the log
context):
- `APP_ENV=production`
- `VERIFIER_ENABLED=true`
- `LOG_LEVEL=INFO`
- `LOG_FORMAT=json`
- `CORS_ALLOWED_ORIGINS` (JSON list, see §6)
- `OAUTH_FRONTEND_URL=https://helios-hridam.vercel.app`
- `OAUTH_BACKEND_URL=https://hridam-helios.hf.space`
- `REDIS_URL=redis://127.0.0.1:6379/0` (embedded Redis inside the Space)
- `CHROMA_PATH=/tmp/chroma` (once `retrieval/vector_store.py` supports the
  PersistentClient path — code change deferred to 004)
- (MinIO — omit; `/ingest` disabled cycle-1)

## 8. Dockerfile.prod issues on Spaces

`backend/Dockerfile.prod` (18 lines, read this cycle):

| Line | Issue on HF | Fix in 004 |
| ---- | ----------- | ---------- |
| 17 | `adduser --disabled-password --gecos "" helios` doesn't pin UID. HF's `spaces-sdks-docker` docs say **"container runs with user ID 1000"** and require `useradd -m -u 1000 user`. If our helios user lands on UID 1001, files owned by 1001 aren't readable by the process running as 1000. | Replace with `RUN useradd -m -u 1000 user` and set `USER user`. |
| 15 | `RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"` — HF's docs suggest the preferred way is `preload_from_hub` in the README YAML, but a `RUN` step is also fine as long as we don't hit outbound-port blocks (HF Hub is on 443, so this works). | Keep, but consider `preload_from_hub` for CLIP too. |
| 27 | `CMD ["sh", "-c", "alembic upgrade head && uvicorn …"]` — with Supabase, this works. With embedded postgres, we'd need `sh -c "supervisord -c /etc/supervisor/supervisord.conf"` and let supervisord start postgres, wait for it, run alembic, then uvicorn. | Under option (c), keep alembic upgrade in CMD but add a `wait_for_db` (already exists on `origin/feature/supabase-integration`) so we don't race Supabase cold-start. |
| 17 | `COPY . .` before `chown -R helios:helios /app` — HF warns explicitly against `chown -R` after `COPY` because it doubles the image layer size. | Switch to `COPY --chown=user . $HOME/app` per HF docs. |
| 6 | `apt-get install gcc g++ libpq-dev` — these are needed for asyncpg / psycopg2 wheels only when a prebuilt wheel isn't available. They stay ~100 MB in the final image. | Consider a multi-stage build: build stage installs `-dev` packages, runtime stage copies the wheels. Optional; not blocking. |

## Spike — not attempted this cycle

The prompt marks the spike optional ("only if cheap"). Assessment:

- The workstation has `docker` available in principle, but the `vercel:*`
  and various MCP tools indicate a busy environment. A local `docker
  build` of `backend/Dockerfile.prod` pulls python:3.11-slim + rebuilds
  the fastembed model into the image and installs the full
  `requirements.txt` (probably 500+ MB of wheels including torch and
  chromadb). On a Mac laptop this typically takes 5–15 min the first
  time.
- Running the built image with `docker run -p 7860:7860` would also
  need a running Postgres reachable from the container, plus a Redis,
  plus a Chroma — all the pieces we'd be rearranging under option (c).
  Reproducing the intended production layout locally is a mini-project
  in itself.
- The `deploy/hf-space/Dockerfile` design becomes clearer AFTER we know
  which of §3's options we're committing to. Doing the spike before
  that decision would prototype the wrong architecture.

**Recommendation:** run the spike as the very first commit of 004 once
the human has confirmed the architecture recommendation (option c).
That way the Dockerfile is designed against the chosen shape rather than
being thrown away.

## Ordered implementation plan for 004 (one commit per step)

Assumes the human has:
- Created `Hridam/helios` Space (Docker SDK, cpu-basic hardware).
- Added the six secrets from §7 in Settings → Secrets.
- Added the ten variables from §7 in Settings → Variables.
- Created a Supabase project and put its session-mode connection string
  in `SUPABASE_DATABASE_URL`.
- Extended the GitHub OAuth app to include the HF callback URL.

Then, on a new branch `feat/hf-space-deploy` cut from `fix/request-failed`:

1. `feat(retrieval): support embedded Chroma via CHROMA_PATH env` —
   `retrieval/vector_store.py` switches to `PersistentClient` when
   `CHROMA_PATH` is set, otherwise keeps the HTTP client. Add a unit
   test that seeds a small collection under `tmp_path` and reads it
   back.
2. `feat(storage): skip MinIO in /health when disabled` — guard
   `ensure_bucket()` in `main.py` and `minio_ping` in `/health` behind a
   `MINIO_ENABLED` flag (default true). `/ingest` returns 503 when
   disabled with a clear message.
3. `feat(config): allow REDIS_URL=embedded to co-run Redis in the Space` —
   trivial re-init to run `redis-server` as a background process via
   supervisord (or a `start.sh` wrapper). Rate-limit + brute-force
   middlewares continue to work unchanged.
4. `feat(observability): no-op tracing when OTEL endpoint empty` — small
   guard around `setup_tracing()`.
5. `feat(deploy): add deploy/hf-space/{Dockerfile,start.sh,README.md}` —
   the actual Space Dockerfile (UID-1000 user, `preload_from_hub` for
   models, `wait_for_db` for Supabase, single-container start.sh under
   supervisord). Cherry-pick `a4b9326` (`connection_info()`) and
   `705087d` (`wait_for_db`) from `origin/feature/supabase-integration`.
6. `docs(deploy): 003b — HF Space setup guide` — human-facing walkthrough
   with the exact secrets, variables, Space YAML, OAuth callback URL,
   and the `git remote add hf https://huggingface.co/spaces/Hridam/helios`
   push instruction.
7. `ci(deploy-frontend): dispatch on merges to feat/hf-space-deploy` —
   opt in the branch to a preview build with `VITE_API_URL=https://hridam-helios.hf.space`.
8. `test(hf-space): smoke script for the Space public URL` — a small
   pytest-based script that hits `/api/v1/health`, `/api/v1/stats`, one
   `/api/v1/query` (via HTTP), and a WebSocket `/ws/query` echo. Guarded
   by an env var so CI doesn't hit the Space by default.

Each step passes `pytest -q` locally.

## What the human must do by hand (Space provisioning)

1. Log in to Hugging Face and create Space `Hridam/helios`:
   - SDK: **Docker**
   - Hardware: **cpu-basic (free)**
   - Visibility: public (or private — either works; private just gates
     the browser preview of the Space page, not the API).
2. In Settings → Variables + Secrets, add the six secrets and ten
   variables from §7. `JWT_SECRET_KEY` should be freshly generated with
   `python -c "import secrets; print(secrets.token_hex(32))"`.
3. Create a Supabase project (any region — session mode works over
   5432). Copy the connection string with the format
   `postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`
   into the `SUPABASE_DATABASE_URL` **secret**. Confirm SSL is enforced
   by Supabase (it is by default).
4. On the GitHub OAuth app at
   https://github.com/settings/developers → Helios app → add
   `https://hridam-helios.hf.space/auth/github/callback` to the
   "Authorization callback URLs" list. Existing Vercel + Azure/DDNS
   entries stay.
5. On the Mac, log in to HF once so `git push hf` works without
   password: `hf auth login` (or the older `huggingface-cli login`).
   The token is not needed anywhere else and this report never sees it.
6. After 004 lands `deploy/hf-space/Dockerfile`, push the current
   checkout to the Space:
   ```
   git remote add hf https://huggingface.co/spaces/Hridam/helios
   git push hf feat/hf-space-deploy:main
   ```
   HF picks up the push, runs `docker build`, and starts the Space.
7. Verify from the laptop:
   ```
   curl -fsS https://hridam-helios.hf.space/api/v1/health | jq
   ```
   Then rotate `VITE_API_URL` and rebuild the frontend (identical to
   step 3 in 002_report.md's command sequence, but with the HF URL).

If Azure comes online first, none of the above is wasted — the HF path
becomes a warm standby with the same code.

## Files read this cycle (no edits)

`backend/main.py`, `backend/api/routes.py` (health + query bodies),
`backend/api/websocket.py`, `backend/pipeline/run.py`,
`backend/agents/executor.py`, `backend/storage/database.py`,
`backend/storage/object_store.py`, `backend/storage/cache.py`,
`backend/storage/read_replica.py`, `backend/retrieval/vector_store.py`,
`backend/retrieval/clip_encoder.py`, `backend/workers/celery_app.py`,
`backend/config.py`, `backend/docker-compose.yml`,
`backend/docker-compose.prod.yml`, `backend/Dockerfile.prod`, plus the
GitHub log for `origin/feature/supabase-integration`. Web fetches:
https://discuss.huggingface.co/t/open-port-for-space-to-connect-to-postgresql/29938,
https://huggingface.co/docs/hub/spaces-config-reference,
https://huggingface.co/docs/hub/en/spaces-sdks-docker.
