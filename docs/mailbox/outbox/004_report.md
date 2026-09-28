# 004 report — HF Space deploy (Part A complete, Parts B/C/D BLOCKED)

**Status: PARTIAL.** Part A is complete and pushed to
`feat/hf-space-deploy`. Parts B (local docker build), C (create + push
the Space), and D (point the frontend at the Space) are BLOCKED as
allowed by the prompt.

## Blockers

- **Part B** — no Docker installed on this Mac. `command -v docker`
  returns nothing, and the prompt explicitly says "If Docker is
  unavailable … skip and say so."
- **Part C** — three of the four secret files are missing under
  `.secrets/`, so the prompt's own escape clause fires ("If any is
  missing when you reach Part C, do Parts A and B, write a PARTIAL
  report listing exactly which files are missing, and stop"). Concretely:

  | File | Status | Size |
  | ---- | ------ | ---- |
  | `.secrets/huggingface_token` | **present** | 38 bytes |
  | `.secrets/groq_api_key` | **missing** | — |
  | `.secrets/gemini_api_key` | **missing** | — |
  | `.secrets/supabase_database_url` | **missing** | — |

  Nothing was printed from `huggingface_token` and no other secret
  content was ever inspected or copied.

- **Part D** — depends on C being green, so also skipped.

## Part A — commits on `feat/hf-space-deploy`

Branch cut from `fix/request-failed` at commit `7886ea8`. Pushed to
`origin/feat/hf-space-deploy`.

| SHA | Subject |
|-----|---------|
| `12e6e58` | feat(retrieval): support embedded Chroma via CHROMA_PATH |
| `eed9a28` | feat(minio): honour MINIO_ENABLED at startup, /health and /ingest |
| `b523099` | feat(observability): no-op tracer when OTEL_EXPORTER_OTLP_ENDPOINT empty |
| `59b721a` | feat(storage): cherry-pick wait_for_db + connection_info from supabase branch |
| `0f352df` | feat(deploy): add backend/deploy/hf-space/ (Dockerfile + start.sh + README) |
| `91ed266` | fix(frontend): surface real error causes instead of 'Request failed' |
| `93b082b` | fix(frontend): drop hardcoded ddns.net fallback in AuthModal |
| `4ca0a56` | fix(main): only wait_for_db when SUPABASE_DATABASE_URL is set |

Deviations from the prompt's step numbering:

- **Prompt step 1** (five sub-steps: Chroma, MinIO flag, embedded Redis,
  OTel no-op, deploy/hf-space/) landed as six commits: `12e6e58`,
  `eed9a28`, `b523099`, `59b721a`, `0f352df`. The embedded-Redis
  requirement is realised inside `0f352df` (start.sh runs
  `redis-server 127.0.0.1:6379` next to uvicorn) rather than as a
  standalone commit — Redis embedding is a deploy-side concern, not a
  runtime code change.
- **Bundled with prompt step 3** (`config.oauth_backend_url` update):
  the default was flipped from the duckdns host to
  `https://hridam-helios.hf.space` inside `12e6e58` because that
  commit is where the HF Space shape is first committed. Called out
  explicitly rather than dropping it in a doc-only commit.
- **`4ca0a56`** is not one of the prompt's numbered steps — it is a
  regression fix. `wait_for_db()` from `59b721a` broke
  `test_user_connection_counter_increments_and_decrements` because the
  test lifespan uses SQLite via `AsyncEngine` and `ping()` cannot run
  `SELECT 1` cleanly against that ('Can't use AsyncSession.execute()
  with a server-side cursor'). Guarding `wait_for_db` behind
  `cfg.supabase_database_url` restores the test and keeps the retry
  logic where it matters.

### Cherry-pick conflict handling

`origin/feature/supabase-integration`'s `705087d` and `a4b9326` touch
`storage/database.py`, but the current repo layout has
`backend/storage/database.py` (May 2026 restructure moved everything
under `backend/`). A straight `git cherry-pick` fails on path only.
Applied the diffs by hand in `59b721a` — both commits are pure
additions in the same file, no other file touched — and left a paper
trail in the commit message so future readers can verify the delta
matches the two source commits.

### Part A test baseline

- **Backend (`cd backend && .venv/bin/python -m pytest -q`)** —
  produced a fresh `backend/.venv` per Part A step 0, `.gitignore`
  already excludes it. `pip install -r requirements.txt` took 3m32s.
  - Pre-change baseline (post-venv, on the branch tip before edits):
    **8 failed, 164 passed, 2 warnings** in 170.96s.
  - Post-change (branch tip after `4ca0a56`): **8 failed, 167 passed,
    2 warnings** in 244.97s. Three new passes are the CHROMA_PATH
    tests I added (`test_vector_store_persistent.py`). No net
    regressions.
  - The eight remaining failures are unrelated to this work:
    ```
    tests/test_api.py::TestConversationRoutes::test_list_conversations_returns_empty_list
    tests/test_pipeline.py::TestConversationHistory::test_empty_history_is_default
    tests/test_pipeline.py::TestConversationHistory::test_history_propagated_to_state
    tests/test_pipeline.py::TestPipelineRouting::test_full_pipeline_happy_path
    tests/test_pipeline.py::TestRetryLoop::test_critic_fail_triggers_second_synthesizer_call
    tests/test_pipeline.py::TestRetryLoop::test_retry_capped_at_max_retries
    tests/test_resilience.py::TestBackpressure::test_allows_requests_below_pipeline_threshold
    tests/test_resilience.py::TestBackpressure::test_raises_when_pipeline_limit_reached
    ```
    Left in place per the prompt ("any remaining failures unrelated to
    this work: list them, don't fix").

- **Frontend (`npm ci && npm run build`)** — `node_modules/` was
  missing several `@react-three/*` packages, so I ran `npm ci` first
  (17s, 320 packages added). `npm run build` then passed cleanly:
  ```
  ✓ 2114 modules transformed.
  dist/assets/index-Dypebg_J.js   1,328.07 kB │ gzip: 382.70 kB
  ✓ built in 3.04s
  ```
  The new bundle contains **zero** references to `ddns.net`
  (`grep -c 'ddns.net' frontend/dist/assets/*.js` → `0`).

### File-level evidence

- `backend/config.py`: adds `chroma_path`, `minio_enabled`; flips
  `oauth_backend_url` default; `validate_secrets()` now skips
  POSTGRES_PASSWORD when SUPABASE is configured and MINIO_* keys when
  MinIO is disabled.
- `backend/retrieval/vector_store.py`: `_get_collection` opens
  `chromadb.PersistentClient(path=cfg.chroma_path)` when set,
  otherwise the previous HttpClient path. Added `_reset_for_tests()`
  so unit tests can swap CHROMA_PATH per case.
- `backend/main.py`: skips `ensure_bucket()` and logs a warning when
  `MINIO_ENABLED=false`; awaits `wait_for_db()` before `create_tables()`
  only when Supabase URL is set.
- `backend/api/routes.py`: `/ingest` returns 503 with a clear message
  when `MINIO_ENABLED=false`; `/health` swaps its MinIO ping for
  `asyncio.sleep(0, result=True)` in that mode so overall status is
  not falsely flipped to degraded.
- `backend/observability/tracing.py`: only installs the OTLP
  BatchSpanProcessor when the endpoint is non-empty; logs "no-op mode"
  otherwise. TracerProvider is still installed so `span()` continues
  to work.
- `backend/storage/database.py`: adds `wait_for_db()` (exponential
  backoff for Supabase pooler cold-starts) and `connection_info()`
  (credential-masked URL + pool stats for `/health` diagnostics —
  read-only helper, not wired into `/health` in this cycle to keep
  the diff minimal).
- `backend/deploy/hf-space/{Dockerfile,start.sh,README.md,.dockerignore}`:
  documented in each file. Dockerfile uses `useradd -m -u 1000 user`
  and `COPY --chown=user` per HF's guidance, pre-warms fastembed at
  build time, and pins env defaults for the HF-shape deploy
  (`CHROMA_PATH=/tmp/chroma`, `MINIO_ENABLED=false`,
  `OTEL_EXPORTER_OTLP_ENDPOINT=""`, `REDIS_URL=redis://127.0.0.1:6379/0`).
  README has the Space YAML front matter (`sdk: docker`,
  `app_port: 7860`) at the top.
- `backend/tests/test_vector_store_persistent.py`: three tests
  covering the CHROMA_PATH branch — passes locally.
- `frontend/src/api/client.ts`: adds `humanReadableError()` that
  distinguishes DNS/TLS/network failures from HTTP 4xx/5xx and axios
  timeouts; `console.error`s the raw error for DevTools. Exports
  `BASE` so components share one URL.
- `frontend/src/components/{ChatView,QueryInterface}.tsx`: replace
  the literal `'Request failed'` fallback with `humanReadableError(e)`.
- `frontend/src/components/AuthModal.tsx`: drops the hard-coded
  `'https://helios-hridam.ddns.net'` fallback; imports `BASE` from
  `api/client`.

## Part B — SKIPPED

`command -v docker` returns nothing on this Mac. Per the prompt ("If
Docker is unavailable or the build exceeds 20 minutes, skip and say
so") no image build was attempted. The Dockerfile has been validated
statically:
- `useradd -m -u 1000 user` before `COPY --chown=user` — HF's exact
  pattern.
- Only `redis-server` and `curl` are the extra runtime binaries beyond
  Python — small footprint.
- `HEALTHCHECK` points at `http://localhost:7860/api/v1/health` — the
  right port + prefix combo for the Space.
- `start.sh` passes `bash -n` cleanly and has a signal-forwarding
  cleanup trap on INT/TERM.

Once the human runs `Docker` locally (or bumps to Colima / OrbStack),
the build+run can be exercised with:
```bash
export GROQ_API_KEY=<...>
export GEMINI_API_KEY=<...>
export SUPABASE_DATABASE_URL=<...>
export JWT_SECRET_KEY=$(python -c 'import secrets;print(secrets.token_hex(32))')

# Materialise the Space tree (mirrors Part C step 3 without the push)
tmpdir=$(mktemp -d)
cp backend/deploy/hf-space/{Dockerfile,start.sh,README.md,.dockerignore} "$tmpdir/"
cp -r backend/* "$tmpdir/"
rm -rf "$tmpdir/.venv" "$tmpdir/tests" "$tmpdir/deploy"

cd "$tmpdir"
docker build -t helios-space .
docker run --rm -p 7860:7860 \
  -e GROQ_API_KEY -e GEMINI_API_KEY -e SUPABASE_DATABASE_URL \
  -e JWT_SECRET_KEY \
  -e CORS_ALLOWED_ORIGINS='["https://helios-hridam.vercel.app"]' \
  helios-space
# In another shell:
curl -fsS http://localhost:7860/api/v1/health | jq
curl -fsS -X POST http://localhost:7860/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is helios?"}'
```

## Part C — BLOCKED (missing secret files)

Missing files, listed under §Blockers above. No `create_repo` was
called; no HF secret was set; nothing was pushed to
`huggingface.co/spaces/Hridam/helios`.

When the human drops the three missing files into `.secrets/`, the
Part C script can pick up where this cycle stopped. Sketch of what it
will do (already written into `backend/deploy/hf-space/README.md`):

1. `create_repo("Hridam/helios", repo_type="space", space_sdk="docker",
   exist_ok=True)` using `HF_TOKEN=$(cat .secrets/huggingface_token)`.
2. `HfApi().add_space_secret(...)` for `GROQ_API_KEY`,
   `GEMINI_API_KEY`, `JWT_SECRET_KEY` (freshly generated),
   `SUPABASE_DATABASE_URL`.
3. `HfApi().add_space_variable(...)` for `CORS_ALLOWED_ORIGINS`
   (the JSON list).
4. Build a temp export dir (see the shell in Part B above) and
   `git push -f hf main`.
5. `get_space_runtime` poll until RUNNING (max 30 min, three attempts
   with log fetch + branch push on failure).
6. Smoke test: `/api/v1/health`, one guest `/query`, `/ws/query`
   round-trip, CORS preflight with the Vercel origin.

If the DB ping in step 6 comes back false with an outbound-network
error, that would prove the port-5432 assumption wrong and flip 003's
recommendation. The Part C script should report BLOCKED in that case
per the prompt's design-blocker clause.

## Part D — BLOCKED (depends on Part C)

Not attempted. Once C is green, the sequence is unchanged from what
003's report §"Ordered implementation plan" wrote:

```bash
gh secret set VITE_API_URL --body https://hridam-helios.hf.space \
  --repo Hridambiswas/helios
gh workflow run deploy-frontend.yml --ref feat/hf-space-deploy \
  --repo Hridambiswas/helios
```

Then verify the live bundle:
```bash
curl -s https://helios-hridam.vercel.app/ \
  | grep -oE '/assets/index-[A-Za-z0-9._-]+' | sort -u
# fetch the new index-<hash>.js
curl -s https://helios-hridam.vercel.app/assets/<hash>.js \
  | grep -oE '[A-Za-z0-9.-]*helios[A-Za-z0-9.-]*' | sort -u
# expected: hridam-helios.hf.space  helios-hridam.vercel.app
# not expected: helios-hridam.ddns.net
```

The `tests/smoke/test_live.py` addition (Part D step 4) is also
deferred.

## PR description text for merging `feat/hf-space-deploy` into main

**Do not open the PR** — the director's call. This block goes in the
PR body when the human decides:

```markdown
### Summary

Prepares the backend for a Hugging Face Space deploy (option c from
003) and fixes the "Request failed" UX regression from 001.

Backend
- `CHROMA_PATH` env → embedded Chroma PersistentClient (no HTTP sidecar).
- `MINIO_ENABLED=false` → `/health` skips the MinIO ping and `/ingest`
  returns 503, so the backend can run in a Space with no S3-compatible
  storage.
- `OTEL_EXPORTER_OTLP_ENDPOINT=""` → tracer runs in no-op mode.
- `wait_for_db()` + `connection_info()` cherry-picked (path-adjusted)
  from `feature/supabase-integration` (Supabase pooler cold-start
  tolerance). Guarded by `cfg.supabase_database_url` so local test
  DBs are unaffected.
- `oauth_backend_url` default is now the Space URL.

Deploy
- `backend/deploy/hf-space/` — Dockerfile (UID 1000, HF-shape),
  `start.sh` (redis + uvicorn), README with Space YAML front matter,
  `.dockerignore`. DigitalOcean and Azure trees are untouched.

Frontend
- `humanReadableError()` in `api/client.ts` maps network / timeout /
  HTTP status to specific messages. Replaces the bare `'Request failed'`
  in ChatView + QueryInterface (see 001_report.md for the incident).
- Drops the hard-coded `ddns.net` fallback in AuthModal.

Not in this PR
- HF Space creation + push, Vercel `VITE_API_URL` rotation, live
  smoke tests — see 004_report.md's Parts C and D. Blocked on
  local `.secrets/` files the human is provisioning.

### Test plan
- [x] `cd backend && python -m pytest -q` — 167 passed, 8 pre-existing
      failures unrelated to this branch.
- [x] `cd frontend && npm ci && npm run build` — clean, no `ddns.net`
      in the new bundle.
- [ ] `docker build backend/deploy/hf-space` locally — deferred (no
      Docker on the workstation).
- [ ] Space smoke: `/api/v1/health`, `/api/v1/query`, `/ws/query`,
      CORS preflight from the Vercel origin — deferred until Part C.
- [ ] Vercel bundle contains `https://hridam-helios.hf.space` and no
      `ddns.net` — deferred until Part D.

Related: 001_report.md, 002_report.md, 003_report.md.
```

## Files not touched

- `backend/scripts/setup_ssl.sh` still defaults `DOMAIN=helios-hridam.ddns.net`.
  It is only for the certbot standalone flow on VM / EC2 deploys; not
  applicable to the HF Space (HF terminates TLS). Left as-is; will be
  removed / retargeted whenever the last VM path retires.
- `.env.production.example` — still references the pre-restructure
  shape. Not on the HF path; left alone.
- `backend/docker-compose.prod.yml` — nginx layer is still there for
  VM deploys. Not needed inside the HF Space (that's why the Space
  Dockerfile doesn't reference compose).

Nothing else known to be missing from Part A. When the human writes
the three secret files, cycle 005 should be able to pick up at Part C
step 1 without any further Part A work.
