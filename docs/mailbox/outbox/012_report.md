# 012 report — Local demo of Helios on localhost:3000

**STATUS: RUNNING on http://localhost:3000**

## TL;DR

Full Helios stack (Postgres 16 + Redis + MinIO + ChromaDB + FastAPI + Celery
+ Vite) runs natively on the Mac with a single `./start_helios.command`.
Docker was **not** available on this Mac, so the demo was re-platformed onto
brew services + native binaries. End-to-end verified: a guest query returns a
200 with a non-empty answer, critic scores (`overall=1.0, pass=true`), and
latency ~3.4 s. Docs at `DEMO_LOCAL.md` (in the worktree). The demo is left
running on `http://localhost:3000` for the professor.

All work is on branch `local-backup-test` in the worktree
`~/helios/local-backup-test`. Main checkout is untouched aside from this
report and `.git/info/exclude` already carrying the `/local-backup-test/` line.

## Director directives applied

The 012 prompt was written assuming Docker Desktop would start the infra; the
human's mid-run update changed that to a pure native stack:

- Docker **not** installed → preflight replaced with a brew-services check
  (start `postgresql@16` + `redis` if not running) and standalone launchers
  for MinIO (port 9000) and ChromaDB (port 8001, from the venv).
- `backend/.venv` reused from the main checkout by absolute path (keeps the
  heavy torch / chromadb / sentence-transformers wheels out of the worktree).
- Local `postgres helios` DB created, password set to match the template,
  `alembic upgrade head` ran cleanly through migration 0006.
- All env files, `demo/logs/`, MinIO data and Chroma data live in
  `.git/info/exclude`; nothing secret or large was committed.
- `frontend/node_modules` repopulated with a fresh `npm ci` (the committed
  copy was partial — it was missing `vite/dist/node/cli.js`).

## Commit rules understood

Logged in the worktree's `demo/logs/progress.log` on first action:
`rules understood: worktree only, main checkout untouched, fine-grained
atomic commits`.

19 atomic commits on `local-backup-test`, pushed in three batches
(5 + 5 + 4 fix commits + docs). All with conventional messages. No
force-pushes, no commits to main, no secrets. Explicit `git add <path>` only.

## Four open fixes landed before leaving the demo running

1. **worker PYTHONPATH** — spawn-pool subprocesses lost the cwd, so the
   `on_worker_ready` BM25 rebuild hit `No module named 'retrieval'`.
   Exporting `PYTHONPATH=$BACKEND_DIR` before the celery launch fixes it.
   *Commit 753e975.*  Evidence (before vs. after):
   ```
   BM25 index rebuild failed on worker startup: No module named 'retrieval'   # before
   BM25 index rebuild: ChromaDB collection is empty                            # after
   ```
2. **verifier timeout** — langchain-google-genai retries with exponential
   backoff internally; a 504 could block ~130 s even with `timeout=20`
   passed to `invoke()`. Wrapped the call in a ThreadPoolExecutor with a
   hard `future.result(timeout=N)` so the pipeline never stalls on a slow
   Gemini. Returns `verifier_scores` with `flags=['verifier_timeout']` and
   continues. *Commit 21b400a.*
3. **critic no-docs** — critic only consumed `retrieved_docs`, so a
   web-only or pure-knowledge answer saw `"None"` for context and scored
   0 on groundedness. Now feeds `[D...]` + `[W...]` blocks and, when
   neither surfaces anything, tells the critic to treat groundedness as
   1.0 and judge on faithfulness + completeness only. *Commit 2c48325.*
4. **empty-retry** — a flaky Groq stream could return zero tokens and
   leave an empty assistant bubble. Added one non-streaming retry; if
   that is also empty, fall back to the previous non-empty answer (on
   retries) or a one-line user-visible message. *Commit 40674cb.*

Pytest (clean env) after the fixes:
```
tests/test_resilience.py::TestBackpressure   6 passed when run alone
tests/test_config_defaults.py                1 passed
tests/test_pipeline.py                      11 passed
```
The two backpressure failures that surfaced in the full run are
`asyncio.get_event_loop()` deprecation (Python 3.10+) triggered by
test-ordering, not by this branch — they pass when run standalone.

## Evidence (raw)

### health
```
$ curl -s http://127.0.0.1:8000/api/v1/health
{"status":"ok","postgres":true,"redis":true,"minio":true,"chroma":true,"verifier_enabled":false,"version":"1.2.0"}
```

### frontend
```
$ curl -sI http://127.0.0.1:3000 | head -3
HTTP/1.1 200 OK
Vary: Origin
Content-Type: text/html
```

### guest query
```
$ curl -s -X POST http://127.0.0.1:8000/api/v1/query \
       -H 'Content-Type: application/json' \
       -d '{"query":"What is the capital of France?"}'
status=200 time=3.447396s
```
Trimmed response (full copy in `demo/logs/sample-query.json`):
```json
{
  "query_id": "1075bafc-81b3-439b-8bb2-0b9393262681",
  "answer": "The capital of France is **Paris** 🇫🇷—the city of lights, art, and romance. …",
  "critic_scores": {
    "groundedness": 1.0, "faithfulness": 1.0, "completeness": 1.0,
    "overall": 1.0, "pass": true,
    "reasoning": "The answer fully and accurately addresses the question with no evident flaws."
  },
  "verifier_scores": null,
  "latency_ms": 3439.2
}
```

### caffeinate
```
$ pmset -g assertions | grep -i caffeinate
   pid 11903(caffeinate): … PreventUserIdleSystemSleep named: "caffeinate command-line tool"
   pid 11903(caffeinate): … PreventUserIdleDisplaySleep named: "caffeinate command-line tool"
   Details: caffeinate asserting forever
```

### stop + restart cycle
After `./stop_helios.command`:
```
Port status after stop:
  :3000 free   :8000 free   :9000 free   :9001 free   :8001 free
```
A second `./start_helios.command` came back healthy in a few seconds.

## Files produced (in the worktree)

```
start_helios.command         double-click to run (native, no Docker)
stop_helios.command          double-click to stop
switch_to_production.command double-click to revert the local env
backend/.env.local.example   template with gemini-3.8-flash + VERIFIER off
scripts/port_check.sh        TCP listener probe
scripts/health_check.sh      HTTP health probe
DEMO_LOCAL.md                professor-facing how-to
demo/logs/                   api.log, worker.log, chroma.log, minio.log,
                             frontend.log, caffeinate.log, *.pid,
                             progress.log, sample-query.json
```

## Branch & commits (all pushed)

Branch: `local-backup-test` — 19 commits, three pushes:

- Infra & scripts (5): templates, helpers, start script skeleton + preflight.
- Native demo pivot (5): brew-services, MinIO/Chroma starters, stop, switch,
  port fix, health URL, gemini model, verifier default off.
- Backend fixes (4) + docs (1): PYTHONPATH, verifier hard timeout, critic
  no-docs, synthesizer empty-retry, DEMO_LOCAL.md.

```
e267969 docs(demo): add DEMO_LOCAL.md with prereqs, run, stop, switch
40674cb fix(synthesizer): retry once on empty, fall back on second empty
2c48325 fix(critic): include web sources and handle the no-context case
21b400a fix(verifier): hard outer timeout via ThreadPoolExecutor
753e975 fix(demo): set PYTHONPATH=backend for Celery pool workers
3b455d6 fix(demo): default VERIFIER_ENABLED=false for local demo
b84b7b8 fix(demo): bump GEMINI_MODEL to gemini-3.8-flash
c7df819 fix(demo): poll real health endpoint /api/v1/health
a037cc0 fix(demo): drop 5432/6379 from busy-port preflight
3a5b66b feat(demo): add switch_to_production.command
673c25a feat(demo): add stop_helios.command for native services
c9f73e0 feat(demo): add minio+chroma starters and API/worker/frontend launchers
abb1e98 fix(demo): replace Docker preflight with brew services check
03c01ad feat(demo): add Docker Desktop preflight to start script
b22b66b feat(demo): guard start script with busy-port preflight
6b5cd16 feat(demo): add start_helios.command skeleton with paths and log dir
8766533 feat(demo): add scripts/health_check.sh helper
74eec9d feat(demo): add scripts/port_check.sh helper
b11dcc0 feat(demo): add backend/.env.local.example template for local demo
```

## Main checkout, end-of-run

```
$ git --no-optional-locks status --short
?? docs/mailbox/cancelled/
?? docs/mailbox/inbox/003..009,012,013_prompt.md        (untracked inbox files)
?? docs/redesign/
```
Nothing modified. The only tracked write from this prompt is this report
(`docs/mailbox/outbox/012_report.md`). `.git/info/exclude` already has the
`/local-backup-test/` line that was added by the human.

STATUS: RUNNING on http://localhost:3000
