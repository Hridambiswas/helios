# Helios — local demo on your Mac

One-click Helios running at **http://localhost:3000**, with a FastAPI backend at
**http://localhost:8000**, all on this Mac. No Docker required. No cloud
dependencies for the demo path.

## Prerequisites (one-time)

1. Homebrew (https://brew.sh).
2. The repo at `~/helios`, with the Python venv already built at
   `~/helios/backend/.venv` (this demo reuses it).
3. Keys in `~/helios/.secrets/`:
   - `groq_api_key`  (for the planner / synthesizer / critic)
   - `gemini_api_key` (for the optional verifier)
4. `brew install postgresql@16 redis minio` and
   `brew services start postgresql@16 redis`.
5. Create the demo DB (once):
   ```sh
   /opt/homebrew/opt/postgresql@16/bin/psql -d postgres -c \
     "CREATE USER helios WITH PASSWORD 'helios_local_dev' CREATEDB;"
   /opt/homebrew/opt/postgresql@16/bin/psql -d postgres -c \
     "CREATE DATABASE helios OWNER helios;"
   ```
6. Copy `backend/.env.local.example` → `backend/.env.local`, paste in the two
   keys, and generate a random `JWT_SECRET_KEY`:
   ```sh
   python -c 'import secrets; print(secrets.token_hex(32))'
   ```
   `start_helios.command` will auto-copy `.env.local` → `.env` on first run.

## Running the demo

Double-click **`start_helios.command`** in Finder (or `./start_helios.command`
from the terminal). The script:

1. Checks that the ports it opens (3000, 8000, 9000, 9001, 8001) are free.
2. Starts brew services (Postgres 16 + Redis) if they are not already running.
3. Launches MinIO on `:9000` (data in `demo/minio-data/`).
4. Launches ChromaDB on `:8001` (data in `demo/chroma-data/`).
5. Copies `backend/.env.local` → `backend/.env` on first run (marked with a
   `# created-by:` header so `switch_to_production.command` can tell it apart).
6. Starts `caffeinate -dims` so the Mac does not sleep mid-demo.
7. Starts the FastAPI API (uvicorn on `:8000`) and the Celery worker,
   with `PYTHONPATH=backend` so the spawn-pool workers can import local modules.
8. Polls `http://localhost:8000/api/v1/health` for up to 120 s.
9. Starts the Vite dev server on `:3000` (`npm ci` first if there is no
   `node_modules/`).
10. Opens `http://localhost:3000` in your default browser.

Every process writes a log to `demo/logs/<name>.log` and its PID to
`demo/logs/<name>.pid`.

## What the professor will see

- The Helios chat UI at `http://localhost:3000` (Vite dev server with HMR).
- Guest queries like "What is the capital of France?" return a non-empty
  answer, critic scores (groundedness / faithfulness / completeness /
  overall), and follow-up questions, all in ~3 s against Groq's hosted
  Llama / gpt-oss models.
- The API docs at `http://localhost:8000/docs`.
- A `/health` JSON that shows Postgres, Redis, MinIO and Chroma all
  reporting `true` — proof the whole stack is live on this Mac.

## Stopping

Double-click **`stop_helios.command`**. The script kills everything by PID
file (api, worker, frontend, caffeinate, minio, chroma), confirms the demo
ports are free, and leaves the brew services (postgresql@16, redis) running —
those are long-lived and safe to share with other work.

To stop them too:
```sh
brew services stop postgresql@16 redis
```

## Switching back to the hosted setup

Double-click **`switch_to_production.command`**. It removes the local
`backend/.env` only if it was auto-created by the demo (identified by the
`# created-by: start_helios.command` marker on the first line); a hand-written
`backend/.env` is never clobbered.

For "full capacity" (the hosted Vercel frontend + DuckDNS backend):
- Frontend: already at https://helios-hridam.vercel.app
- Backend URL is baked into the frontend bundle at build time via
  `VITE_API_URL` (see `.github/workflows/deploy-frontend.yml` secrets).
- Nothing on this Mac has to change — this script just tidies the local env.

## Verifier (Gemini) is off by default

The demo keeps `VERIFIER_ENABLED=false` in the template because Gemini free
tier returns 503/504 for `gemini-3.8-flash` under load, which would stall the
primary path for up to ~130 s on retries. The critic (Groq) already provides
groundedness / faithfulness / completeness scores. To enable verification, set
`VERIFIER_ENABLED=true` in `backend/.env.local` and pick a stable Gemini model.
The verifier now has a hard outer timeout, so even if Gemini is slow the
pipeline continues and the answer is marked `unverified` instead of failing.

## Files

```
start_helios.command         double-click to run
stop_helios.command          double-click to stop
switch_to_production.command double-click to restore hosted mode
scripts/port_check.sh        helper: is a TCP port free?
scripts/health_check.sh      helper: poll http(s) health
demo/logs/                   per-process logs + PID files + progress.log
demo/minio-data/             MinIO on-disk state
demo/chroma-data/            ChromaDB on-disk state
backend/.env.local.example   fill in your keys, copy to backend/.env.local
```
