# Helios, instructions for Claude Code

## Project
Helios is a distributed multi agent GenAI platform. Six agents: planner, multimodal
retriever (CLIP + BM25 + ChromaDB), sandboxed Python executor, synthesizer, LLM as judge
critic (Groq), verifier (Gemini). FastAPI backend (`backend/main.py`, routes in
`backend/api/`), Celery workers (`backend/workers/`), PostgreSQL, Redis, MinIO, ChromaDB,
OpenTelemetry, WebSocket streaming (`backend/api/websocket.py`). Frontend is Vite + React
in `frontend/`, deployed to Vercel at https://helios-hridam.vercel.app. The API base URL is
baked into the bundle at build time from `VITE_API_URL` (GitHub secret, see
`.github/workflows/deploy-frontend.yml`). Backend deploy tooling: `backend/deploy/`
(digitalocean, oracle, duckdns, k8s, render).

## Mailbox protocol (you are driven by a director)
- Prompts arrive as `docs/mailbox/inbox/NNN_prompt.md`. Reports go to
  `docs/mailbox/outbox/NNN_report.md`.
- Each cycle: if `docs/mailbox/STOP` exists, stop. Otherwise execute the lowest NNN prompt
  that has no matching report. Write the full report (commands run, raw evidence, findings,
  what changed, what is left), commit, push.
- Never edit, rename or delete anything in `docs/mailbox/inbox/`.
- If a prompt is blocked on something only the human can do (credentials, DNS panel,
  cloud console, billing), say so clearly at the top of the report under `BLOCKED:` and
  stop that prompt. Do not invent workarounds that touch infrastructure.

## Autonomy
Every mailbox prompt is pre approved. Do not wait for confirmation inside a prompt.
If you run out of time or context, commit partial work with a report marked PARTIAL,
push, and resume from it next cycle.

## Branch rules (hard)
- Never commit to or push to `main`. Never force push.
- Working branch for this debugging effort: `fix/request-failed` (branched from
  `feature/do-deploy-with-verifier`). Mailbox commits go here.
- If a prompt asks for a separate fix branch, create it from `fix/request-failed`,
  push it, and name it in the report. Merging to main happens only via a PR the human
  approves.
- Always use `git --no-optional-locks` for read commands (status, log, diff). Plain
  `git status` leaves an index.lock that cannot be removed in this setup.

## Engineering rules
- Understand before changing: read the relevant architecture and code path end to end
  before editing. No trial and error edits.
- Reproduce before fixing. Every fix needs evidence that the bug existed and is gone.
- Never commit secrets, `.env` files, tokens or keys. Never print API keys in reports
  (show only presence and length, or the last 4 characters).
- One commit per logical change, conventional commit messages.
- Run the relevant tests (`cd backend && pytest -q`, `cd frontend && npm run build`)
  before pushing code changes.
