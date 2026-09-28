---
title: Helios
emoji: 🧠
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Multi-agent GenAI platform — retrieval + code exec + verification.
---

# Helios — Hugging Face Space

This directory is the source of the Space at `Hridam/helios`. The
`Dockerfile`, `start.sh`, and this README are pushed to the Space repo
root; the backend Python code (`backend/*`) is copied to the same root
before push.

## Architecture

Single-container deploy per HF's one-port-per-Space rule. Inside the
container:

- **uvicorn** on `0.0.0.0:7860` (FastAPI + Helios pipeline)
- **redis-server** on `127.0.0.1:6379` (in-memory, no persistence — HF
  blocks outbound port 6379 so external Redis is unreachable)
- **ChromaDB** as an embedded `PersistentClient` at `/tmp/chroma`
  (ephemeral — Space rootfs is wiped on every restart)
- **Supabase Postgres** over port 5432 (**IS** on the HF outbound
  allow-list — evidence in the 003 report). Users, conversations and
  query history persist here.

Not running inside the Space:
- **MinIO** — no S3-compatible backend reachable. `MINIO_ENABLED=false`
  makes `/ingest` return 503 and skips the startup bucket check.
- **Celery worker / beat** — sync `/query` is the only path used. The
  async `/query/async` endpoint is unavailable in this deploy.
- **OTel collector, Prometheus** — no collector reachable;
  `OTEL_EXPORTER_OTLP_ENDPOINT=""` puts the tracer in no-op mode.

## Space configuration (set by the human)

### Secrets (Settings → Secrets)

- `GROQ_API_KEY` — Groq console key (Llama / gpt-oss models)
- `GEMINI_API_KEY` — Google AI Studio key (verifier)
- `JWT_SECRET_KEY` — 64-hex-char random string
  (`python -c 'import secrets;print(secrets.token_hex(32))'`)
- `SUPABASE_DATABASE_URL` — session-mode string, format
  `postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`

### Variables (Settings → Variables)

`CORS_ALLOWED_ORIGINS` is the only non-secret runtime value; the rest
are baked into the Dockerfile ENV. Set:

```
CORS_ALLOWED_ORIGINS=["https://helios-hridam.vercel.app","https://frontend-omega-blush-87.vercel.app"]
```

## Publishing this Space

Done by the human once, then automated by the Part C script in 004:

```bash
# 1. Log in to HF (token from .secrets/huggingface_token or hf auth login)
export HF_TOKEN="$(cat .secrets/huggingface_token)"

# 2. Create the Space repo
python -c "from huggingface_hub import create_repo; \
  create_repo('Hridam/helios', repo_type='space', space_sdk='docker', \
              exist_ok=True, token='$HF_TOKEN')"

# 3. Build the Space tree in a temp dir
tmpdir=$(mktemp -d)
cp backend/deploy/hf-space/{Dockerfile,start.sh,README.md} "$tmpdir/"
cp -r backend/* "$tmpdir/"
# Strip stuff we do not want in the Space (already in .dockerignore too):
rm -rf "$tmpdir/.venv" "$tmpdir/tests" "$tmpdir/deploy"

# 4. Push
cd "$tmpdir" && git init -b main && git add . \
  && git commit -m "helios space bootstrap" \
  && git remote add hf "https://user:$HF_TOKEN@huggingface.co/spaces/Hridam/helios" \
  && git push -f hf main
```

The Space then polls: `huggingface_hub.get_space_runtime("Hridam/helios")`
until status is `RUNNING`. Cold build on cpu-basic is expected to take
5–8 minutes (fastembed + torch wheels are the slow bits).

## Verifying

```bash
curl -fsS https://hridam-helios.hf.space/api/v1/health | jq
```

Expected:
```json
{"status": "ok", "postgres": true, "redis": true, "minio": true, "chroma": true, "verifier_enabled": true}
```

`"minio": true` is a sentinel — with `MINIO_ENABLED=false` the ping is
skipped and the field is reported as true so overall status stays "ok".
See `backend/api/routes.py::health()`.
