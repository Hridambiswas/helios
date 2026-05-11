# Changelog

All notable changes to Helios are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-05-11

### Added
- **Web search** — DuckDuckGo integration answers any query without needing uploaded documents
- **Source citations** — Answers cite web sources `[W1]`, `[W2]` with clickable URLs
- **Follow-up questions** — 2 contextual follow-ups suggested after every answer
- **Supabase integration** — Managed PostgreSQL backend for user data (zero-ops)
- **Markdown rendering** — Frontend renders bold, bullets, code blocks, links properly

### Fixed
- **Registration crash** — Replaced `passlib 1.7.4` with direct `bcrypt` calls (incompatible with bcrypt >= 4.0)
- **OOM kills on query** — Removed CLIP model (600 MB) from retrieval pipeline entirely
- **Startup crash** — Removed `retrieval/clip_encoder.py` import that referenced removed `torch`
- **Disk full on deploy** — Added `docker system prune -a -f` before each Docker build
- **fastembed runtime download** — Pre-bake model into Docker image at build time

### Changed
- Embeddings: `sentence-transformers` + PyTorch → **FastEmbed ONNX** (120 MB vs 1.5 GB)
- Agents now initialize lazily on first query instead of at startup
- API container memory limit raised from 300 MB to 512 MB

---

## [1.0.0] — 2026-04-15

### Added
- Full LangGraph multi-agent pipeline (Planner → Retriever → Executor → Synthesizer → Critic)
- FastAPI REST API + WebSocket streaming
- JWT authentication with refresh token rotation
- Document ingestion (PDF/text) with ChromaDB vector indexing + BM25
- Celery async task queue
- Prometheus metrics + OpenTelemetry tracing
- Docker Compose production deployment
- GitHub Actions CI/CD for EC2 + Vercel
- Nginx reverse proxy with Let's Encrypt SSL
- Resilience patterns: backpressure, circuit breaker, rate limiting, bulkhead

---

[1.1.0]: https://github.com/Hridambiswas/helios/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Hridambiswas/helios/releases/tag/v1.0.0
