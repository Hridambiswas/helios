# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.0] — 2026-09-24

### Added

- **VerifierAgent** — new sixth agent (`backend/agents/verifier.py`) that runs
  after the Critic and independently cross-checks the Synthesizer's answer.
  Uses `langchain-google-genai` + `gemini-2.5-flash`. Emits `verifier_scores`
  (groundedness, faithfulness, agreement, overall) and a `verifier_passed`
  boolean; both are surfaced in `QueryResponse`, `QueryHistoryItem`, the WS
  `done` event, and stored as a nullable JSON column on `QueryRecord`
  (Alembic migration `0006_verifier_scores`).
- **Attribution badges in the chat UI** — every answered query now shows
  `Ans by gpt-oss 120B · Cited by Hybrid Retriever · Judged by Critic <N%> ·
  Checked by Gemini <N%>` under the query header. Verifier score bars +
  reasoning appear in the eval tab.
- **Prometheus metrics** — `helios_verifier_score{dimension}` histogram and
  `helios_verifier_pass_total{result}` counter.
- **Config knobs** — `VERIFIER_ENABLED`, `VERIFIER_MIN_SCORE`,
  `VERIFIER_TIMEOUT_SECONDS`, `GEMINI_API_KEY`, `GEMINI_MODEL`.
- **DigitalOcean deploy tooling** — `backend/deploy/digitalocean/` scripts
  (`bootstrap.sh`, `first-deploy.sh`, `rotate-secrets.sh`, `verify.sh`) for a
  one-shot cut-over from EC2. Oracle Cloud variant kept in-tree at
  `backend/deploy/oracle/` as a fallback path.
- **DuckDNS integration** — `backend/deploy/duckdns/` (systemd timer +
  `install.sh`) re-asserts the A record every 5 min so the API URL is stable.

### Changed

- **Default LLM refresh** — `groq_model` default swapped from the retired
  `llama-3.3-70b-versatile` to `openai/gpt-oss-120b`. Planner, Synthesizer,
  and Critic all pick up the new model. `.env.example`,
  `.env.production.example`, `docker-compose.prod.yml`, and every README /
  PipelineSection / ChatView / Hero string were updated in lockstep.
- **Pipeline version** bumped to `1.2.0`. Startup log now reports
  `verifier=on/off` and the active `groq_model`.
- **Terminal state** — the LangGraph now ends at the Verifier node instead of
  the Critic. Retry loop still lives on Critic → Synthesizer.
- **Backend host: EC2 → DigitalOcean droplet.** Reserved (Floating) IP replaces
  AWS's dynamic IP, so the API endpoint no longer disappears when the instance
  restarts. Ubuntu 22.04 x86_64, region BLR1.
- **DNS: No-IP `.ddns.net` → DuckDNS `.duckdns.org`.** No-IP's free tier
  expires monthly; DuckDNS does not.
- **CI: `deploy-backend.yml`** rewritten to target DO host / user / key
  secrets (`DO_HOST`, `DO_USER`, `DO_SSH_KEY`) with a post-deploy nginx
  smoke test.

### Docs

- README architecture diagram: added Verifier node under Critic; refreshed
  all Llama 3.3 boxes to `gpt-oss 120B`; new v1.2 "What's new" table;
  Gemini verifier badge added. Full migration playbook at
  `docs/migration/oracle-arm.md`.

---

## [1.1.0] — 2026-05-12

### Added

- **Multi-turn conversation memory** — `QueryRequest` now accepts a `history` list
  (up to 20 `{role, content}` items). Planner and Synthesizer use the prior turns
  for context so follow-up questions resolve correctly without repeating background.

- **Per-token streaming** — `SynthesizerAgent` uses `llm.stream()` when a
  `_token_callback` is present in `HeliosState`. Tokens are forwarded through an
  `asyncio.Queue` to the WebSocket event loop and emitted as `token` events.
  The frontend accumulates tokens and renders a blinking cursor during streaming.

- **Server-side conversation persistence** — new `Conversation` and
  `ConversationMessage` ORM models (Alembic migration `0005_conversations`).
  Five REST endpoints (`GET/POST /conversations`, `GET/POST/DELETE
  /conversations/{id}`) allow the frontend to sync chat history across devices.
  `useConversations` lazy-loads messages and persists each turn after login.

- **Critic retry loop** — when the Critic scores an answer below the passing
  threshold the pipeline routes back to the Synthesizer once (`_MAX_RETRIES = 1`)
  with the critic's improvement suggestions injected as explicit fix instructions.
  A `retrying` WebSocket event and "Improving answer…" UI step notify the user.

- **Guest query limit raised from 1 to 5** — `config.py` `guest_query_limit` is
  now `5`, letting guests evaluate the platform before signing in.

- **Mobile layout** — `MobileBottomNav` fixed bottom bar (`sm:hidden`) with Home /
  Chat / Upload / Sign-in tabs. Safe-area input bar (`env(safe-area-inset-bottom)`).
  Auto-collapsing sidebar on `xs`. Full-width user bubbles. Pipeline step labels
  hidden on narrow screens. `viewport-fit=cover` for notched displays.

- **Document chunk feedback** — Upload panel now shows expandable chunk previews
  per document and a test-retrieval search box. Two new endpoints:
  `GET /documents/{id}/chunks` and `POST /documents/{id}/search`.

### Changed

- `sendWSQuery` helper now accepts an optional `history` parameter and includes
  it in the WebSocket message payload.
- `queries.run()` client method now accepts a `history` parameter.
- WebSocket handler parses `history` from each incoming message and validates
  role/content before forwarding to the pipeline.

### Tests added

- `tests/test_agents.py` — `TestConversationMemory`, `TestSynthesizerStreaming`
- `tests/test_pipeline.py` — `TestRetryLoop`, `TestConversationHistory`
- `tests/test_schemas.py` — `TestHistoryMessage`, `TestConversationSchemas`
- `tests/test_api.py` — `TestConversationRoutes`, `TestDocumentChunkRoutes`
- `tests/test_ws_streaming.py` — token events, synthesizing step, history forwarding

---

## [1.0.0] — 2026-04-26

Initial production release.

### Core features

- Five-agent LangGraph pipeline: Planner → Retriever → Executor → Synthesizer → Critic
- Hybrid retrieval: dense (BAAI/bge-large-en-v1.5) + CLIP + BM25, RRF fusion
- Sandboxed Python execution with AST allowlist
- LLM-as-judge critic scoring (groundedness, faithfulness, completeness)
- Celery async workers with Redis broker; task status polling
- JWT auth with refresh-token rotation; GitHub OAuth
- WebSocket streaming with per-pipeline-step events
- Document ingestion: `.txt .md .pdf .csv .json .rst`, 50 MB max, chunked + indexed
- Rate limiting (60 req/60 s) and brute-force protection
- OpenTelemetry + Prometheus observability
- EC2 (backend) + Vercel (frontend) + Supabase PostgreSQL deployment
