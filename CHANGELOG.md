# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
