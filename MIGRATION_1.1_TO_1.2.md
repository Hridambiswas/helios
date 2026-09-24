# Migrating from Helios 1.1.0 → 1.2.0

Two independent changes ship together. Do both.

## 1. Retired model — `llama-3.3-70b-versatile`

Groq has retired the previous default. If you were pinning the old id in
your `.env`, replace it:

```
- GROQ_MODEL=llama-3.3-70b-versatile
+ GROQ_MODEL=openai/gpt-oss-120b
```

No code changes are required — Planner, Synthesizer, and Critic all read
`cfg.groq_model`.

## 2. New Verifier stage — Google Gemini

The pipeline now runs a sixth agent (`agents/verifier.py`) after the Critic.
It re-reads the answer with `gemini-2.5-flash` and emits a `verifier_scores`
dict plus a `verifier_passed` boolean.

### Required env vars

```
GEMINI_API_KEY=...          # get one at https://aistudio.google.com/app/apikey
GEMINI_MODEL=gemini-2.5-flash
VERIFIER_ENABLED=true       # set to false to disable the stage
VERIFIER_MIN_SCORE=0.5
VERIFIER_TIMEOUT_SECONDS=20
```

### Database

Run Alembic migration `0006_verifier_scores` — it adds a nullable JSON
column `verifier_scores` to `query_records`:

```
alembic upgrade head
```

### API shape

`QueryResponse`, `QueryHistoryItem`, and the WebSocket `done` event now
include `verifier_scores` and `verifier_passed`. Both are `null` when the
Verifier is disabled. Existing clients keep working — the new fields are
additive.

### UI

The chat surfaces `Ans by · Cited by · Judged by · Checked by` badges
under each answered query. The eval tab shows Gemini's dimension bars
plus its reasoning line.
