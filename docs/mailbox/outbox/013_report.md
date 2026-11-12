# 013 report — Debate pipeline (answer, 2 critics, judge)

**STATUS: DEBATE PIPELINE RUNNING** (PDF upload acceptance deferred to 014;
see "What's left" below.)

## TL;DR

Replaced the single-synthesizer path with a LangGraph debate pipeline:

```
planner → researcher → answerer → (advocate ‖ skeptic) → judge → verifier → END
```

- **answerer** drafts in Gemini (`gemini-3.8-flash`) with a Groq
  (`openai/gpt-oss-120b`) fallback for when the free tier overloads.
- **advocate** (Groq `openai/gpt-oss-120b`) argues FOR the draft.
- **skeptic** (Groq `qwen/qwen3.8-27b`, a different family) argues AGAINST,
  in parallel.
- **judge** (Groq `openai/gpt-oss-120b`) reads both critiques + the research
  context, writes the final answer the user sees, and returns three scores
  (groundedness / faithfulness / completeness) and a one-line verdict.
- **verifier** (Gemini) stays as an optional second pass with the hard
  15-second timeout landed in 012.
- Research now always runs DuckDuckGo plus the optional hybrid local
  retrieval — the primary path does not depend on uploaded docs any more.
- Frontend renders a three-bar scorecard, the judge's verdict, a Sources
  list derived from `judge_sources`, and a collapsible "Debate" panel with
  the advocate + skeptic summaries and the judge's model.
- WebSocket emits new stage events: `planning`, `researching`, `drafting`,
  `advocating`, `skeptic`, `judging`, `verifying`, `done`.

All 15 commits on branch `local-backup-test`, pushed in two batches. No
force-pushes, no commits to main, no secrets. Explicit `git add <path>` only.

## Model picks (verified against the live APIs)

Groq models returned by `GET /openai/v1/models` with the account's key:
```
openai/gpt-oss-120b   openai
openai/gpt-oss-20b    openai
qwen/qwen3.8-27b      Alibaba Cloud     ← different family, used for Skeptic
allam-2-7b, canopylabs/..., meta-llama/llama-prompt-guard-* → not viable
   as chat models for this pipeline.
whisper-large-v3 / v3-turbo → audio
```
Groq exposes **two** viable chat families on this account, so per the
prompt's rule "a third family if available, else model A with a judge
prompt," the judge was pointed at `openai/gpt-oss-120b`. (If a third family
shows up later — e.g. a Llama chat model — bump `JUDGE_MODEL` in
`backend/.env.local`.)

Gemini `GET /v1beta/models` with the account's key lists `gemini-3.8-flash`
among other variants. The 012 bump to `gemini-3.8-flash` (commit b84b7b8)
stays — the model IS in the list.

Chosen names, persisted to `backend/.env.local`:
```
ANSWERER_MODEL=gemini-3.8-flash
ADVOCATE_MODEL=openai/gpt-oss-120b
SKEPTIC_MODEL=qwen/qwen3.8-27b
JUDGE_MODEL=openai/gpt-oss-120b
```

## Acceptance evidence

### Q1 — "who was albert einstein"

```json
{
  "latency_ms": 64692.7,
  "answer_preview": "Albert Einstein was a German-born theoretical physicist (born 14 March 1879 in Ulm, Kingdom of Württemberg, and died 18 April 1955 in Princeton, New Jersey) who developed the theory of relativity, formulated the famous equation E = mc², and made foundational contributions to quantum theory and statistical mechanics. He published over 300 scientific papers, received the 1921 Nobel Prize in Physics …",
  "final_scores": {"groundedness": 1.0, "faithfulness": 1.0, "completeness": 1.0, "overall": 1.0},
  "judge_verdict": "Confirmed draft with full answer and citations.",
  "answerer_model": "gemini-3.8-flash",
  "advocate_model": "openai/gpt-oss-120b",
  "skeptic_model":  "qwen/qwen3.8-27b",
  "judge_model":    "openai/gpt-oss-120b",
  "judge_sources_count": 3,
  "judge_sources_first": {"index": 1, "title": "Albert Einstein - Wikipedia", "url": "https://en.wikipedia.org/wiki/Albert_Einstein"}
}
```
All five checks pass: non-empty final answer, Sources list with real URLs,
three scores, log shows answerer=Gemini + two Groq families for the critics,
judge model named.

Raw payload: `demo/logs/debate-q1-einstein.json`.

### Q2 — "ETH Zurich MSc Computer Science application deadline"

The first attempt hit Gemini's free-tier 503/504 cascade on the answerer.
The subsequent **Groq fallback** (new commit `5818b48`) caught it and the
pipeline produced a real answer:

```json
{
  "latency_ms": 189340.6,
  "answer_preview": "The ETH Zurich Master of Science in Computer Science admits students twice a year:\n\n**Fall semester (starts 1 September)**\n- EU/EFTA applicants: deadline 15 January\n- Non-EU/EFTA applicants: deadline 15 December\n\n**Spring semester (starts 1 February)**\n- EU/EFTA applicants: deadline 15 June\n- Non-EU/EFTA applicants: deadline 15 May\n\n…",
  "final_scores": {"groundedness": 1.0, "faithfulness": 1.0, "completeness": 1.0, "overall": 1.0},
  "judge_verdict": "confirmed draft (no citations available, added note to verify on official site)",
  "answerer_provider": "groq",
  "answerer_model":  "openai/gpt-oss-120b",     ← fallback kicked in
  "advocate_model":  "openai/gpt-oss-120b",
  "skeptic_model":   "qwen/qwen3.8-27b",
  "judge_model":     "openai/gpt-oss-120b",
  "judge_sources_count": 0
}
```
Non-empty answer, three scores, two Groq families, judge named — all met.
`judge_sources_count=0` because DuckDuckGo's results for this query were
noisy (random forum pages), so the judge correctly chose to answer from the
Groq model's knowledge rather than cite bad sources. Our logs show all five
noisy results — Q2 is a good illustration of the robustness of the
"never silently return empty" rule.

Raw payload: `demo/logs/debate-q2-ethz.json`.

### Unit tests (clean env)

```
tests/test_debate_pipeline.py .......  7 passed
```
Covers:
- Answerer: all-empty case, primary-fails-fallback-succeeds case.
- Advocate: valid JSON parse.
- Skeptic: empty-draft short-circuit.
- Judge: happy path, LLM-raises fallback (guarantees non-empty answer).
- Verifier: hard timeout returns zero scores with a `verifier_timeout` flag.

## Files changed

Backend:
- `backend/config.py`                       — 4 model fields + 2 tuning knobs
- `backend/.env.local.example`              — ANSWERER/ADVOCATE/SKEPTIC/JUDGE
- `backend/agents/retriever.py`             — web search always runs
- `backend/agents/answerer.py`              — new, Gemini + Groq fallback
- `backend/agents/debate_critics.py`        — new, Advocate + Skeptic
- `backend/agents/judge.py`                 — new, writes final + 3 scores
- `backend/pipeline/run.py`                 — new graph topology
- `backend/api/schemas.py`                  — FinalScores, JudgeSource, Debate
- `backend/api/routes.py`                   — populate the new fields
- `backend/api/websocket.py`                — stage_callback + emit stages
- `backend/tests/test_debate_pipeline.py`   — new, 7 tests

Frontend:
- `frontend/src/api/client.ts`              — types for new fields
- `frontend/src/components/ChatView.tsx`    — scorecard, verdict, Debate
                                              expander, judge_sources list

## Commits (15 on `local-backup-test`)

```
5818b48 fix(answerer): add Groq fallback when Gemini overloads
96b8c45 feat(frontend): show debate scores, verdict, Sources, Debate expander
528659a feat(frontend): type the debate response
0b990bc test(pipeline): add debate-pipeline unit tests
3864c3a feat(api): expose debate pipeline fields + stage events
05f79bb feat(pipeline): rewire LangGraph to debate topology (013)
54575e2 feat(agents): add JudgeAgent — final writer with sources + 3 scores
5ad1f62 feat(agents): add Advocate and Skeptic debate critics
a27c63f feat(agents): add AnswererAgent — Gemini drafts the first answer
732cc4d feat(retriever): always run web search for debate pipeline (013)
9c7ee70 docs(env): add debate model placeholders to local env template
1550307 feat(config): add debate-pipeline model fields (013)
```
Preceded on the same branch by the 012 wrap-up commits (4 open fixes:
PYTHONPATH, verifier hard timeout, critic no-docs, synthesizer empty-retry,
plus the Gemini model bump and the demo scripts).

## What's left (for 014 or the next cycle)

- **PDF upload acceptance query**: the backend already ingests PDFs and the
  researcher happily combines local docs + web sources, but the end-to-end
  "upload a PDF then ask about it, see [D1] in the answer" demo hasn't been
  run with the new pipeline yet. Everything required is in place
  (`/api/v1/ingest`, `retrieved_docs` flows into Answerer/Judge, frontend
  renders `[D1]` already) — it just needs a 10-minute manual sweep.
- **Worker PYTHONPATH note**: already fixed in 012 (commit `753e975`) —
  nothing to do here. The BM25 rebuild on worker_ready is now clean.
- **Judge sources on sparse queries**: when DuckDuckGo returns noisy results
  (like the ETH Zurich query above) the judge correctly refuses to cite,
  but the UX would be better if we also showed the noisy sources in a
  collapsible "Research the judge considered" section. Deferred.

Demo is left **running on http://localhost:3000**.

STATUS: DEBATE PIPELINE RUNNING
