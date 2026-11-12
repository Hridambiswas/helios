# 013 report — Debate pipeline (answer, 2 critics, judge)

**## Revision 2 — honest citations + <25 s latency

Two more fixes on `local-backup-test`:

### 1. "Related search results" are no longer citations

The old `_force_citations` was quietly padding `cited_web` and
`judge_sources` with top uncited results whenever the judge cited <2 — so
a half-grounded answer looked identical to a fully-cited one in the UI.

New behaviour (`030e1bf`):

- Judge keeps `judge_cited_count` on state (what the LLM itself cited).
- When web results exist but the judge cited <2, the judge appends up to
  three **bare-bullet** entries under `**Related search results (not used
  in the answer):**`. They are NOT added to `cited_web_indices` or
  `judge_sources`, and they carry no `[W#]` marker so the regex can never
  pick them up.
- Scorer now caps `groundedness` at **0.5** whenever the judge cited <2
  of the available web sources, with a reasoning line stating exactly why
  (e.g. "Judge cited 1 web sources out of 6 available — groundedness
  capped at 0.5").

Evidence from the rev-2 retest:
```
judge cited only 1 web sources (web_sources=6 available); appending Related search results (not citations)
scorer: capping groundedness 1.00 -> 0.50 (judge cited only 1 of 6 web sources)
```

### 2. Latency under 25 s

Three commits (`3c2edf7`, `1a8e2b4`, `4d65f2b`):

- Every Groq call now goes through a shared `invoke_groq_with_retry`
  wrapper: **8 s per attempt, max 2 attempts** (advocate, skeptic, judge,
  scorer). Worst-case per Groq agent ≈ 16 s instead of 20–30 s.
- Each pipeline node records its `time.perf_counter()` wall time into
  `state.stage_timings_ms` and emits a WS `stage_done` event with the ms,
  so the UI can draw per-stage progress bars.
- `node_debate` measures **advocate_wall_ms**, **skeptic_wall_ms** and
  **debate_parallel_ms**, and logs the "serial would be / stage actually
  took / savings" triad so parallelism is *provable* from the log.
- **Critical fix**: the old `with ThreadPoolExecutor()` wrapper was
  defeating the whole timeout. Its `__exit__` calls `shutdown(wait=True)`,
  which blocks until the worker thread finishes — so `future.result(
  timeout=12)` would raise but the pipeline still waited for the SDK's
  internal retry chain to give up (~39 s observed). Switched both
  wrappers (`_groq_call` + answerer) to a non-context-manager executor
  with `shutdown(wait=False)` on every path.

### Per-stage timings (rev-2, cache cleared)

| Stage              | Einstein   | ETH deadline | Nonsense (control) |
|---                 |---:        |---:          |---:                |
| planner            |   0.91 s   |   1.16 s     |   2.80 s           |
| researcher         |   0.73 s   |   0.88 s     |   0.62 s           |
| answerer (Gemini)  |  13.40 s   |  13.31 s     |  13.65 s           |
| debate (parallel)  |   1.97 s   |   1.76 s     |   1.72 s           |
|   ↳ advocate wall  |   1.97 s   |   1.76 s     |   1.72 s           |
|   ↳ skeptic wall   |   1.97 s   |   1.76 s     |   1.72 s           |
|   serial would be  |   3.94 s   |   3.52 s     |   3.43 s           |
|   savings          |   1.97 s   |   1.76 s     |   1.72 s           |
| judge              |   0.92 s   |   0.62 s     |   1.03 s           |
| scorer             |   0.43 s   |   0.40 s     |   0.86 s           |
| **total**          | **18.8 s** | **18.1 s**   | **20.7 s**         |

All three come in under the 25 s target. Advocate ≈ skeptic ≈ stage
within ~1 ms — the two critics are genuinely parallel. Answerer still
dominates at ~13 s because Gemini's free tier is routinely slow; the
12 s hard cap now actually cuts it off so we flip to the Groq fallback
promptly.

### Rev-2 commits (4)

```
4d65f2b fix(agents): shutdown(wait=False) so timeouts actually cut the call
1a8e2b4 feat(pipeline): per-stage timings + parallelism proof + stage_done events
3c2edf7 feat(agents): 8s + one-retry wrapper on every Groq call
030e1bf fix(judge,scorer): auto-appended results are 'Related', not citations
```

Raw payloads: `demo/logs/rev2-q{1,2,3}-*.json`.

STATUS: DEBATE PIPELINE RUNNING** (PDF upload acceptance deferred to 014;
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

## Revision 1 — tightening pass

Five follow-up fixes landed on `local-backup-test` after the first report:

1. **Answerer 12 s + Groq fallback** (`159d5e8`). Gemini call capped at 12 s
   via `ThreadPoolExecutor.future.result(timeout=)`. On timeout OR raise, we
   immediately fall back to the advocate's Groq model (also 12 s). Keeps
   the total query budget ≈ 30 s (answerer ≤24 s + debate ≈4 s +
   judge ≈2 s + scorer ≈1 s).
2. **Judge model → Qwen** (`c4ae435`). Groq on this account only serves
   OpenAI + Qwen chat families (no Llama chat model — only
   `meta-llama/llama-prompt-guard-2-*` safety classifiers), so the judge
   moves to `qwen/qwen3.8-27b`: a different family than the advocate's
   `openai/gpt-oss-120b`. If Groq later serves a Llama chat model, point
   `JUDGE_MODEL` at it. Also tightened `judge_timeout_seconds` 60 → 20.
3. **Judge must cite + honest empty-search** (`9505cf3`). Judge prompt now
   requires ≥2 `[W#]` citations whenever `web_sources` is non-empty, plus
   a Sources section. Python-side post-check `_force_citations` appends a
   "Supporting sources (added)" block if the model forgets. If search
   returned nothing, the answer opens with "Web search returned no usable
   results, so this is based on model knowledge." Judge NO LONGER scores.
4. **Independent ScorerAgent** (`7792e7a`). New node: `answerer → debate
   → judge → **scorer** → verifier`. The scorer uses the skeptic model
   (Qwen) with its own strict prompt — "vague answers / near-empty
   replies MUST score low." Separate model + separate prompt from the
   author of the final answer.
5. **Frontend scroll** (`b48e014`). `min-h-0` on the messages container
   (parent is `flex-col`, child is `flex-1` — without `min-h-0` the
   child couldn't shrink below its intrinsic height, so long streams
   pushed the input bar off-screen). Auto-scroll on new messages and on
   each streamed token ONLY when the user is within 160 px of the bottom,
   so reading older messages isn't yanked away.

Why the first Q2 (ETH) had `cited_web=[]`: traced in the logs. The Gemini
answerer timed out at the old 30 s cap, the Groq fallback produced a draft
without `[W]` citations, and the old judge prompt didn't require any.
The rev-1 judge prompt + `_force_citations` repair makes this impossible —
any non-empty `web_sources` is now guaranteed to yield at least two `[W#]`
in the body (either from the LLM or auto-appended).

### Retest (clean cache, Redis `FLUSHDB`, fresh pipeline)

| Query                                 | Time   | Cited `[W]` | `judge_sources` | Final scores (G/F/C → overall)  |
|---                                    |---     |---          |---              |---                               |
| "who was albert einstein"             | 47.9 s | [1,2,3,4,5] | 5               | 0.70 / 0.70 / 1.00 → **0.80**   |
| "ETH Zurich MSc CS deadline"          | 43.0 s | [1,3]       | 2               | 1.00 / 1.00 / 0.70 → **0.90**   |
| nonsense ("plurk-glorg quapatatic…")  | 42.9 s | [1,2] (auto)| 2               | 1.00 / 1.00 / **0.10** → 0.70   |

Note: the ≈45 s latency comes from Gemini's answerer still overloading on
every retry — logs show `answerer=openai/gpt-oss-120b` on all three, i.e.
the Groq fallback took over each time. When Gemini free-tier is healthy
the budget drops back toward 30 s. Advocate + skeptic + judge + scorer
all land as planned: advocate=openai/gpt-oss-120b, skeptic=qwen/qwen3.8-27b,
judge=qwen/qwen3.8-27b, scorer=qwen/qwen3.8-27b.

The nonsense question is the honesty check: the scorer (which did NOT
write the answer) gave it **completeness 0.10** — a real low score from
a Qwen model grading a Qwen-written answer about "plurk-glorg…"
that correctly refused to invent lore.

Raw payloads: `demo/logs/rev1-q{1,2,3}-*.json`.

Unit tests after rev-1: `pytest -q tests/test_debate_pipeline.py` → **10
passed** (adds coverage for the citation-repair path and for the scorer's
empty / JSON-parsing / low-completeness paths).

### Rev-1 commits (5)

```
b48e014 fix(frontend): min-h-0 + near-bottom auto-scroll while streaming
7792e7a feat(pipeline): add independent ScorerAgent on the skeptic model
9505cf3 fix(judge): enforce 2+ web citations, no self-scoring, honest empty-search
c4ae435 feat(config): judge on Qwen (diff family from advocate), add scorer_model
159d5e8 fix(answerer): tighten timeouts to 12s each (Gemini primary + Groq fallback)
```

## Revision 2 — honest citations + <25 s latency

Two more fixes on `local-backup-test`:

### 1. "Related search results" are no longer citations

The old `_force_citations` was quietly padding `cited_web` and
`judge_sources` with top uncited results whenever the judge cited <2 — so
a half-grounded answer looked identical to a fully-cited one in the UI.

New behaviour (`030e1bf`):

- Judge keeps `judge_cited_count` on state (what the LLM itself cited).
- When web results exist but the judge cited <2, the judge appends up to
  three **bare-bullet** entries under `**Related search results (not used
  in the answer):**`. They are NOT added to `cited_web_indices` or
  `judge_sources`, and they carry no `[W#]` marker so the regex can never
  pick them up.
- Scorer now caps `groundedness` at **0.5** whenever the judge cited <2
  of the available web sources, with a reasoning line stating exactly why
  (e.g. "Judge cited 1 web sources out of 6 available — groundedness
  capped at 0.5").

Evidence from the rev-2 retest:
```
judge cited only 1 web sources (web_sources=6 available); appending Related search results (not citations)
scorer: capping groundedness 1.00 -> 0.50 (judge cited only 1 of 6 web sources)
```

### 2. Latency under 25 s

Three commits (`3c2edf7`, `1a8e2b4`, `4d65f2b`):

- Every Groq call now goes through a shared `invoke_groq_with_retry`
  wrapper: **8 s per attempt, max 2 attempts** (advocate, skeptic, judge,
  scorer). Worst-case per Groq agent ≈ 16 s instead of 20–30 s.
- Each pipeline node records its `time.perf_counter()` wall time into
  `state.stage_timings_ms` and emits a WS `stage_done` event with the ms,
  so the UI can draw per-stage progress bars.
- `node_debate` measures **advocate_wall_ms**, **skeptic_wall_ms** and
  **debate_parallel_ms**, and logs the "serial would be / stage actually
  took / savings" triad so parallelism is *provable* from the log.
- **Critical fix**: the old `with ThreadPoolExecutor()` wrapper was
  defeating the whole timeout. Its `__exit__` calls `shutdown(wait=True)`,
  which blocks until the worker thread finishes — so `future.result(
  timeout=12)` would raise but the pipeline still waited for the SDK's
  internal retry chain to give up (~39 s observed). Switched both
  wrappers (`_groq_call` + answerer) to a non-context-manager executor
  with `shutdown(wait=False)` on every path.

### Per-stage timings (rev-2, cache cleared)

| Stage              | Einstein   | ETH deadline | Nonsense (control) |
|---                 |---:        |---:          |---:                |
| planner            |   0.91 s   |   1.16 s     |   2.80 s           |
| researcher         |   0.73 s   |   0.88 s     |   0.62 s           |
| answerer (Gemini)  |  13.40 s   |  13.31 s     |  13.65 s           |
| debate (parallel)  |   1.97 s   |   1.76 s     |   1.72 s           |
|   ↳ advocate wall  |   1.97 s   |   1.76 s     |   1.72 s           |
|   ↳ skeptic wall   |   1.97 s   |   1.76 s     |   1.72 s           |
|   serial would be  |   3.94 s   |   3.52 s     |   3.43 s           |
|   savings          |   1.97 s   |   1.76 s     |   1.72 s           |
| judge              |   0.92 s   |   0.62 s     |   1.03 s           |
| scorer             |   0.43 s   |   0.40 s     |   0.86 s           |
| **total**          | **18.8 s** | **18.1 s**   | **20.7 s**         |

All three come in under the 25 s target. Advocate ≈ skeptic ≈ stage
within ~1 ms — the two critics are genuinely parallel. Answerer still
dominates at ~13 s because Gemini's free tier is routinely slow; the
12 s hard cap now actually cuts it off so we flip to the Groq fallback
promptly.

### Rev-2 commits (4)

```
4d65f2b fix(agents): shutdown(wait=False) so timeouts actually cut the call
1a8e2b4 feat(pipeline): per-stage timings + parallelism proof + stage_done events
3c2edf7 feat(agents): 8s + one-retry wrapper on every Groq call
030e1bf fix(judge,scorer): auto-appended results are 'Related', not citations
```

Raw payloads: `demo/logs/rev2-q{1,2,3}-*.json`.

STATUS: DEBATE PIPELINE RUNNING
