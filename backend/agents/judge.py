# agents/judge.py — Helios Judge / final writer
# Author: Hridam Biswas | Project: Helios
#
# Final node of the debate pipeline (013). Reads the question, the draft,
# both critiques and the research context, decides each disputed claim, and
# WRITES THE FINAL ANSWER the user sees. Returns three scores (groundedness,
# faithfulness, completeness), a one-line verdict summarising what changed,
# and a list of the web sources actually cited.

from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent


_SYSTEM_PROMPT = """\
You are the JUDGE in Helios's debate pipeline. A draft answer was produced
by the answerer, then one Advocate argued FOR it and one Skeptic argued
AGAINST it. You must now:

1. Weigh each disputed claim, side with the evidence (web [W#] or local
   [D#]) when it's available, and your own reasoning when it's not.
2. WRITE THE FINAL ANSWER the user will see. Use the same inline citations
   ([W#], [D#]) for every fact that has a supporting source. Be concrete
   and direct — no "I" / "we", no meta commentary about the debate itself.
3. End the final answer with a "Sources" section listing every web source
   you actually cited, as:
       **Sources:**
       - [W1] Title — URL
       - [W2] Title — URL
   Only include sources you actually cited in the body. If none were
   cited, omit the Sources section.
4. Score the final answer on three dimensions, each 0.00–1.00:
     - groundedness  : are claims supported by the sources you cited?
                       If no sources were available, use 1.0 (n/a).
     - faithfulness  : do the cited sources actually say what the answer
                       says they say? If no citations, 1.0.
     - completeness  : does the answer fully address the question?
5. Give a one-line verdict: what changed vs the draft (or "confirmed draft").

Output VALID JSON only — no markdown fences, no prose before or after:
{
  "final_answer": "<the answer the user sees, markdown, with [W#]/[D#] citations and a Sources section if any were cited>",
  "scores": {
    "groundedness": 0.00,
    "faithfulness":  0.00,
    "completeness":  0.00,
    "overall":       0.00
  },
  "verdict": "<one sentence>",
  "cited_web_indices": [1, 2, ...],
  "cited_doc_indices": [1, ...]
}
Compute overall = round((g + f + c) / 3, 3).
"""


class JudgeAgent(BaseAgent):
    """Groq-backed judge that decides the debate and writes the final answer."""

    name = "judge"

    def __init__(self) -> None:
        super().__init__()
        self._model = cfg.judge_model
        self._llm = ChatGroq(
            model=self._model,
            temperature=0.1,
            api_key=cfg.groq_api_key,
        )
        self.logger.info("judge initialised: provider=groq model=%s", self._model)

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        draft: str = state.get("draft_answer", "") or ""
        advocate: dict = state.get("advocate_critique", {}) or {}
        skeptic: dict = state.get("skeptic_critique", {}) or {}
        docs: list[dict] = state.get("retrieved_docs", []) or []
        web: list[dict] = state.get("web_sources", []) or []

        doc_block = "\n\n".join(
            f"[D{i}] {d.get('document','')[:400]}"
            for i, d in enumerate(docs[:5], 1)
        ) or "(none)"
        web_block = "\n\n".join(
            f"[W{i}] {w.get('title','')}\nURL: {w.get('url','')}\n{w.get('snippet','')[:400]}"
            for i, w in enumerate(web[:cfg.debate_web_top_k], 1)
        ) or "(none)"

        user_msg = (
            f"Question: {query}\n\n"
            f"Draft (from answerer):\n{draft or '(empty)'}\n\n"
            f"Advocate critique (JSON):\n{json.dumps(advocate, ensure_ascii=False)}\n\n"
            f"Skeptic critique (JSON):\n{json.dumps(skeptic, ensure_ascii=False)}\n\n"
            f"Local documents:\n{doc_block}\n\n"
            f"Web results:\n{web_block}\n"
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        try:
            raw = self._invoke_with_timeout(messages, cfg.judge_timeout_seconds)
        except Exception as exc:
            self.logger.warning("judge failed (%s); using draft as final answer", exc)
            return self._fallback_payload(state, draft, web, docs, reason=str(exc))

        parsed = self._parse(raw)
        if not parsed or not parsed.get("final_answer"):
            self.logger.warning("judge produced empty final_answer; falling back to draft")
            return self._fallback_payload(state, draft, web, docs, reason="empty_judge_output")

        final = str(parsed.get("final_answer", "")).strip()
        scores = self._sanitize_scores(parsed.get("scores") or {})
        verdict = (parsed.get("verdict") or "").strip()[:280]

        # Figure out which web/doc sources were actually cited in the body.
        cited_web = sorted({int(x) for x in re.findall(r"\[W(\d+)\]", final)})
        cited_docs = sorted({int(x) for x in re.findall(r"\[D(\d+)\]", final)})
        sources = [
            {"index": i, "title": web[i-1].get("title",""), "url": web[i-1].get("url","")}
            for i in cited_web if 1 <= i <= len(web)
        ]

        self.logger.info(
            "judge wrote final: %d chars, scores=%s, cited_web=%s cited_docs=%s, model=%s",
            len(final), scores, cited_web, cited_docs, self._model,
        )
        return {
            **state,
            "answer": final,
            "final_scores": scores,
            "judge_verdict": verdict,
            "judge_sources": sources,
            "cited_doc_ids": [str(i) for i in cited_docs],
            "cited_web_indices": cited_web,
            "judge_provider": "groq",
            "judge_model": self._model,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _invoke_with_timeout(self, messages, timeout_seconds: float) -> str:
        def _call() -> str:
            resp = self._llm.invoke(messages, timeout=timeout_seconds)
            return (
                resp.content if isinstance(resp.content, str) else str(resp.content)
            ).strip()

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_call)
            try:
                return fut.result(timeout=timeout_seconds)
            except FuturesTimeout as exc:
                fut.cancel()
                raise TimeoutError(f"judge exceeded {timeout_seconds}s") from exc

    @staticmethod
    def _parse(raw: str) -> dict | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Strip common ```json fences before giving up.
            stripped = raw.strip().strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _sanitize_scores(raw: dict) -> dict:
        def _f(k: str, default: float = 0.0) -> float:
            try:
                v = float(raw.get(k, default))
                return max(0.0, min(1.0, v))
            except (TypeError, ValueError):
                return default
        g = _f("groundedness")
        f = _f("faithfulness")
        c = _f("completeness")
        overall = round((g + f + c) / 3, 3)
        return {"groundedness": g, "faithfulness": f, "completeness": c, "overall": overall}

    @staticmethod
    def _fallback_payload(
        state: dict[str, Any],
        draft: str,
        web: list[dict],
        docs: list[dict],  # kept for parity with the main path; unused here
        reason: str,
    ) -> dict[str, Any]:
        """Guarantee a non-empty answer even if the judge LLM fails."""
        if draft:
            final = draft
            verdict = f"judge unavailable ({reason}); surfaced draft unchanged"
        else:
            # Extremely rare: no draft and no judge. Give the research links.
            lines = [
                "I couldn't produce a final answer right now. "
                "Here are the top web results I found:",
            ]
            for i, w in enumerate(web[:5], 1):
                lines.append(f"- [W{i}] [{w.get('title','')}]({w.get('url','')})")
            final = "\n".join(lines)
            verdict = f"judge+answerer unavailable ({reason}); showing research only"

        # Minimal scores so the UI still renders.
        scores = {
            "groundedness": 0.5,
            "faithfulness": 0.5,
            "completeness": 0.5,
            "overall": 0.5,
        }
        cited_web = sorted({int(x) for x in re.findall(r"\[W(\d+)\]", final)})
        cited_docs = sorted({int(x) for x in re.findall(r"\[D(\d+)\]", final)})
        sources = [
            {"index": i, "title": web[i-1].get("title",""), "url": web[i-1].get("url","")}
            for i in cited_web if 1 <= i <= len(web)
        ]
        return {
            **state,
            "answer": final,
            "final_scores": scores,
            "judge_verdict": verdict,
            "judge_sources": sources,
            "cited_doc_ids": [str(i) for i in cited_docs],
            "cited_web_indices": cited_web,
            "judge_provider": "groq",
            "judge_model": cfg.judge_model,
            "judge_fallback": True,
        }
