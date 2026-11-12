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
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent
from agents._groq_call import invoke_groq_with_retry


_SYSTEM_PROMPT = """\
You are the JUDGE in Helios's debate pipeline. A draft answer was produced
by the answerer, then one Advocate argued FOR it and one Skeptic argued
AGAINST it. You must now:

1. Weigh each disputed claim. Side with the evidence (web [W#] or local
   [D#]) when it's available; use your own reasoning when it's not.
2. WRITE THE FINAL ANSWER the user will see. Hard rules:
   - If the Web results list is NON-EMPTY, you MUST cite at least two
     distinct web sources as [W#] in the body. Pick the most relevant
     web results for this specific question. Never invent URLs.
   - If the Web results list is EMPTY, open the answer with one sentence
     that says so plainly, e.g. "Web search returned no usable results,
     so this is based on model knowledge." Then answer from your own
     knowledge without inventing citations.
   - Cite local docs as [D#] for every fact backed by one.
   - Be concrete and direct. No "I" / "we", no meta commentary about
     the debate itself.
3. End the final answer with a "Sources" section listing EVERY web
   source you cited (title + url). If the Web results list was empty,
   omit the Sources section.
4. Give a one-line verdict: what changed vs the draft (or "confirmed draft").

Output VALID JSON only — no markdown fences, no prose before or after:
{
  "final_answer": "<the answer the user sees, markdown, with [W#]/[D#] citations and a Sources section if web results were available>",
  "verdict": "<one sentence>",
  "cited_web_indices": [1, 2, ...],
  "cited_doc_indices": [1, ...]
}
Do NOT include scores — a separate scoring step runs afterwards.
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
            raw = invoke_groq_with_retry(
                self._llm, messages,
                agent_name="judge",
                per_call_timeout=8.0,
                attempts=2,
            )
        except Exception as exc:
            self.logger.warning("judge failed (%s); using draft as final answer", exc)
            return self._fallback_payload(state, draft, web, docs, reason=str(exc))

        parsed = self._parse(raw)
        if not parsed or not parsed.get("final_answer"):
            self.logger.warning("judge produced empty final_answer; falling back to draft")
            return self._fallback_payload(state, draft, web, docs, reason="empty_judge_output")

        final = str(parsed.get("final_answer", "")).strip()
        verdict = (parsed.get("verdict") or "").strip()[:280]

        # Figure out which web/doc sources were actually cited in the body.
        cited_web = sorted({int(x) for x in re.findall(r"\[W(\d+)\]", final)})
        cited_docs = sorted({int(x) for x in re.findall(r"\[D(\d+)\]", final)})

        # Number of citations the JUDGE itself produced in the body.
        # Used by the scorer to penalise low-citation answers.
        judge_cited_count = len(cited_web)
        auto_related_added = False

        # If web_sources existed but the judge cited <2, append a labeled
        # "Related search results (not used in the answer)" block. These are
        # NOT citations — they do not go into cited_web or judge_sources,
        # and the scorer knows to dock groundedness.
        if web and judge_cited_count < 2:
            self.logger.warning(
                "judge cited only %d web sources (web_sources=%d available); "
                "appending Related search results (not citations)",
                judge_cited_count, len(web),
            )
            final = self._append_related_results(final, web, already_cited=cited_web)
            auto_related_added = True
            verdict = (verdict + " [related results appended]")[:280]

        # If the judge DID cite but forgot the Sources block, add it now.
        # Only citations the judge itself made are listed here.
        if cited_web and "**Sources:**" not in final and "Sources:" not in final[-400:]:
            final = self._append_sources_block(final, web, cited_web)

        # If search returned nothing, open the answer with the honest note.
        if not web and not cited_web:
            opener = "Web search returned no usable results, so this is based on model knowledge. "
            if not final.lower().startswith("web search"):
                final = opener + final

        # judge_sources contains ONLY what the judge actually cited.
        sources = [
            {"index": i, "title": web[i-1].get("title",""), "url": web[i-1].get("url","")}
            for i in cited_web if 1 <= i <= len(web)
        ]

        self.logger.info(
            "judge wrote final: %d chars, cited_web=%s cited_docs=%s, model=%s "
            "(web_sources_in=%d, auto_related=%s)",
            len(final), cited_web, cited_docs, self._model, len(web), auto_related_added,
        )
        return {
            **state,
            "answer": final,
            # final_scores intentionally omitted here — ScorerAgent fills it.
            "judge_verdict": verdict,
            "judge_sources": sources,
            "judge_cited_count": judge_cited_count,
            "judge_auto_related": auto_related_added,
            "cited_doc_ids": [str(i) for i in cited_docs],
            "cited_web_indices": cited_web,
            "judge_provider": "groq",
            "judge_model": self._model,
        }

    @staticmethod
    def _append_related_results(
        final: str, web: list[dict], already_cited: list[int], wanted: int = 3
    ) -> str:
        """
        When the judge cites <2 web sources, append up to `wanted` of the
        top uncited search results under a header that is EXPLICITLY not a
        citation: 'Related search results (not used in the answer)'. This
        keeps the UI's sources list honest — only sources the judge truly
        used end up as citations — while still exposing what the retriever
        pulled up.

        Deliberately uses bare bullets (no [W#]) so the body regex that
        feeds cited_web never picks these up.
        """
        picked: list[int] = []
        for i, _ in enumerate(web, 1):
            if len(picked) >= wanted:
                break
            if i in already_cited:
                continue
            picked.append(i)
        if not picked:
            return final
        tail_lines = [
            "",
            "**Related search results (not used in the answer):**",
        ]
        for i in picked:
            w = web[i-1]
            title = w.get("title", "") or w.get("url", "")
            tail_lines.append(f"- {title} — {w.get('url','')}")
        return final.rstrip() + "\n\n" + "\n".join(tail_lines)

    @staticmethod
    def _append_sources_block(final: str, web: list[dict], cited: list[int]) -> str:
        lines = ["", "**Sources:**"]
        for i in cited:
            if 1 <= i <= len(web):
                w = web[i-1]
                lines.append(f"- [W{i}] {w.get('title','') or w.get('url','')} — {w.get('url','')}")
        return final.rstrip() + "\n\n" + "\n".join(lines)

    # ── helpers ───────────────────────────────────────────────────────────────

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
            if web:
                lines = [
                    "Web search returned results, but the final writer was "
                    "unavailable. Here are the top web results that matched "
                    "the query:"
                ]
                for i, w in enumerate(web[:5], 1):
                    lines.append(f"- [W{i}] [{w.get('title','')}]({w.get('url','')})")
                final = "\n".join(lines)
            else:
                final = (
                    "Web search returned no usable results, so this is based "
                    "on model knowledge — but the final writer was "
                    "unavailable. Please retry in a moment."
                )
            verdict = f"judge+answerer unavailable ({reason}); showing research only"

        cited_web = sorted({int(x) for x in re.findall(r"\[W(\d+)\]", final)})
        cited_docs = sorted({int(x) for x in re.findall(r"\[D(\d+)\]", final)})
        sources = [
            {"index": i, "title": web[i-1].get("title",""), "url": web[i-1].get("url","")}
            for i in cited_web if 1 <= i <= len(web)
        ]
        return {
            **state,
            "answer": final,
            # scorer will still run on this and produce proper scores.
            "judge_verdict": verdict,
            "judge_sources": sources,
            "cited_doc_ids": [str(i) for i in cited_docs],
            "cited_web_indices": cited_web,
            "judge_provider": "groq",
            "judge_model": cfg.judge_model,
            "judge_fallback": True,
        }
