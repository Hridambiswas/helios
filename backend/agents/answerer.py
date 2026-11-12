# agents/answerer.py — Helios Answerer (draft) agent
# Author: Hridam Biswas | Project: Helios
#
# First stage of the debate pipeline (013). Uses Gemini to draft an initial
# answer from its own knowledge + the research context (web sources + any
# uploaded documents). The draft is then interrogated in parallel by the
# advocate and skeptic critics before the judge writes the final answer.

from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.answerer")


_SYSTEM_PROMPT = """\
You are the Answerer in Helios's debate pipeline — you write the FIRST DRAFT.

Draft a helpful, grounded answer to the user's question. Use the research
context when it's relevant; fall back on your own knowledge when the context
is thin. Be direct, concrete and specific.

Citation rules:
- Cite web sources inline as [W1], [W2], … matching the numbered list below.
- Cite local documents inline as [D1], [D2], … matching the numbered list.
- You MAY state knowledge without a citation when no source covers it, but
  flag it as common knowledge rather than invent a citation.

Length: aim for 120–350 words. Start answering immediately — no preamble.
End the answer with a one-line "Draft note:" summarising what you were less
sure about (one sentence; "none" if you're confident).
"""


class AnswererAgent(BaseAgent):
    """Gemini-backed drafter that produces the initial answer the critics interrogate."""

    name = "answerer"

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatGoogleGenerativeAI(
            model=cfg.answerer_model,
            temperature=0.3,
            google_api_key=cfg.gemini_api_key.get_secret_value(),
        )
        self.logger.info("answerer initialised: provider=gemini model=%s", cfg.answerer_model)

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        docs: list[dict] = state.get("retrieved_docs", []) or []
        web: list[dict] = state.get("web_sources", []) or []

        doc_block = "\n\n".join(
            f"[D{i}] {d.get('document','')[:500]}"
            for i, d in enumerate(docs[:5], 1)
        ) or "(none)"
        web_block = "\n\n".join(
            f"[W{i}] {w.get('title','')}\nURL: {w.get('url','')}\n{w.get('snippet','')[:400]}"
            for i, w in enumerate(web[:cfg.debate_web_top_k], 1)
        ) or "(none)"

        user_msg = (
            f"Question: {query}\n\n"
            f"Local documents (may be empty):\n{doc_block}\n\n"
            f"Web search results (may be empty):\n{web_block}\n"
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
            draft = self._invoke_with_timeout(messages, 30.0)
        except Exception as exc:
            self.logger.warning("answerer failed (%s); using empty draft", exc)
            draft = ""

        if not draft:
            # Non-fatal: the judge can still rescue the pipeline from the
            # research context alone. Surface the empty state in logs.
            self.logger.warning("answerer produced empty draft")

        self.logger.info(
            "answerer draft: %d chars, provider=gemini model=%s", len(draft), cfg.answerer_model
        )
        return {
            **state,
            "draft_answer": draft,
            "answerer_provider": "gemini",
            "answerer_model": cfg.answerer_model,
        }

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
                raise TimeoutError(f"answerer exceeded {timeout_seconds}s") from exc
