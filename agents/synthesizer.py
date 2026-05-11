# agents/synthesizer.py — Helios Synthesizer Agent
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import re
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.synthesizer")

_FOLLOW_UP_SEPARATOR = "<<<FOLLOW_UPS>>>"

_SYSTEM_PROMPT = """\
You are Helios, a smart and conversational AI assistant. You have access to both
a local knowledge base and live web search results to give the most accurate,
up-to-date answers.

Answer naturally and helpfully — like you're explaining something to a curious friend.
Be warm, clear, and direct. Never sound robotic.

Citation rules:
- Cite local documents inline as [D1], [D2], etc.
- Cite web sources inline as [W1], [W2], etc.
- If neither local docs nor web results are available, answer from your own knowledge.

At the very end of your response, add this section — always, no exceptions:

<<<FOLLOW_UPS>>>
1. [a specific follow-up question the user would naturally want to ask next]
2. [another specific follow-up question that deepens understanding]

The follow-up questions must be relevant to what you just answered.
Do not repeat the original query. Do not add any text after the two questions.
"""


def _format_local_docs(docs: list[dict]) -> str:
    if not docs:
        return ""
    lines = ["**Local Knowledge Base:**"]
    for i, d in enumerate(docs[:5], 1):
        snippet = d.get("document", "")[:400].strip()
        lines.append(f"[D{i}] {snippet}")
    return "\n\n".join(lines)


def _format_web_sources(web: list[dict]) -> str:
    if not web:
        return ""
    lines = ["**Web Search Results:**"]
    for i, w in enumerate(web, 1):
        title = w.get("title", "")
        url = w.get("url", "")
        snippet = w.get("snippet", "")[:400].strip()
        lines.append(f"[W{i}] {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(lines)


def _format_execution(result: dict | None) -> str:
    if not result:
        return ""
    if result.get("success"):
        return f"Code output:\n{result.get('stdout', '').strip()}"
    return f"Code failed: {result.get('error', 'unknown error')}"


def _parse_follow_ups(text: str) -> tuple[str, list[str]]:
    """Split answer from follow-up questions. Returns (clean_answer, [q1, q2])."""
    if _FOLLOW_UP_SEPARATOR not in text:
        return text.strip(), []
    parts = text.split(_FOLLOW_UP_SEPARATOR, 1)
    answer = parts[0].strip()
    follow_up_block = parts[1].strip()
    questions = []
    for line in follow_up_block.splitlines():
        line = line.strip()
        m = re.match(r"^[12][.)]\s*(.+)", line)
        if m:
            questions.append(m.group(1).strip())
    return answer, questions[:2]


class SynthesizerAgent(BaseAgent):
    name = "synthesizer"

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatGroq(
            model=cfg.groq_model,
            temperature=0.4,
            api_key=cfg.groq_api_key,
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        docs: list[dict] = state.get("retrieved_docs", [])
        web: list[dict] = state.get("web_sources", [])
        exec_result: dict | None = state.get("execution_result")

        local_block = _format_local_docs(docs)
        web_block = _format_web_sources(web)
        exec_block = _format_execution(exec_result)

        context_parts = [p for p in [local_block, web_block] if p]
        context_section = "\n\n".join(context_parts) if context_parts else "No external context available — answer from your knowledge."

        user_msg = f"""User question: {query}

--- Context ---
{context_section}

{('--- Code Execution Result ---\n' + exec_block) if exec_block else ''}

Now answer the question. Remember to end with the <<<FOLLOW_UPS>>> section containing exactly 2 follow-up questions."""

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        response = self._llm.invoke(messages, timeout=45)
        raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()

        answer, follow_ups = _parse_follow_ups(raw)

        # Build source citation block at end of answer
        cited_doc_ids = list(set(re.findall(r"\[D(\d+)\]", answer)))
        cited_web_ids = list(set(re.findall(r"\[W(\d+)\]", answer)))

        if cited_web_ids and web:
            citation_lines = ["\n\n---\n**Sources:**"]
            for wid in sorted(cited_web_ids, key=int):
                idx = int(wid) - 1
                if 0 <= idx < len(web):
                    w = web[idx]
                    citation_lines.append(f"- [W{wid}] [{w['title']}]({w['url']})")
            answer += "\n".join(citation_lines)

        self.logger.info(
            "Synthesized: %d chars, local_docs_cited=%s web_cited=%s follow_ups=%d",
            len(answer), cited_doc_ids, cited_web_ids, len(follow_ups),
        )

        return {
            **state,
            "answer": answer,
            "cited_doc_ids": cited_doc_ids,
            "follow_up_questions": follow_ups,
            "web_sources": web,
        }
