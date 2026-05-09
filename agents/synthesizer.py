# agents/synthesizer.py — Helios Synthesizer Agent
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.synthesizer")

_SYSTEM_PROMPT_RAG = """\
You are the Synthesizer agent in Helios — a grounded answer writer.

Your job: produce a final, comprehensive answer by weaving together:
  1. Retrieved document chunks (the primary source of truth)
  2. Code execution output (if present)

Writing rules:
  - Ground every factual claim in the retrieved context. Never hallucinate.
  - Cite sources inline using [doc_id] notation immediately after each claim.
  - If code was executed, integrate its output naturally — don't just dump stdout.
  - If the context is insufficient to fully answer the query, say so clearly
    rather than guessing or padding with generic statements.
  - Prefer clear, concise prose. Use bullet points or code blocks only when
    they genuinely improve clarity.
  - Do not repeat the query verbatim at the start of your answer.
  - Match the depth of the answer to the complexity of the query.
"""

_SYSTEM_PROMPT_DIRECT = """\
You are the Synthesizer agent in Helios — a helpful, direct answer writer.

Your job: answer the user's query directly from your own knowledge.
No documents were retrieved because none are needed for this query.

Writing rules:
  - Answer clearly and concisely from your general knowledge.
  - Do NOT mention documents, context, or retrieval — they are irrelevant here.
  - If code was executed, integrate its output naturally.
  - Do not repeat the query verbatim at the start of your answer.
  - Match the depth of the answer to the complexity of the query.
"""


def _format_docs(docs: list[dict]) -> str:
    if not docs:
        return "No documents retrieved."
    lines = []
    for d in docs:
        doc_id = d.get("id", "?")
        score = d.get("score", 0.0)
        snippet = d.get("document", "")[:500]
        lines.append(f"[{doc_id}] (score={score:.3f})\n{snippet}")
    return "\n\n".join(lines)


def _format_execution(result: dict | None) -> str:
    if not result:
        return ""
    if result.get("success"):
        return f"Code output:\n{result.get('stdout', '').strip()}"
    return f"Code failed: {result.get('error', 'unknown error')}"


class SynthesizerAgent(BaseAgent):
    """
    Combines retrieved docs and execution results into a grounded final answer.
    Injects document citations using [doc_id] notation.
    """

    name = "synthesizer"

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatGroq(
            model=cfg.groq_model,
            temperature=0.2,
            api_key=cfg.groq_api_key,
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        docs: list[dict] = state.get("retrieved_docs", [])
        exec_result: dict | None = state.get("execution_result")
        requires_retrieval: bool = state.get("plan", {}).get("requires_retrieval", True)

        exec_block = _format_execution(exec_result)

        # Use RAG prompt only when documents were actually retrieved
        if docs:
            system_prompt = _SYSTEM_PROMPT_RAG
            context_section = f"--- Retrieved Context ---\n{_format_docs(docs)}\n"
        elif requires_retrieval:
            # Retrieval was attempted but returned nothing
            system_prompt = _SYSTEM_PROMPT_RAG
            context_section = "--- Retrieved Context ---\nNo relevant documents found in the knowledge base.\n"
        else:
            # No retrieval needed — answer directly
            system_prompt = _SYSTEM_PROMPT_DIRECT
            context_section = ""

        user_msg = f"""User query: {query}

{context_section}--- Code Execution Result ---
{exec_block if exec_block else "N/A"}

Now produce the final answer:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ]
        response = self._llm.invoke(messages, timeout=45)
        answer = (response.content if isinstance(response.content, str) else str(response.content)).strip()

        # Extract cited doc IDs from the answer
        import re
        cited_ids = list(set(re.findall(r"\[([^\]]+)\]", answer)))

        self.logger.info(
            "Synthesized answer: %d chars, cited_docs=%s", len(answer), cited_ids
        )

        # Append citation list at end of answer for traceability
        if cited_ids and docs:
            doc_map = {d["id"]: d for d in docs}
            citation_block = "\n\n---\n**Sources:**"
            for cid in cited_ids:
                if cid in doc_map:
                    meta = doc_map[cid].get("metadata", {})
                    fname = meta.get("filename", cid)
                    citation_block += f"\n- [{cid}] {fname}"
            answer += citation_block

        return {**state, "answer": answer, "cited_doc_ids": cited_ids}
