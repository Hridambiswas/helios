# agents/synthesizer.py — Helios Synthesizer Agent
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.synthesizer")

_SYSTEM_PROMPT = """\
You are the Synthesizer agent in a multi-agent AI system called Helios.

Your job: produce a final, complete answer to the user's query by combining:
  1. Retrieved document chunks (semantic context)
  2. Code execution output (if present)

Rules:
  - Ground your answer in the provided context. Do not hallucinate.
  - Cite sources inline using [doc_id] notation where relevant.
  - If code was executed, integrate its output naturally into the answer.
  - If the context is insufficient, say so explicitly rather than guessing.
  - Write in clear, concise prose unless the query asks for code or structured output.
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
        self._llm = ChatOpenAI(
            model=cfg.openai_model,
            temperature=0.2,
            api_key=cfg.openai_api_key,
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        docs: list[dict] = state.get("retrieved_docs", [])
        exec_result: dict | None = state.get("execution_result")

        context_block = _format_docs(docs)
        exec_block = _format_execution(exec_result)

        user_msg = f"""User query: {query}

--- Retrieved Context ---
{context_block}

--- Code Execution Result ---
{exec_block if exec_block else "N/A"}

Now produce the final answer:"""

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        response = self._llm.invoke(messages)
        answer = response.content.strip()

        # Extract cited doc IDs from the answer
        import re
        cited_ids = list(set(re.findall(r"\[([^\]]+)\]", answer)))

        self.logger.info(
            "Synthesized answer: %d chars, cited_docs=%s", len(answer), cited_ids
        )
        return {**state, "answer": answer, "cited_doc_ids": cited_ids}
