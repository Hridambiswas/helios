# agents/planner.py — Helios Planner Agent
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.planner")

_SYSTEM_PROMPT = """\
You are the Planner agent in Helios, an agentic AI research assistant.

Your job: analyse the user's query and decompose it into an ordered list of
subtasks that downstream agents (retriever, executor, synthesizer) will execute.

Rules:
1. Produce between 1 and {max_subtasks} atomic, unambiguous subtasks.
2. Classify each subtask as: retrieve | execute | synthesize | answer_direct.
   - retrieve  → look up information from the knowledge base
   - execute   → run Python code to compute or transform data
   - synthesize → combine retrieved evidence into a coherent answer
   - answer_direct → no retrieval needed; answer from LLM knowledge
3. Set requires_retrieval=false only when the query is clearly factual and
   can be answered from LLM parametric knowledge alone (e.g. "what is 2+2?").
4. Set requires_code=true only when numerical computation or data manipulation
   is explicitly requested.
5. Output ONLY valid JSON — no markdown fences, no prose before or after.

Output schema (strict):
{{
  "query_type": "factual | analytical | code | multi_step",
  "subtasks": [
    {{"id": 1, "type": "retrieve | execute | synthesize | answer_direct", "description": "..."}}
  ],
  "requires_retrieval": true | false,
  "requires_code": true | false
}}
"""


class PlannerAgent(BaseAgent):
    name = "planner"

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatGroq(
            model=cfg.groq_model,
            temperature=0,
            api_key=cfg.groq_api_key,
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        history: list[dict] = state.get("conversation_history", [])
        self.logger.info("Planning query: %.80s ...", query)

        context_note = ""
        if history:
            last_user = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
            if last_user:
                context_note = f"\n\nConversation context (most recent prior user message): {last_user[:200]}"

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(max_subtasks=cfg.planner_max_subtasks)),
            HumanMessage(content=query + context_note),
        ]
        response = self._llm.invoke(messages, timeout=30)
        raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            # Graceful fallback: treat as single retrieval subtask
            self.logger.warning("Planner returned non-JSON; falling back to default plan")
            plan = {
                "query_type": "factual",
                "subtasks": [{"id": 1, "type": "retrieve", "description": query}],
                "requires_retrieval": True,
                "requires_code": False,
            }

        # Enforce max_subtasks cap
        plan["subtasks"] = plan["subtasks"][: cfg.planner_max_subtasks]

        # Resolve task dependencies: ensure each id is unique and ascending
        for i, task in enumerate(plan["subtasks"], 1):
            task["id"] = i   # re-number in case LLM produced duplicates

        self.logger.info(
            "Plan: type=%s  subtasks=%d  retrieval=%s  code=%s",
            plan.get("query_type"),
            len(plan["subtasks"]),
            plan.get("requires_retrieval"),
            plan.get("requires_code"),
        )
        return {**state, "plan": plan}
