# agents/planner.py — Helios Planner Agent
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.planner")

_SYSTEM_PROMPT = """\
You are the Planner agent in a multi-agent AI system called Helios.

Your job: decompose the user's query into an ordered list of subtasks that
downstream agents (retriever, executor, synthesizer) will execute.

Rules:
1. Produce between 1 and {max_subtasks} subtasks.
2. Each subtask must be atomic and unambiguous.
3. Mark subtasks as one of: [retrieve | execute | synthesize | answer_direct].
4. If the query can be answered directly without retrieval or code, use answer_direct.
5. Output ONLY valid JSON — no markdown, no preamble.

Output schema:
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
        self._llm = ChatOpenAI(
            model=cfg.openai_model,
            temperature=0,
            api_key=cfg.openai_api_key,
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        self.logger.info("Planning query: %.80s ...", query)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(max_subtasks=cfg.planner_max_subtasks)),
            HumanMessage(content=query),
        ]
        response = self._llm.invoke(messages)
        raw = response.content.strip()

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
