# agents/critic.py — Helios LLM-as-Judge Critic Agent
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.critic")

_SYSTEM_PROMPT = """\
You are the Critic agent in a multi-agent AI system called Helios.
Your role: evaluate the quality of a generated answer along three dimensions.

Scoring rubric (each dimension scored 0.0 – 1.0):

GROUNDEDNESS (0–1):
  How well is every claim in the answer supported by the provided context?
  1.0 = every claim has direct evidence in context
  0.5 = some claims supported, some speculative
  0.0 = answer is not grounded in context at all

FAITHFULNESS (0–1):
  Does the answer accurately represent what the context says, without distortion?
  1.0 = no hallucinations, no contradictions
  0.5 = minor paraphrasing errors
  0.0 = contradicts or significantly misrepresents context

COMPLETENESS (0–1):
  Does the answer fully address all aspects of the query?
  1.0 = all sub-questions answered
  0.5 = partial coverage
  0.0 = query largely unanswered

Output ONLY valid JSON — no markdown, no preamble:
{
  "groundedness": 0.0,
  "faithfulness": 0.0,
  "completeness": 0.0,
  "overall": 0.0,
  "pass": true,
  "reasoning": "one sentence explanation",
  "suggestions": ["improvement1", "improvement2"]
}

overall = (groundedness + faithfulness + completeness) / 3
pass = overall >= threshold (provided in the user message)
"""


class CriticAgent(BaseAgent):
    """
    LLM-as-judge evaluator scoring groundedness, faithfulness, and completeness.
    If overall score < cfg.critic_min_score, flags the answer for retry.
    """

    name = "critic"

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatOpenAI(
            model=cfg.openai_model,
            temperature=0,
            api_key=cfg.openai_api_key,
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        answer: str = state.get("answer", "")
        docs: list[dict] = state.get("retrieved_docs", [])

        if not answer:
            self.logger.warning("No answer to evaluate — returning zero scores")
            scores = self._zero_scores()
            return {**state, "critic_scores": scores, "critic_passed": False}

        context_snippet = "\n\n".join(
            f"[{d['id']}]: {d['document'][:300]}" for d in docs[:5]
        )

        user_msg = f"""Query: {query}

Context used:
{context_snippet or "None"}

Answer to evaluate:
{answer}

Minimum passing threshold: {cfg.critic_min_score}"""

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        response = self._llm.invoke(messages)
        raw = response.content.strip()

        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("Critic returned non-JSON; defaulting to zero scores")
            scores = self._zero_scores()

        # Recompute overall in case LLM drifted
        for dim in ("groundedness", "faithfulness", "completeness"):
            scores.setdefault(dim, 0.0)
        scores["overall"] = round(
            (scores["groundedness"] + scores["faithfulness"] + scores["completeness"]) / 3, 3
        )
        scores["pass"] = scores["overall"] >= cfg.critic_min_score

        self.logger.info(
            "Critic scores — G=%.2f F=%.2f C=%.2f overall=%.2f pass=%s",
            scores["groundedness"], scores["faithfulness"],
            scores["completeness"], scores["overall"], scores["pass"],
        )
        return {**state, "critic_scores": scores, "critic_passed": scores["pass"]}

    @staticmethod
    def _zero_scores() -> dict:
        return {
            "groundedness": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "overall": 0.0,
            "pass": False,
            "reasoning": "Evaluation failed",
            "suggestions": [],
        }
