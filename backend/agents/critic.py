# agents/critic.py — Helios LLM-as-Judge Critic Agent
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.critic")

_SYSTEM_PROMPT = """\
You are the Critic agent in Helios — an LLM-as-judge evaluator.
Your role: score a generated answer on three independent dimensions.

━━━ SCORING RUBRIC (each dimension: 0.00 – 1.00) ━━━

GROUNDEDNESS — are all claims traceable to the provided context?
  1.0 → every factual claim has explicit support in the retrieved context
  0.7 → most claims supported; one or two lean on general LLM knowledge
  0.4 → significant portions are unsupported or speculative
  0.0 → answer is entirely ungrounded / invented

FAITHFULNESS — does the answer accurately represent what the context says?
  1.0 → no hallucinations, no contradictions, citations correct
  0.7 → minor paraphrasing; meaning preserved
  0.4 → one or more facts distorted or reversed
  0.0 → contradicts or fabricates what the context says

COMPLETENESS — does the answer fully address every part of the query?
  1.0 → every explicit and implicit sub-question answered
  0.7 → main question answered; minor aspects skipped
  0.4 → partial answer; important aspects omitted
  0.0 → query largely left unanswered

━━━ OUTPUT FORMAT ━━━
Output ONLY valid JSON — no markdown fences, no text before or after:
{
  "groundedness": 0.00,
  "faithfulness": 0.00,
  "completeness": 0.00,
  "overall": 0.00,
  "pass": true,
  "reasoning": "concise one-sentence explanation of the dominant weakness",
  "suggestions": ["specific actionable suggestion 1", "specific actionable suggestion 2"]
}

Compute: overall = round((groundedness + faithfulness + completeness) / 3, 3)
Set pass = true if overall >= the threshold supplied in the user message.
"""


class CriticAgent(BaseAgent):
    """
    LLM-as-judge evaluator scoring groundedness, faithfulness, and completeness.
    If overall score < cfg.critic_min_score, flags the answer for retry.
    """

    name = "critic"

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatGroq(
            model=cfg.groq_model,
            temperature=0,
            api_key=cfg.groq_api_key,
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
        response = self._llm.invoke(messages, timeout=30)
        raw = (response.content if isinstance(response.content, str) else str(response.content)).strip()

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

        # Record per-dimension scores in Prometheus
        from observability.metrics import critic_score_histogram, critic_pass_counter
        for dim in ("groundedness", "faithfulness", "completeness", "overall"):
            critic_score_histogram.labels(dimension=dim).observe(scores[dim])
        critic_pass_counter.labels(result="pass" if scores["pass"] else "fail").inc()

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
