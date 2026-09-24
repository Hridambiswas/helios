# agents/verifier.py — Helios Gemini-backed cross-verifier
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent

logger = logging.getLogger("helios.agents.verifier")

_SYSTEM_PROMPT = """\
You are the Verifier agent in Helios — a Gemini-backed second-opinion judge.
The Groq synthesizer has already produced an answer. Your job is an INDEPENDENT
cross-check: read the query, the retrieved context, and the answer, then decide
whether the answer holds up on its own merits.

━━━ SCORING RUBRIC (each dimension: 0.00 – 1.00) ━━━

GROUNDEDNESS — are the claims supported by the retrieved context?
  1.0 → every material claim traces cleanly to the context
  0.5 → some claims are unsupported or lean on outside knowledge
  0.0 → answer is fabricated or contradicted by the context

FAITHFULNESS — does the answer accurately reflect what the context says?
  1.0 → no distortion, no citation errors, no reversed meanings
  0.5 → paraphrase drifts on secondary points
  0.0 → key facts are misrepresented

AGREEMENT — would an independent reader reach the same conclusion?
  1.0 → answer is one you would also give from the same context
  0.5 → you would phrase it differently or add/remove qualifiers
  0.0 → you would give a materially different answer

━━━ OUTPUT FORMAT ━━━
Output ONLY valid JSON — no markdown fences, no prose before or after:
{
  "groundedness": 0.00,
  "faithfulness": 0.00,
  "agreement": 0.00,
  "overall": 0.00,
  "pass": true,
  "reasoning": "one sentence: the strongest reason for the score",
  "flags": ["short tag per red flag, if any"]
}

Compute overall = round((groundedness + faithfulness + agreement) / 3, 3).
Set pass = true iff overall >= the threshold in the user message.
"""


class VerifierAgent(BaseAgent):
    """Gemini-backed second-opinion judge that cross-checks the synthesizer."""

    name = "verifier"

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatGoogleGenerativeAI(
            model=cfg.gemini_model,
            temperature=0,
            google_api_key=cfg.gemini_api_key.get_secret_value(),
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        answer: str = state.get("answer", "")
        docs: list[dict] = state.get("retrieved_docs", [])
