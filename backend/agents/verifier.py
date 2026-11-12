# agents/verifier.py — Helios Gemini-backed cross-verifier
# Author: Hridam Biswas | Project: Helios
#
# The Verifier is the sixth agent in the LangGraph pipeline. It runs after
# the Critic and independently re-reads the Synthesizer's answer using a
# different provider (Google Gemini) to catch systemic Groq failure modes.
# Toggleable via VERIFIER_ENABLED; safe to disable in local dev.

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

        if not answer:
            self.logger.warning("No answer to verify — returning zero scores")
            scores = self._zero_scores()
            return {**state, "verifier_scores": scores, "verifier_passed": False}

        context_snippet = "\n\n".join(
            f"[{d['id']}]: {d['document'][:300]}" for d in docs[:5]
        )
        user_msg = (
            f"Query: {query}\n\n"
            f"Retrieved context:\n{context_snippet or 'None'}\n\n"
            f"Synthesizer answer:\n{answer}\n\n"
            f"Minimum passing threshold: {cfg.verifier_min_score}"
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        # Hard outer timeout. The Gemini client retries with exponential
        # backoff internally — up to ~2 minutes before it surfaces the
        # error — which would fail the whole pipeline. Wrap the call in a
        # thread + timeout so the pipeline can cleanly report "unverified"
        # on slow or flaky upstream behaviour.
        try:
            raw = self._invoke_with_hard_timeout(
                messages, cfg.verifier_timeout_seconds
            )
        except Exception as exc:
            self.logger.warning(
                "Verifier timed out or failed in %.1fs: %s",
                cfg.verifier_timeout_seconds, exc,
            )
            scores = self._zero_scores()
            scores["flags"] = ["verifier_timeout"]
            return {**state, "verifier_scores": scores, "verifier_passed": False}

        try:
            scores = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("Verifier returned non-JSON; defaulting to zero scores")
            scores = self._zero_scores()

        for dim in ("groundedness", "faithfulness", "agreement"):
            scores.setdefault(dim, 0.0)
        scores["overall"] = round(
            (scores["groundedness"] + scores["faithfulness"] + scores["agreement"]) / 3, 3
        )
        scores["pass"] = scores["overall"] >= cfg.verifier_min_score
        scores.setdefault("reasoning", "")
        scores.setdefault("flags", [])

        from observability.metrics import verifier_score_histogram, verifier_pass_counter
        for dim in ("groundedness", "faithfulness", "agreement", "overall"):
            verifier_score_histogram.labels(dimension=dim).observe(scores[dim])
        verifier_pass_counter.labels(result="pass" if scores["pass"] else "fail").inc()

        self.logger.info(
            "Verifier scores — G=%.2f F=%.2f A=%.2f overall=%.2f pass=%s",
            scores["groundedness"], scores["faithfulness"],
            scores["agreement"], scores["overall"], scores["pass"],
        )
        return {**state, "verifier_scores": scores, "verifier_passed": scores["pass"]}

    def _invoke_with_hard_timeout(self, messages, timeout_seconds: float) -> str:
        """Wrap the LLM call so a flaky upstream can never block longer than N s."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        def _call() -> str:
            response = self._llm.invoke(messages, timeout=timeout_seconds)
            return (
                response.content if isinstance(response.content, str)
                else str(response.content)
            ).strip()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeout as exc:
                future.cancel()
                raise TimeoutError(
                    f"Verifier exceeded {timeout_seconds}s"
                ) from exc

    @staticmethod
    def _zero_scores() -> dict:
        return {
            "groundedness": 0.0,
            "faithfulness": 0.0,
            "agreement": 0.0,
            "overall": 0.0,
            "pass": False,
            "reasoning": "Verification failed",
            "flags": ["verifier_error"],
        }
