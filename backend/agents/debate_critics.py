# agents/debate_critics.py — Helios debate critics (advocate + skeptic)
# Author: Hridam Biswas | Project: Helios
#
# Two Groq-backed critics that argue in parallel about the Answerer's draft:
#   AdvocateCriticAgent  — "the draft is good because …"  (model A)
#   SkepticCriticAgent   — "the draft is wrong because …" (model B, DIFFERENT family)
#
# Each emits structured JSON that the Judge consumes. Neither writes the final
# answer — the user never sees these outputs directly, they only feed the judge
# and the UI's "Debate" expander.

from __future__ import annotations
import json
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent
from agents._groq_call import invoke_groq_with_retry


_ADVOCATE_SYSTEM = """\
You are the ADVOCATE critic in Helios's debate pipeline. A draft answer is on
trial. Your job: argue why the draft is CORRECT. For each material claim,
state the claim briefly and cite the source that supports it (web [W#] or
local [D#]). Flag claims that look well-grounded but could be strengthened.

Output VALID JSON only — no markdown, no prose before or after:
{
  "verdict": "mostly_correct" | "partially_correct" | "weak",
  "supported_claims": [
    {"claim": "<short>", "support": "[W2]" | "[D1]" | "common knowledge"}
  ],
  "strengthen": ["<short suggestion>", "..."],
  "summary": "<one sentence, no longer than 30 words>"
}
"""


_SKEPTIC_SYSTEM = """\
You are the SKEPTIC critic in Helios's debate pipeline. A draft answer is on
trial. Your job: argue why the draft is WRONG, UNSUPPORTED, OUTDATED or
INCOMPLETE. For each problem, state what's wrong and either tie it to a
contradicting source ([W#]/[D#]) or mark it "unsupported". Be surgical, not
pedantic — don't invent flaws.

Output VALID JSON only — no markdown, no prose before or after:
{
  "verdict": "has_issues" | "minor_issues" | "looks_clean",
  "problems": [
    {"issue": "<short>", "evidence": "[W3]" | "unsupported" | "outdated",
     "severity": "high" | "medium" | "low"}
  ],
  "missing": ["<claim the draft should have made but didn't>", "..."],
  "summary": "<one sentence, no longer than 30 words>"
}
"""


class _DebateCriticBase(BaseAgent):
    """Shared Groq-backed debate critic. Subclasses pick the model + prompt."""

    #: Overridden by subclasses
    _system_prompt: str = ""
    _model_attr: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._model = getattr(cfg, self._model_attr)
        self._llm = ChatGroq(
            model=self._model,
            temperature=0.1,
            api_key=cfg.groq_api_key,
        )
        self.logger.info(
            "%s initialised: provider=groq model=%s", self.name, self._model
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        draft: str = state.get("draft_answer", "") or ""
        docs: list[dict] = state.get("retrieved_docs", []) or []
        web: list[dict] = state.get("web_sources", []) or []

        if not draft:
            self.logger.warning("%s: empty draft — returning empty critique", self.name)
            payload = self._empty_payload()
        else:
            doc_block = "\n\n".join(
                f"[D{i}] {d.get('document','')[:300]}"
                for i, d in enumerate(docs[:5], 1)
            ) or "(none)"
            web_block = "\n\n".join(
                f"[W{i}] {w.get('title','')} — {w.get('snippet','')[:300]}"
                for i, w in enumerate(web[:cfg.debate_web_top_k], 1)
            ) or "(none)"

            user_msg = (
                f"Question: {query}\n\nDraft answer on trial:\n{draft}\n\n"
                f"Local documents:\n{doc_block}\n\n"
                f"Web results:\n{web_block}\n"
            )
            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=user_msg),
            ]
            try:
                raw = invoke_groq_with_retry(
                    self._llm, messages,
                    agent_name=self.name,
                    per_call_timeout=8.0,
                    attempts=2,
                )
                payload = self._parse(raw)
            except Exception as exc:
                self.logger.warning("%s failed after retries: %s", self.name, exc)
                payload = self._empty_payload()

        # Tag the payload with provider/model so the UI can show them.
        payload["provider"] = "groq"
        payload["model"] = self._model

        self.logger.info(
            "%s critique produced: provider=groq model=%s len=%d",
            self.name, self._model, len(json.dumps(payload)),
        )
        return {**state, self._state_key(): payload}

    def _parse(self, raw: str) -> dict:
        """Strict JSON parse with a one-shot repair attempt (strip fences)."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            stripped = raw.strip().lstrip("`").rstrip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                self.logger.warning("%s: non-JSON output, using empty payload", self.name)
                return self._empty_payload()

    def _state_key(self) -> str:
        raise NotImplementedError

    def _empty_payload(self) -> dict:
        raise NotImplementedError


class AdvocateCriticAgent(_DebateCriticBase):
    name = "advocate_critic"
    _system_prompt = _ADVOCATE_SYSTEM
    _model_attr = "advocate_model"

    def _state_key(self) -> str:
        return "advocate_critique"

    def _empty_payload(self) -> dict:
        return {
            "verdict": "weak",
            "supported_claims": [],
            "strengthen": [],
            "summary": "No draft to defend.",
        }


class SkepticCriticAgent(_DebateCriticBase):
    name = "skeptic_critic"
    _system_prompt = _SKEPTIC_SYSTEM
    _model_attr = "skeptic_model"

    def _state_key(self) -> str:
        return "skeptic_critique"

    def _empty_payload(self) -> dict:
        return {
            "verdict": "has_issues",
            "problems": [],
            "missing": [],
            "summary": "No draft to challenge.",
        }
