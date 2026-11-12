# agents/scorer.py — Helios independent scorer
# Author: Hridam Biswas | Project: Helios
#
# Scores the final answer written by the Judge. Runs on the SKEPTIC model
# (Qwen) so the author (judge) and grader are different roles AND different
# prompts — the final scores are NOT self-graded.
#
# Three dimensions, each 0.0–1.0:
#   groundedness : is every claim backed by [W]/[D] in the context,
#                  or is it the model's own knowledge?
#   faithfulness : do the cited sources actually say what the answer says?
#   completeness : does the answer fully address the question?

from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import cfg
from agents.base import BaseAgent


_SYSTEM_PROMPT = """\
You are the SCORER in Helios's debate pipeline. A JUDGE wrote the final
answer the user will see; you were NOT the writer. Your job: grade the
answer honestly, as a tough but fair reviewer.

Three dimensions, each 0.00–1.00 (smaller is worse; be strict):

GROUNDEDNESS
  1.0  every material claim has an explicit [W#] or [D#] citation,
       and the cited source is actually relevant to the claim.
  0.7  most claims are cited; some lean on general knowledge.
  0.4  only a handful are cited; large sections are unsupported.
  0.1  essentially no citations even though sources were available.
  If NO sources were provided at all, treat groundedness as 1.0 (n/a).

FAITHFULNESS
  1.0  cited sources genuinely support what the answer says.
  0.7  minor paraphrase drift on secondary points.
  0.4  one or more cited sources do NOT say what the answer claims.
  0.1  major misrepresentation of the cited sources.
  If no sources were cited, score as 1.0 (n/a).

COMPLETENESS
  1.0  the question is fully answered — every explicit sub-question too.
  0.7  the main question is answered; a minor aspect is missed.
  0.4  a partial answer; important aspects skipped.
  0.1  the answer is off-topic, hedged, or refuses to answer a reasonable question.

Bias to be HONEST. A vague answer, an "I don't know" fluff, or a
near-empty answer must score LOW (<=0.3 completeness). Do NOT be
charitable just because the author tried.

Output VALID JSON only — no markdown fences, no prose before or after:
{
  "groundedness": 0.00,
  "faithfulness": 0.00,
  "completeness": 0.00,
  "reasoning": "<one sentence, 30 words max, pointing at the biggest flaw>"
}
The pipeline will compute overall = round((g+f+c)/3, 3).
"""


class ScorerAgent(BaseAgent):
    """Independent grader. Different model + prompt from the Judge."""

    name = "scorer"

    def __init__(self) -> None:
        super().__init__()
        self._model = cfg.scorer_model
        self._llm = ChatGroq(
            model=self._model,
            temperature=0.0,
            api_key=cfg.groq_api_key,
        )
        self.logger.info("scorer initialised: provider=groq model=%s", self._model)

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        query: str = state["query"]
        answer: str = state.get("answer", "") or ""
        web: list[dict] = state.get("web_sources", []) or []
        docs: list[dict] = state.get("retrieved_docs", []) or []
        cited_web: list[int] = state.get("cited_web_indices", []) or []
        cited_docs: list[str] = state.get("cited_doc_ids", []) or []

        if not answer:
            scores = self._zero()
            scores["reasoning"] = "Empty answer."
            self.logger.info("scorer: empty answer -> zero scores")
            return {**state, "final_scores": scores}

        web_block = "\n\n".join(
            f"[W{i}] {w.get('title','')} — {w.get('snippet','')[:300]}"
            for i, w in enumerate(web[:cfg.debate_web_top_k], 1)
        ) or "(no web results)"
        doc_block = "\n\n".join(
            f"[D{i}] {d.get('document','')[:300]}"
            for i, d in enumerate(docs[:5], 1)
        ) or "(no local docs)"

        user_msg = (
            f"Question: {query}\n\n"
            f"Final answer:\n{answer}\n\n"
            f"Web sources available:\n{web_block}\n\n"
            f"Local docs available:\n{doc_block}\n\n"
            f"Sources actually cited in the answer: "
            f"W={cited_web or 'none'}, D={cited_docs or 'none'}\n"
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
            raw = self._invoke_with_timeout(messages, cfg.scorer_timeout_seconds)
        except Exception as exc:
            self.logger.warning("scorer failed (%s); using zero scores", exc)
            scores = self._zero()
            scores["reasoning"] = f"scorer unavailable: {exc}"
            return {**state, "final_scores": scores}

        parsed = self._parse(raw) or {}
        scores = self._sanitize(parsed)
        self.logger.info(
            "scorer: g=%.2f f=%.2f c=%.2f overall=%.2f (model=%s)",
            scores["groundedness"], scores["faithfulness"],
            scores["completeness"], scores["overall"], self._model,
        )
        return {**state, "final_scores": scores}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _invoke_with_timeout(self, messages, timeout_seconds: float) -> str:
        def _call() -> str:
            resp = self._llm.invoke(messages, timeout=timeout_seconds)
            return (
                resp.content if isinstance(resp.content, str) else str(resp.content)
            ).strip()

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_call)
            try:
                return fut.result(timeout=timeout_seconds)
            except FuturesTimeout as exc:
                fut.cancel()
                raise TimeoutError(f"scorer exceeded {timeout_seconds}s") from exc

    @staticmethod
    def _parse(raw: str) -> dict | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            stripped = raw.strip().strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].lstrip()
            # Fish out the first {...} block if the LLM wrapped it in prose.
            m = re.search(r"\{.*\}", stripped, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _sanitize(raw: dict) -> dict:
        def _f(k: str) -> float:
            try:
                v = float(raw.get(k, 0.0))
                return max(0.0, min(1.0, v))
            except (TypeError, ValueError):
                return 0.0
        g = _f("groundedness")
        f = _f("faithfulness")
        c = _f("completeness")
        return {
            "groundedness": g,
            "faithfulness": f,
            "completeness": c,
            "overall": round((g + f + c) / 3, 3),
            "reasoning": (raw.get("reasoning") or "")[:280],
            "scorer_model": cfg.scorer_model,
        }

    @staticmethod
    def _zero() -> dict:
        return {
            "groundedness": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "overall": 0.0,
            "reasoning": "",
            "scorer_model": cfg.scorer_model,
        }
