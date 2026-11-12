# tests/test_debate_pipeline.py — unit tests for the 013 debate pipeline.
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def state_with_draft():
    """Base state after planner + researcher + answerer have run."""
    return {
        "query": "What is CARLE?",
        "plan": {
            "query_type": "research",
            "subtasks": [],
            "requires_retrieval": True,
            "requires_code": False,
        },
        "retrieved_docs": [
            {"id": "d1", "document": "CARLE is lossless semantic compression.",
             "metadata": {}, "score": 0.9, "source": "dense"},
        ],
        "web_sources": [
            {"title": "CARLE paper", "url": "https://example.com/carle",
             "snippet": "CARLE compresses meaning without loss."},
        ],
        "draft_answer": "CARLE is a lossless semantic compression method.",
    }


# ── Answerer: empty-fallback path ─────────────────────────────────────────────

class TestAnswerer:

    def test_both_llms_empty_yields_empty_draft(self):
        """Primary + fallback both return empty → empty draft (not a crash)."""
        from agents.answerer import AnswererAgent

        with patch("agents.answerer.ChatGoogleGenerativeAI"), patch("agents.answerer.ChatGroq"):
            agent = AnswererAgent()
            agent._llm = MagicMock()
            agent._fallback = MagicMock()
            agent._llm.invoke.return_value.content = ""
            agent._fallback.invoke.return_value.content = ""

            out = agent._run({"query": "hi", "retrieved_docs": [], "web_sources": []})
            assert out["draft_answer"] == ""

    def test_fallback_runs_when_primary_fails(self):
        """Gemini raises → Groq fallback supplies the draft."""
        from agents.answerer import AnswererAgent

        with patch("agents.answerer.ChatGoogleGenerativeAI"), patch("agents.answerer.ChatGroq"):
            agent = AnswererAgent()
            agent._llm = MagicMock()
            agent._fallback = MagicMock()
            agent._llm.invoke.side_effect = RuntimeError("gemini overloaded")
            agent._fallback.invoke.return_value.content = "Fallback draft."

            out = agent._run({"query": "hi", "retrieved_docs": [], "web_sources": []})
            assert out["draft_answer"] == "Fallback draft."
            assert out["answerer_provider"] == "groq"


# ── Debate critics: parallel, JSON output ─────────────────────────────────────

class TestDebateCritics:

    def _patch_groq(self, raw_json: str):
        """Patch ChatGroq used by the debate critics."""
        from unittest.mock import patch
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = raw_json
        return patch("agents.debate_critics.ChatGroq", return_value=mock_llm)

    def test_advocate_parses_valid_json(self, state_with_draft):
        from agents.debate_critics import AdvocateCriticAgent
        payload = {
            "verdict": "mostly_correct",
            "supported_claims": [{"claim": "CARLE is lossless", "support": "[W1]"}],
            "strengthen": [],
            "summary": "Draft holds up.",
        }
        with self._patch_groq(json.dumps(payload)):
            agent = AdvocateCriticAgent()
            out = agent._run(state_with_draft)
            cr = out["advocate_critique"]
            assert cr["verdict"] == "mostly_correct"
            assert cr["provider"] == "groq"
            assert cr["model"]  # non-empty

    def test_skeptic_empty_draft_yields_empty_payload(self):
        from agents.debate_critics import SkepticCriticAgent
        with self._patch_groq("ignored"):
            agent = SkepticCriticAgent()
            out = agent._run({"query": "x", "draft_answer": ""})
            cr = out["skeptic_critique"]
            # No draft means no debate — empty payload.
            assert cr["problems"] == []
            assert cr["provider"] == "groq"


# ── Judge: always produces non-empty answer ──────────────────────────────────

class TestJudge:

    def test_judge_writes_final_answer(self, state_with_draft):
        from agents.judge import JudgeAgent
        # Draft has two [W#] citations so the judge doesn't trigger
        # the auto-citation repair path.
        state_with_draft_two_webs = {
            **state_with_draft,
            "web_sources": [
                {"title": "CARLE paper", "url": "https://example.com/carle", "snippet": "…"},
                {"title": "Compression survey", "url": "https://example.com/survey", "snippet": "…"},
            ],
        }
        judge_output = {
            "final_answer": (
                "CARLE is a lossless semantic compression method [W1][W2].\n\n"
                "**Sources:**\n- [W1] CARLE paper — https://example.com/carle\n"
                "- [W2] Compression survey — https://example.com/survey"
            ),
            "verdict": "confirmed draft",
            "cited_web_indices": [1, 2],
            "cited_doc_indices": [],
        }
        with patch("agents.judge.ChatGroq") as mock_groq:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = json.dumps(judge_output)
            mock_groq.return_value = mock_llm

            agent = JudgeAgent()
            out = agent._run({
                **state_with_draft_two_webs,
                "advocate_critique": {"summary": "ok"},
                "skeptic_critique": {"summary": "ok"},
            })

        assert out["answer"].startswith("CARLE is a lossless semantic compression")
        assert out["cited_web_indices"] == [1, 2]
        assert {s["url"] for s in out["judge_sources"]} == {
            "https://example.com/carle", "https://example.com/survey",
        }
        assert out["judge_verdict"] == "confirmed draft"
        # Judge no longer self-grades — scorer runs next.
        assert "final_scores" not in out or out.get("final_scores") is None
        assert out.get("judge_fallback") is not True

    def test_judge_auto_repairs_missing_citations(self, state_with_draft):
        """If web sources exist but judge cites <2, auto-cite the top results."""
        from agents.judge import JudgeAgent
        state = {
            **state_with_draft,
            "web_sources": [
                {"title": "A", "url": "https://a.test", "snippet": "…"},
                {"title": "B", "url": "https://b.test", "snippet": "…"},
                {"title": "C", "url": "https://c.test", "snippet": "…"},
            ],
        }
        judge_output = {
            "final_answer": "CARLE compresses meaning without loss.",  # no [W]
            "verdict": "confirmed draft",
            "cited_web_indices": [],
            "cited_doc_indices": [],
        }
        with patch("agents.judge.ChatGroq") as mock_groq:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = json.dumps(judge_output)
            mock_groq.return_value = mock_llm

            out = JudgeAgent()._run({**state, "advocate_critique": {}, "skeptic_critique": {}})

        assert len(out["cited_web_indices"]) >= 2
        assert "Supporting sources" in out["answer"]

    def test_judge_fallback_when_llm_raises(self, state_with_draft):
        """Pipeline must never return an empty answer — judge fallback keeps the draft."""
        from agents.judge import JudgeAgent

        with patch("agents.judge.ChatGroq") as mock_groq:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = RuntimeError("Groq is down")
            mock_groq.return_value = mock_llm

            agent = JudgeAgent()
            out = agent._run(state_with_draft)

        assert out["answer"] == state_with_draft["draft_answer"]
        assert out["judge_fallback"] is True


class TestScorer:

    def test_scorer_strict_low_for_empty(self):
        from agents.scorer import ScorerAgent
        with patch("agents.scorer.ChatGroq"):
            agent = ScorerAgent()
            out = agent._run({"query": "q", "answer": "", "web_sources": [], "retrieved_docs": []})
        assert out["final_scores"]["overall"] == 0.0

    def test_scorer_parses_json(self):
        from agents.scorer import ScorerAgent
        with patch("agents.scorer.ChatGroq"):
            agent = ScorerAgent()
            agent._llm = MagicMock()
            agent._llm.invoke.return_value.content = json.dumps({
                "groundedness": 0.2,
                "faithfulness": 0.9,
                "completeness": 0.4,
                "reasoning": "Mostly unsupported."
            })
            out = agent._run({
                "query": "q", "answer": "x", "web_sources": [{"title": "t", "url": "u", "snippet": "s"}],
                "retrieved_docs": [], "cited_web_indices": [], "cited_doc_ids": [],
            })
        fs = out["final_scores"]
        assert fs["groundedness"] == 0.2
        assert 0.3 < fs["overall"] < 0.6
        assert "unsupported" in fs["reasoning"].lower()


# ── Verifier: hard timeout path (regression from 012) ─────────────────────────

class TestVerifierTimeout:

    def test_verifier_timeout_returns_zero_scores(self):
        """A slow Gemini must never stall the pipeline longer than N seconds."""
        import os
        os.environ.setdefault("VERIFIER_ENABLED", "true")  # ensure the agent constructs
        from agents.verifier import VerifierAgent
        import time as _time

        with patch("agents.verifier.ChatGoogleGenerativeAI"):
            agent = VerifierAgent()
            # Simulate a LLM.invoke that hangs forever
            def _hang(*_, **__):
                _time.sleep(5)
                return MagicMock(content='{"groundedness":1,"faithfulness":1,"agreement":1,"overall":1}')
            agent._llm = MagicMock()
            agent._llm.invoke.side_effect = _hang

            # Monkey-patch the hard timeout to 0.1s so the test is fast.
            from config import cfg
            orig = cfg.verifier_timeout_seconds
            object.__setattr__(cfg, "verifier_timeout_seconds", 0.1)
            try:
                state = {"query": "q", "answer": "a", "retrieved_docs": []}
                out = agent._run(state)
            finally:
                object.__setattr__(cfg, "verifier_timeout_seconds", orig)

            assert out["verifier_passed"] is False
            flags = out["verifier_scores"].get("flags", [])
            assert any("timeout" in f for f in flags)
