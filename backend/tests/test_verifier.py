# tests/test_verifier.py — VerifierAgent unit tests (Gemini mocked)
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import json
from types import SimpleNamespace
from unittest.mock import patch


def _make_agent():
    # Avoid touching the real Gemini SDK at import time.
    with patch("agents.verifier.ChatGoogleGenerativeAI"):
        from agents.verifier import VerifierAgent
        return VerifierAgent()


def test_zero_scores_when_no_answer():
    agent = _make_agent()
    result = agent.run({"query": "q", "answer": "", "retrieved_docs": []})
    assert result["verifier_passed"] is False
    assert result["verifier_scores"]["overall"] == 0.0


def test_high_scores_pass_threshold():
    agent = _make_agent()
    fake_response = SimpleNamespace(content=json.dumps({
        "groundedness": 0.9,
        "faithfulness": 0.9,
        "agreement": 0.9,
        "reasoning": "solid answer",
        "flags": [],
    }))
    agent._llm.invoke = lambda *_a, **_k: fake_response  # type: ignore[assignment]
    state = {
        "query": "q",
        "answer": "a",
        "retrieved_docs": [{"id": "D1", "document": "supports a"}],
    }
    result = agent.run(state)
    assert result["verifier_passed"] is True
    assert 0.85 < result["verifier_scores"]["overall"] <= 1.0


def test_malformed_json_falls_back_to_zero():
    agent = _make_agent()
    agent._llm.invoke = lambda *_a, **_k: SimpleNamespace(content="not json")  # type: ignore[assignment]
    state = {"query": "q", "answer": "a", "retrieved_docs": []}
    result = agent.run(state)
    assert result["verifier_passed"] is False
    assert result["verifier_scores"]["flags"] == ["verifier_error"]
