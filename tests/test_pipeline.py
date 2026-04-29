# tests/test_pipeline.py — Integration smoke tests for the LangGraph pipeline
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
from unittest.mock import patch


def _mock_planner_output(state):
    return {**state, "plan": {
        "query_type": "factual",
        "subtasks": [{"id": 1, "type": "retrieve", "description": "fetch docs"}],
        "requires_retrieval": True,
        "requires_code": False,
    }}


def _mock_retriever_output(state):
    return {**state, "retrieved_docs": [
        {"id": "doc1", "document": "CARLE compresses semantic maps", "metadata": {}, "score": 0.9, "source": "dense"}
    ]}


def _mock_synthesizer_output(state):
    return {**state, "answer": "CARLE is a lossless semantic compression method.", "cited_doc_ids": ["doc1"]}


def _mock_critic_output(state):
    return {**state, "critic_scores": {
        "groundedness": 0.9, "faithfulness": 0.9, "completeness": 0.8,
        "overall": 0.87, "pass": True, "reasoning": "Good answer", "suggestions": [],
    }, "critic_passed": True}


class TestPipelineRouting:

    def test_full_pipeline_happy_path(self):
        with (
            patch("agents.planner.PlannerAgent._run", side_effect=_mock_planner_output),
            patch("agents.retriever.RetrieverAgent._run", side_effect=_mock_retriever_output),
            patch("agents.synthesizer.SynthesizerAgent._run", side_effect=_mock_synthesizer_output),
            patch("agents.critic.CriticAgent._run", side_effect=_mock_critic_output),
        ):
            from pipeline.run import run_pipeline
            result = run_pipeline("What is CARLE?")
            assert result["answer"] == "CARLE is a lossless semantic compression method."
            assert result["critic_passed"] is True
            assert len(result["retrieved_docs"]) == 1

    def test_pipeline_returns_error_key_on_crash(self):
        with patch("agents.planner.PlannerAgent._run", side_effect=RuntimeError("LLM timeout")):
            from pipeline.run import run_pipeline
            result = run_pipeline("test query")
            assert result.get("error") is not None

    def test_code_only_routing_skips_retriever(self):
        called = {"retriever": False}

        def mock_planner(state):
            return {**state, "plan": {
                "query_type": "code", "subtasks": [],
                "requires_retrieval": False, "requires_code": True,
            }}

        def mock_retriever(state):
            called["retriever"] = True
            return state

        def mock_executor(state):
            return {**state, "execution_result": {"stdout": "4", "stderr": "", "error": None, "success": True}}

        with (
            patch("agents.planner.PlannerAgent._run", side_effect=mock_planner),
            patch("agents.retriever.RetrieverAgent._run", side_effect=mock_retriever),
            patch("agents.executor.ExecutorAgent._run", side_effect=mock_executor),
            patch("agents.synthesizer.SynthesizerAgent._run", side_effect=_mock_synthesizer_output),
            patch("agents.critic.CriticAgent._run", side_effect=_mock_critic_output),
        ):
            from pipeline.run import run_pipeline
            run_pipeline("compute 2+2", code_to_run="print(2+2)")
            assert called["retriever"] is False
