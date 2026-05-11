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


class TestRetryLoop:

    def test_critic_fail_triggers_second_synthesizer_call(self):
        synthesizer_calls = {"count": 0}

        def mock_synthesizer(state):
            synthesizer_calls["count"] += 1
            return {**state,
                    "answer": f"Answer attempt {synthesizer_calls['count']}",
                    "cited_doc_ids": [],
                    "follow_up_questions": [],
                    "web_sources": [],
                    "retry_count": state.get("retry_count", 0) + 1}

        critic_call = {"count": 0}

        def mock_critic(state):
            critic_call["count"] += 1
            # Fail on first evaluation, pass on second
            passed = critic_call["count"] > 1
            return {**state, "critic_scores": {
                "groundedness": 0.3 if not passed else 0.9,
                "faithfulness": 0.3 if not passed else 0.9,
                "completeness": 0.3 if not passed else 0.9,
                "overall": 0.3 if not passed else 0.9,
                "pass": passed, "reasoning": "test", "suggestions": ["Be more specific"],
            }, "critic_passed": passed}

        with (
            patch("agents.planner.PlannerAgent._run", side_effect=_mock_planner_output),
            patch("agents.retriever.RetrieverAgent._run", side_effect=_mock_retriever_output),
            patch("agents.executor.ExecutorAgent._run", side_effect=lambda s: s),
            patch("agents.synthesizer.SynthesizerAgent._run", side_effect=mock_synthesizer),
            patch("agents.critic.CriticAgent._run", side_effect=mock_critic),
        ):
            from pipeline.run import run_pipeline
            result = run_pipeline("test query")

        assert synthesizer_calls["count"] == 2
        assert critic_call["count"] == 2
        assert result["critic_passed"] is True
        assert result["answer"] == "Answer attempt 2"

    def test_retry_capped_at_max_retries(self):
        """Pipeline should not retry more than _MAX_RETRIES times."""
        synthesizer_calls = {"count": 0}

        def always_fail_critic(state):
            return {**state, "critic_scores": {
                "groundedness": 0.1, "faithfulness": 0.1, "completeness": 0.1,
                "overall": 0.1, "pass": False, "reasoning": "always fail", "suggestions": [],
            }, "critic_passed": False}

        def counting_synthesizer(state):
            synthesizer_calls["count"] += 1
            return {**state, "answer": "answer", "cited_doc_ids": [],
                    "follow_up_questions": [], "web_sources": [],
                    "retry_count": state.get("retry_count", 0) + 1}

        with (
            patch("agents.planner.PlannerAgent._run", side_effect=_mock_planner_output),
            patch("agents.retriever.RetrieverAgent._run", side_effect=_mock_retriever_output),
            patch("agents.executor.ExecutorAgent._run", side_effect=lambda s: s),
            patch("agents.synthesizer.SynthesizerAgent._run", side_effect=counting_synthesizer),
            patch("agents.critic.CriticAgent._run", side_effect=always_fail_critic),
        ):
            from pipeline.run import run_pipeline
            result = run_pipeline("test query")

        from pipeline.run import _MAX_RETRIES
        assert synthesizer_calls["count"] == 1 + _MAX_RETRIES
        assert result["critic_passed"] is False
