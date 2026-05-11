# tests/test_agents.py — Unit tests for Helios agent layer
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
from unittest.mock import patch


# ── Executor tests (no LLM needed) ────────────────────────────────────────────

class TestExecutorAgent:

    def test_safe_math_code(self):
        from agents.executor import ExecutorAgent
        agent = ExecutorAgent()
        state = {
            "query": "compute 2+2",
            "plan": {"requires_code": True},
            "code_to_run": "print(2 + 2)",
        }
        result = agent.run(state)
        assert result["execution_result"]["success"] is True
        assert "4" in result["execution_result"]["stdout"]

    def test_forbidden_import_rejected(self):
        from agents.executor import ExecutorAgent
        agent = ExecutorAgent()
        state = {
            "query": "open a file",
            "plan": {"requires_code": True},
            "code_to_run": "import os\nprint(os.listdir('/'))",
        }
        result = agent.run(state)
        assert result["execution_result"]["success"] is False
        assert "import os" in result["execution_result"]["error"]

    def test_timeout_enforcement(self):
        from agents.executor import ExecutorAgent
        agent = ExecutorAgent()
        state = {
            "query": "infinite loop",
            "plan": {"requires_code": True},
            "code_to_run": "while True: pass",
        }
        with patch("config.cfg.executor_timeout_seconds", 1):
            result = agent.run(state)
        assert result["execution_result"]["success"] is False
        assert "timed out" in result["execution_result"]["error"].lower()

    def test_skips_when_not_required(self):
        from agents.executor import ExecutorAgent
        agent = ExecutorAgent()
        state = {"query": "hello", "plan": {"requires_code": False}}
        result = agent.run(state)
        assert result["execution_result"] is None

    def test_syntax_error_caught(self):
        from agents.executor import ExecutorAgent
        agent = ExecutorAgent()
        state = {
            "query": "bad code",
            "plan": {"requires_code": True},
            "code_to_run": "def broken(:\n  pass",
        }
        result = agent.run(state)
        assert result["execution_result"]["success"] is False


# ── Scorer tests ──────────────────────────────────────────────────────────────

class TestScorers:

    def test_full_keyword_hit(self):
        from eval.scorers import score_keyword_coverage
        hit_rate, hallucination = score_keyword_coverage(
            "CARLE uses run-length encoding with triplets",
            ["run-length", "triplet"],
            [],
        )
        assert hit_rate == 1.0
        assert hallucination is False

    def test_partial_keyword_hit(self):
        from eval.scorers import score_keyword_coverage
        hit_rate, _ = score_keyword_coverage(
            "CARLE uses run-length encoding",
            ["run-length", "triplet"],
            [],
        )
        assert hit_rate == 0.5

    def test_hallucination_detected(self):
        from eval.scorers import score_keyword_coverage
        _, hallucination = score_keyword_coverage(
            "CARLE uses pixel reconstruction",
            [],
            ["pixel reconstruction"],
        )
        assert hallucination is True

    def test_empty_keywords_max_score(self):
        from eval.scorers import score_keyword_coverage
        hit_rate, hallucination = score_keyword_coverage("anything", [], [])
        assert hit_rate == 1.0
        assert hallucination is False

    def test_batch_score_pass_rate(self):
        from eval.scorers import batch_score, DimensionScore
        results = [
            {"score": DimensionScore(0.8, 0.9, 0.7, 0.8, True, 1.0, False), "latency_ms": 100},
            {"score": DimensionScore(0.3, 0.4, 0.3, 0.33, False, 0.5, False), "latency_ms": 200},
        ]
        stats = batch_score(results)
        assert stats["pass_rate"] == 0.5
        assert stats["n"] == 2


# ── BM25 tests ────────────────────────────────────────────────────────────────

class TestBM25Index:

    def test_add_and_search(self):
        from retrieval.bm25_search import BM25Index
        idx = BM25Index()
        idx.add("doc1", "CARLE is a compression algorithm for semantic maps")
        idx.add("doc2", "PyTorch is a deep learning framework")
        idx.add("doc3", "Attention mechanisms in neural networks process sequences")
        results = idx.search("semantic compression", top_k=2)
        assert len(results) > 0
        assert results[0]["id"] == "doc1"

    def test_remove(self):
        from retrieval.bm25_search import BM25Index
        idx = BM25Index()
        idx.add("doc1", "remove this document")
        idx.remove("doc1")
        results = idx.search("remove", top_k=1)
        assert len(results) == 0

    def test_empty_index_returns_empty(self):
        from retrieval.bm25_search import BM25Index
        idx = BM25Index()
        results = idx.search("anything", top_k=5)
        assert results == []


# ── Conversation memory tests ─────────────────────────────────────────────────

class TestConversationMemory:

    def test_history_included_in_synthesizer_messages(self):
        from unittest.mock import MagicMock, patch
        from agents.synthesizer import SynthesizerAgent
        from langchain_core.messages import HumanMessage, AIMessage

        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        agent.logger = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "Answer text\n<<<FOLLOW_UPS>>>\n1. Q1\n2. Q2"

        with patch.object(agent, '_llm', create=True) as mock_llm:
            mock_llm.invoke.return_value = mock_response
            state = {
                "query": "tell me more",
                "retrieved_docs": [],
                "web_sources": [],
                "execution_result": None,
                "conversation_history": [
                    {"role": "user", "content": "What is RAG?"},
                    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
                ],
            }
            agent._run(state)

        called_messages = mock_llm.invoke.call_args[0][0]
        # Should have system + 2 history msgs + current user msg = 4 total
        assert len(called_messages) == 4
        assert isinstance(called_messages[1], HumanMessage)
        assert isinstance(called_messages[2], AIMessage)
        assert "RAG" in called_messages[1].content

    def test_empty_history_works(self):
        from unittest.mock import MagicMock, patch
        from agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        agent.logger = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Answer\n<<<FOLLOW_UPS>>>\n1. Q1\n2. Q2"

        with patch.object(agent, '_llm', create=True) as mock_llm:
            mock_llm.invoke.return_value = mock_response
            state = {"query": "hello", "retrieved_docs": [], "web_sources": [],
                     "execution_result": None, "conversation_history": []}
            agent._run(state)

        called_messages = mock_llm.invoke.call_args[0][0]
        # system + current user msg = 2
        assert len(called_messages) == 2

    def test_planner_appends_context_note(self):
        from unittest.mock import MagicMock, patch
        from agents.planner import PlannerAgent
        import json

        agent = PlannerAgent.__new__(PlannerAgent)
        agent.logger = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "query_type": "factual", "subtasks": [{"id": 1, "type": "retrieve", "description": "x"}],
            "requires_retrieval": True, "requires_code": False,
        })

        with patch.object(agent, '_llm', create=True) as mock_llm:
            mock_llm.invoke.return_value = mock_response
            state = {
                "query": "tell me more",
                "conversation_history": [{"role": "user", "content": "What is Helios?"}],
            }
            agent._run(state)

        called_content = mock_llm.invoke.call_args[0][0][1].content
        assert "Helios" in called_content


# ── Per-token streaming tests ─────────────────────────────────────────────────

class TestSynthesizerStreaming:

    def test_token_callback_called_for_each_chunk(self):
        from unittest.mock import MagicMock, patch
        from agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        agent.logger = MagicMock()

        chunks = [MagicMock(content="Hello"), MagicMock(content=" world"),
                  MagicMock(content="<<<FOLLOW_UPS>>>\n1. Q1\n2. Q2")]

        tokens_received: list[str] = []

        with patch.object(agent, '_llm', create=True) as mock_llm:
            mock_llm.stream.return_value = iter(chunks)
            state = {
                "query": "hi",
                "retrieved_docs": [], "web_sources": [],
                "execution_result": None, "conversation_history": [],
                "_token_callback": tokens_received.append,
            }
            result = agent._run(state)

        assert tokens_received == ["Hello", " world", "<<<FOLLOW_UPS>>>\n1. Q1\n2. Q2"]
        assert result["answer"] == "Hello world"

    def test_no_callback_uses_invoke(self):
        from unittest.mock import MagicMock, patch
        from agents.synthesizer import SynthesizerAgent

        agent = SynthesizerAgent.__new__(SynthesizerAgent)
        agent.logger = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Answer\n<<<FOLLOW_UPS>>>\n1. Q1\n2. Q2"

        with patch.object(agent, '_llm', create=True) as mock_llm:
            mock_llm.invoke.return_value = mock_resp
            state = {"query": "hi", "retrieved_docs": [], "web_sources": [],
                     "execution_result": None, "conversation_history": []}
            agent._run(state)

        mock_llm.invoke.assert_called_once()
        mock_llm.stream.assert_not_called()
