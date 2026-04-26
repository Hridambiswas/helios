# tests/conftest.py — Shared pytest fixtures for Helios test suite
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def base_state() -> dict:
    """Minimal pipeline state dict that satisfies all agents."""
    return {
        "query": "What is CARLE?",
        "user_id": "user-123",
        "plan": {
            "query_type": "factual",
            "subtasks": [{"id": 1, "description": "Look up CARLE"}],
            "requires_retrieval": True,
            "requires_code": False,
        },
        "retrieved_docs": [
            {
                "id": "doc1::chunk::0",
                "document": "CARLE is a compression algorithm.",
                "metadata": {"filename": "carle.txt", "doc_id": "doc1"},
                "score": 0.95,
                "source": "dense",
            }
        ],
        "answer": "CARLE is a class-aware run-length encoding algorithm.",
        "execution_result": None,
        "critic_scores": {
            "groundedness": 0.9,
            "faithfulness": 0.85,
            "completeness": 0.8,
            "overall": 0.85,
        },
        "critic_passed": True,
    }


@pytest.fixture
def mock_openai_response():
    """Mock for openai chat completions used by Planner/Synthesizer/Critic."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = '{"query_type":"factual","subtasks":[],"requires_retrieval":true,"requires_code":false}'
    return mock


@pytest.fixture
def mock_db_session():
    """Async mock SQLAlchemy session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session
