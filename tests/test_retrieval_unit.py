"""Unit tests for retrieval utilities."""
import pytest
from agents.retriever import _reciprocal_rank_fusion, _deduplicate


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self):
        hits = [
            {"id": "a", "score": 0.9, "document": "doc a"},
            {"id": "b", "score": 0.7, "document": "doc b"},
        ]
        result = _reciprocal_rank_fusion([hits])
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"

    def test_merges_two_lists(self):
        list1 = [{"id": "a", "score": 0.9, "document": "x"}, {"id": "b", "score": 0.5, "document": "y"}]
        list2 = [{"id": "b", "score": 0.8, "document": "y"}, {"id": "c", "score": 0.6, "document": "z"}]
        result = _reciprocal_rank_fusion([list1, list2])
        ids = [r["id"] for r in result]
        # b appears in both lists, so it should rank high
        assert "b" in ids[:2]

    def test_empty_lists_returns_empty(self):
        assert _reciprocal_rank_fusion([]) == []

    def test_rrf_scores_are_positive(self):
        hits = [{"id": "x", "score": 1.0, "document": "test"}]
        result = _reciprocal_rank_fusion([hits])
        assert all(r["score"] > 0 for r in result)


class TestDeduplicate:
    def test_removes_duplicate_ids(self):
        hits = [
            {"id": "a", "score": 0.5, "document": "first"},
            {"id": "a", "score": 0.9, "document": "second"},
            {"id": "b", "score": 0.3, "document": "third"},
        ]
        result = _deduplicate(hits)
        assert len(result) == 2
        a_result = next(r for r in result if r["id"] == "a")
        assert a_result["score"] == 0.9  # keeps highest score

    def test_preserves_order_by_score(self):
        hits = [
            {"id": "c", "score": 0.3, "document": "low"},
            {"id": "a", "score": 0.9, "document": "high"},
            {"id": "b", "score": 0.6, "document": "mid"},
        ]
        result = _deduplicate(hits)
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"
        assert result[2]["id"] == "c"
