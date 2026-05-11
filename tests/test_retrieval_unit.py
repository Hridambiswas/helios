"""Unit tests for hybrid retrieval helpers — RRF and deduplication."""
import pytest
from agents.retriever import _reciprocal_rank_fusion, _deduplicate


def _hit(doc_id: str, score: float = 1.0) -> dict:
    return {"id": doc_id, "score": score, "text": "x"}


class TestRRF:
    def test_single_list_preserves_order(self):
        hits = [_hit("a"), _hit("b"), _hit("c")]
        merged = _reciprocal_rank_fusion([hits])
        ids = [h["id"] for h in merged]
        assert ids == ["a", "b", "c"]

    def test_two_lists_boost_shared_docs(self):
        dense = [_hit("a"), _hit("b"), _hit("c")]
        bm25  = [_hit("b"), _hit("c"), _hit("d")]
        merged = _reciprocal_rank_fusion([dense, bm25])
        ids = [h["id"] for h in merged]
        # b and c appear in both lists → should rank above a and d
        assert ids.index("b") < ids.index("a")
        assert ids.index("c") < ids.index("d")

    def test_empty_lists_ignored(self):
        hits = [_hit("x"), _hit("y")]
        merged = _reciprocal_rank_fusion([hits, []])
        assert len(merged) == 2

    def test_all_empty_returns_empty(self):
        assert _reciprocal_rank_fusion([[], []]) == []

    def test_rrf_score_is_sum_of_reciprocal_ranks(self):
        # doc "a" is rank 1 in both lists → score = 1/61 + 1/61 = 2/61
        merged = _reciprocal_rank_fusion([[_hit("a")], [_hit("a")]])
        assert len(merged) == 1
        expected = 2.0 / (60 + 1)
        assert abs(merged[0]["score"] - expected) < 1e-9


class TestDeduplicate:
    def test_keeps_highest_score(self):
        hits = [_hit("a", 0.5), _hit("a", 0.9), _hit("b", 0.7)]
        deduped = _deduplicate(hits)
        a_hit = next(h for h in deduped if h["id"] == "a")
        assert a_hit["score"] == 0.9

    def test_no_duplicates_unchanged(self):
        hits = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
        assert len(_deduplicate(hits)) == 3

    def test_result_sorted_by_score_desc(self):
        hits = [_hit("c", 0.3), _hit("a", 0.9), _hit("b", 0.6)]
        deduped = _deduplicate(hits)
        scores = [h["score"] for h in deduped]
        assert scores == sorted(scores, reverse=True)
