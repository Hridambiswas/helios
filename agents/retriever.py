# agents/retriever.py — Helios Hybrid Retriever Agent (Dense + BM25 + Web)
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
from typing import Any

from langchain_community.embeddings import FastEmbedEmbeddings

from config import cfg
from agents.base import BaseAgent
import retrieval.vector_store as vs
import retrieval.bm25_search as bm25
from observability.metrics import retrieval_docs_histogram, retrieval_score_summary

logger = logging.getLogger("helios.agents.retriever")


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    hit_by_id: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            doc_id = hit["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in hit_by_id or hit.get("score", 0) > hit_by_id[doc_id].get("score", 0):
                hit_by_id[doc_id] = hit

    merged = []
    for doc_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = {**hit_by_id[doc_id], "score": rrf_score}
        merged.append(entry)
    return merged


def _deduplicate(hits: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for hit in hits:
        doc_id = hit["id"]
        if doc_id not in seen or hit["score"] > seen[doc_id]["score"]:
            seen[doc_id] = hit
    return sorted(seen.values(), key=lambda h: h["score"], reverse=True)


def _web_search(query: str, max_results: int = 4) -> list[dict]:
    """Search the web via DuckDuckGo. Returns [{title, url, snippet}]."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
            if r.get("href")
        ]
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return []


class RetrieverAgent(BaseAgent):
    """
    Hybrid retriever: FastEmbed dense + BM25 sparse merged via RRF.
    Always supplements with DuckDuckGo web search for up-to-date context.
    """

    name = "retriever"

    def __init__(self) -> None:
        super().__init__()
        self._embedder = FastEmbedEmbeddings(model_name=cfg.embedding_model)

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        plan: dict = state.get("plan", {})
        query: str = state["query"]

        if not plan.get("requires_retrieval", True):
            self.logger.info("Retrieval skipped — plan says not required")
            return {**state, "retrieved_docs": [], "web_sources": []}

        top_k = cfg.retriever_top_k

        # ── Dense path ────────────────────────────────────────────────────────
        dense_hits: list[dict] = []
        try:
            embedding = self._embedder.embed_query(query)
            dense_hits = vs.query(embedding, top_k=top_k)
            for h in dense_hits:
                h["score"] = h["score"] * cfg.retriever_dense_weight
                h["source"] = "dense"
            retrieval_docs_histogram.labels(source="dense").observe(len(dense_hits))
            for h in dense_hits:
                retrieval_score_summary.labels(source="dense").observe(h["score"])
        except Exception as exc:
            self.logger.warning("Dense retrieval failed: %s", exc)

        # ── Sparse path: BM25 ─────────────────────────────────────────────────
        sparse_hits: list[dict] = []
        try:
            sparse_hits = bm25.search(query, top_k=top_k)
            for h in sparse_hits:
                h["score"] = h["score"] * cfg.retriever_bm25_weight
                h["source"] = "bm25"
            retrieval_docs_histogram.labels(source="bm25").observe(len(sparse_hits))
            for h in sparse_hits:
                retrieval_score_summary.labels(source="bm25").observe(h["score"])
        except Exception as exc:
            self.logger.warning("BM25 retrieval failed: %s", exc)

        merged = _reciprocal_rank_fusion(
            [h for h in [dense_hits, sparse_hits] if h]
        )[:top_k]
        retrieval_docs_histogram.labels(source="merged").observe(len(merged))

        # ── Web search — always runs to provide up-to-date context ────────────
        web_sources = _web_search(query, max_results=4)
        self.logger.info(
            "Retrieved %d local docs (dense=%d bm25=%d merged=%d) + %d web results",
            len(merged), len(dense_hits), len(sparse_hits), len(merged), len(web_sources),
        )

        return {**state, "retrieved_docs": merged, "web_sources": web_sources}
