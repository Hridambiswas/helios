# agents/retriever.py — Helios Multi-Modal Retriever Agent (CLIP + BM25)
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings

from config import cfg
from agents.base import BaseAgent
import retrieval.vector_store as vs
import retrieval.bm25_search as bm25
from retrieval.clip_encoder import encode_text
from observability.metrics import retrieval_docs_histogram, retrieval_score_summary

logger = logging.getLogger("helios.agents.retriever")


def _deduplicate(hits: list[dict]) -> list[dict]:
    """
    Merge results from multiple retrieval paths.
    Deduplicates by doc id, keeps the highest score across all sources.
    """
    seen: dict[str, dict] = {}
    for hit in hits:
        doc_id = hit["id"]
        if doc_id not in seen or hit["score"] > seen[doc_id]["score"]:
            seen[doc_id] = hit
    return sorted(seen.values(), key=lambda h: h["score"], reverse=True)


class RetrieverAgent(BaseAgent):
    """
    Hybrid retriever: OpenAI dense embedding + CLIP + BM25 sparse.
    Scores weighted by cfg.retriever_clip_weight / bm25_weight, then merged.
    """

    name = "retriever"

    def __init__(self) -> None:
        super().__init__()
        self._embedder = OpenAIEmbeddings(
            model=cfg.openai_embedding_model,
            api_key=cfg.openai_api_key,
        )

    def _run(self, state: dict[str, Any]) -> dict[str, Any]:
        plan: dict = state.get("plan", {})
        query: str = state["query"]

        if not plan.get("requires_retrieval", True):
            self.logger.info("Retrieval skipped — plan says not required")
            return {**state, "retrieved_docs": []}

        top_k = cfg.retriever_top_k

        # ── Dense path: OpenAI embedding → ChromaDB ───────────────────────
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

        # ── CLIP path: CLIP text embedding → ChromaDB ─────────────────────
        clip_hits: list[dict] = []
        try:
            clip_emb = encode_text(query)[0]
            clip_hits = vs.query(clip_emb, top_k=top_k)
            for h in clip_hits:
                h["score"] = h["score"] * cfg.retriever_clip_weight
                h["source"] = "clip"
            retrieval_docs_histogram.labels(source="clip").observe(len(clip_hits))
        except Exception as exc:
            self.logger.warning("CLIP retrieval failed: %s", exc)

        # ── Sparse path: BM25 ─────────────────────────────────────────────
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

        merged = _deduplicate(dense_hits + clip_hits + sparse_hits)[:top_k]
        retrieval_docs_histogram.labels(source="merged").observe(len(merged))

        self.logger.info(
            "Retrieved %d docs (dense=%d  clip=%d  bm25=%d → merged=%d)",
            len(merged), len(dense_hits), len(clip_hits), len(sparse_hits), len(merged),
        )
        return {**state, "retrieved_docs": merged}
