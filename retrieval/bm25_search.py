# retrieval/bm25_search.py — BM25 sparse text search
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger("helios.retrieval.bm25_search")

_TOKENIZE_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    return _TOKENIZE_RE.findall(text.lower())


@dataclass
class BM25Index:
    """
    In-memory BM25 index that mirrors the ChromaDB corpus.
    Thread-safe for concurrent reads; writes use a lock.
    """
    _docs: list[str] = field(default_factory=list)
    _ids: list[str] = field(default_factory=list)
    _metas: list[dict] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        with self._lock:
            self._docs.append(text)
            self._ids.append(doc_id)
            self._metas.append(metadata or {})
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._docs])

    def add_batch(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        with self._lock:
            # Deduplicate: skip IDs already in the index
            existing = set(self._ids)
            new_ids, new_texts, new_metas = [], [], []
            for i, doc_id in enumerate(ids):
                if doc_id not in existing:
                    new_ids.append(doc_id)
                    new_texts.append(texts[i])
                    new_metas.append((metadatas or [{}] * len(ids))[i])
            self._docs.extend(new_texts)
            self._ids.extend(new_ids)
            self._metas.extend(new_metas)
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._docs])
        logger.info("BM25 index rebuilt: %d documents (%d new)", len(self._docs), len(new_ids))

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Return top-k hits as [{id, document, metadata, score}].
        Scores are normalised to [0, 1].
        """
        if not self._docs or self._bm25 is None:
            return []

        tokens = _tokenize(query)
        scores: np.ndarray = self._bm25.get_scores(tokens)

        top_indices = scores.argsort()[::-1][:top_k]
        max_score = float(scores.max()) if scores.max() > 0 else 1.0

        results = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue
            results.append({
                "id": self._ids[idx],
                "document": self._docs[idx],
                "metadata": self._metas[idx],
                "score": float(scores[idx]) / max_score,
            })
        return results

    def remove(self, doc_id: str) -> None:
        with self._lock:
            try:
                i = self._ids.index(doc_id)
            except ValueError:
                return
            self._docs.pop(i)
            self._ids.pop(i)
            self._metas.pop(i)
            self._bm25 = BM25Okapi([_tokenize(d) for d in self._docs]) if self._docs else None

    def __len__(self) -> int:
        return len(self._docs)


# ── Global singleton index ────────────────────────────────────────────────────

_index: BM25Index = BM25Index()


def get_index() -> BM25Index:
    return _index


def search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    return _index.search(query, top_k=top_k)
