# retrieval/__init__.py — Helios retrieval layer public API
# Author: Hridam Biswas | Project: Helios

from retrieval.bm25_search import BM25Index, get_index as get_bm25_index
from retrieval.vector_store import query as vector_query, upsert_batch, ping as vector_ping

__all__ = [
    "BM25Index",
    "get_bm25_index",
    "vector_query",
    "upsert_batch",
    "vector_ping",
]

# Web search is handled by RetrieverAgent; not exported from retrieval package
# to keep the separation between vector/BM25 retrieval and web retrieval.
