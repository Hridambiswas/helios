# retrieval/vector_store.py — Helios ChromaDB vector store
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations
import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import cfg

logger = logging.getLogger("helios.retrieval.vector_store")

_client: Any = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Return the singleton Chroma collection.

    When CHROMA_PATH is set, use an embedded PersistentClient rooted at
    that directory (required inside HF Spaces, which only expose one
    inbound port). Otherwise use the HttpClient against the sidecar
    Chroma container defined in docker-compose.yml.
    """
    global _client, _collection
    if _collection is None:
        if cfg.chroma_path:
            _client = chromadb.PersistentClient(
                path=cfg.chroma_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            mode = f"persistent @ {cfg.chroma_path}"
        else:
            _client = chromadb.HttpClient(
                host=cfg.chroma_host,
                port=cfg.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            mode = f"http @ {cfg.chroma_host}:{cfg.chroma_port}"
        _collection = _client.get_or_create_collection(
            name=cfg.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d docs, %s)",
            cfg.chroma_collection,
            _collection.count(),
            mode,
        )
    return _collection


def _reset_for_tests() -> None:
    """Clear cached client/collection so tests can swap CHROMA_PATH per case."""
    global _client, _collection
    _client = None
    _collection = None


# ── Write ops ─────────────────────────────────────────────────────────────────

def upsert(
    doc_id: str,
    embedding: list[float],
    document: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Upsert a single document chunk into the collection."""
    _get_collection().upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata or {}],
    )


def upsert_batch(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    """Batch upsert — preferred for ingestion pipelines."""
    _get_collection().upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas or [{} for _ in ids],
    )
    logger.info("Upserted %d chunks into ChromaDB", len(ids))


def delete(doc_id: str) -> None:
    _get_collection().delete(ids=[doc_id])


def delete_batch(ids: list[str]) -> None:
    """Remove a batch of chunk IDs — used by saga compensation."""
    if ids:
        _get_collection().delete(ids=ids)
        logger.info("Deleted %d chunks from ChromaDB", len(ids))


# ── Query ops ─────────────────────────────────────────────────────────────────

def query_by_text(text: str, top_k: int | None = None, where: dict | None = None) -> list[dict]:
    """Convenience wrapper: embed text with CLIP then query ChromaDB."""
    from retrieval.clip_encoder import encode_text
    embedding = encode_text(text)[0]
    return query(embedding, top_k=top_k, where=where)


def query(
    embedding: list[float],
    top_k: int | None = None,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search by embedding. Returns list of:
      {id, document, metadata, distance, score (1-distance)}
    """
    k = top_k or cfg.retriever_top_k
    results = _get_collection().query(
        query_embeddings=[embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        hits.append({
            "id": doc_id,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": distance,
            "score": 1.0 - distance,   # cosine similarity
        })
    return hits


def count() -> int:
    return _get_collection().count()


def ping() -> bool:
    try:
        _get_collection().count()
        return True
    except Exception as exc:
        logger.error("ChromaDB ping failed: %s", exc)
        return False


def get_chunks_for_doc(doc_id: str, limit: int = 20, offset: int = 0) -> list[dict]:
    """Return raw text chunks stored in ChromaDB for a given document."""
    col = _get_collection()
    results = col.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
        limit=limit,
        offset=offset,
    )
    chunks = []
    for i, chunk_id in enumerate(results["ids"]):
        chunks.append({
            "id": chunk_id,
            "text": results["documents"][i] if results["documents"] else "",
            "metadata": results["metadatas"][i] if results["metadatas"] else {},
        })
    return chunks


def query_by_doc(query: str, doc_id: str, top_k: int = 5) -> list[dict]:
    """Dense-only retrieval scoped to a single document id."""
    from langchain_community.embeddings import FastEmbedEmbeddings
    embedder = FastEmbedEmbeddings(model_name=cfg.embedding_model)
    embedding = embedder.embed_query(query)
    col = _get_collection()
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(top_k, col.count()),
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i, chunk_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        hits.append({
            "id": chunk_id,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1.0 - distance,
            "source": "dense",
        })
    return hits
