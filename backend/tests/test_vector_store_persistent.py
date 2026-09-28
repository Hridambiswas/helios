# tests/test_vector_store_persistent.py — Guard embedded Chroma mode.
# When CHROMA_PATH is set, retrieval.vector_store must open an on-disk
# PersistentClient at that path and read/write through it — no HTTP
# connection attempted. This is the mode used inside the Hugging Face
# Space (see 004_prompt.md).
# Author: Hridam Biswas | Project: Helios

from __future__ import annotations

import os


def _clear_all_caches():
    """Reset both config's lru_cache and vector_store's module globals so
    each test observes fresh env-var → settings → collection state."""
    import config
    config.get_settings.cache_clear()
    from retrieval import vector_store as vs
    vs._reset_for_tests()


def _fresh_settings():
    """Return a Settings() built from the *current* environment, bypassing
    the cached module-level `cfg` object."""
    import config
    config.get_settings.cache_clear()
    return config.get_settings()


def test_persistent_mode_creates_files_on_disk(monkeypatch, tmp_path):
    """The real evidence that PersistentClient (not HttpClient) is active
    is that Chroma writes files under CHROMA_PATH — HttpClient would try to
    open a socket and fail."""
    path = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PATH", str(path))
    monkeypatch.setenv("CHROMA_COLLECTION", "helios_test_a")
    _clear_all_caches()

    settings = _fresh_settings()
    assert settings.chroma_path == str(path)

    from retrieval import vector_store as vs
    # Rebind vs.cfg to the fresh settings so it picks up CHROMA_PATH
    # regardless of whether an earlier import bound the old cfg.
    vs.cfg = settings

    coll = vs._get_collection()
    assert coll.count() == 0
    # PersistentClient materialises its sqlite index eagerly.
    assert path.exists() and any(path.iterdir()), (
        "PersistentClient should have written files under CHROMA_PATH; "
        f"tree: {list(path.iterdir())}"
    )
    _clear_all_caches()


def test_upsert_and_query_roundtrip_persistent(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "helios_test_b")
    _clear_all_caches()

    settings = _fresh_settings()
    from retrieval import vector_store as vs
    vs.cfg = settings

    embedding = [0.1] * 384  # BAAI/bge-small-en-v1.5 dimension
    vs.upsert_batch(
        ids=["doc-1", "doc-2"],
        embeddings=[embedding, [0.2] * 384],
        documents=["hello world", "second document"],
        metadatas=[{"doc_id": "doc-1"}, {"doc_id": "doc-2"}],
    )
    hits = vs.query(embedding, top_k=2)
    ids = [h["id"] for h in hits]
    assert "doc-1" in ids
    _clear_all_caches()


def test_http_client_default_when_chroma_path_unset(monkeypatch):
    """Without CHROMA_PATH, chroma_path is empty and the HTTP branch is
    selected. We don't open the client (no sidecar in unit tests) — just
    confirm the config bit that drives the branch."""
    monkeypatch.delenv("CHROMA_PATH", raising=False)
    # Sanity: the environment we hand to Settings must not have it.
    assert "CHROMA_PATH" not in os.environ

    settings = _fresh_settings()
    assert settings.chroma_path == ""
