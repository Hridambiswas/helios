#!/usr/bin/env python3
# scripts/ingest_demo.py — Bulk-ingest a directory of text files into Helios
# Author: Hridam Biswas | Project: Helios
"""
Usage:
    python scripts/ingest_demo.py --dir data/docs --ext .txt .md

Indexes all matching files into ChromaDB + BM25 without going through the HTTP API.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import OpenAIEmbeddings
from config import cfg
from retrieval.vector_store import upsert_batch
from retrieval.bm25_search import get_index
from storage.object_store import upload, ensure_bucket
import uuid


def chunk_text(text: str, size: int = 500) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    chunks = []
    for para in paras:
        for i in range(0, len(para), size):
            chunks.append(para[i: i + size])
    return chunks


def ingest_file(path: Path, embedder: OpenAIEmbeddings) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text)
    if not chunks:
        return 0

    doc_id = str(uuid.uuid4())
    minio_key = f"docs/{doc_id}/{path.name}"
    ensure_bucket()
    upload(minio_key, path.read_bytes())

    chunk_ids = [f"{doc_id}::chunk::{i}" for i in range(len(chunks))]
    embeddings = embedder.embed_documents(chunks)
    metas = [{"doc_id": doc_id, "filename": path.name, "chunk_idx": i} for i in range(len(chunks))]

    upsert_batch(chunk_ids, embeddings, chunks, metas)
    get_index().add_batch(chunk_ids, chunks, metas)
    print(f"  {path.name} → {len(chunks)} chunks")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory of files to ingest")
    parser.add_argument("--ext", nargs="+", default=[".txt", ".md"], help="File extensions")
    args = parser.parse_args()

    src = Path(args.dir)
    if not src.is_dir():
        print(f"Not a directory: {src}")
        sys.exit(1)

    files = [f for ext in args.ext for f in src.rglob(f"*{ext}")]
    print(f"Found {len(files)} files in {src}")

    embedder = OpenAIEmbeddings(model=cfg.openai_embedding_model, api_key=cfg.openai_api_key)
    total = sum(ingest_file(f, embedder) for f in files)
    print(f"\nDone — {total} total chunks indexed")


if __name__ == "__main__":
    main()
