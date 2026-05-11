# Retrieval Strategy

Helios uses **three-path hybrid retrieval** merging dense vector search, BM25 lexical search, and live web search.

## Why Hybrid?

| Method | Strength |
|--------|----------|
| Dense (semantic) | Paraphrase, concept similarity |
| BM25 (lexical) | Exact keywords, proper nouns, code identifiers |
| Web (DuckDuckGo) | Real-time information not in the knowledge base |

## Reciprocal Rank Fusion (RRF)

Local results are merged using RRF: `score(doc) = Σ 1/(k + rank_i)` where k=60.

## Web Search

DuckDuckGo runs in parallel on every query where retrieval is needed. Results appear in `web_sources` and are cited as `[W1]`, `[W2]` in the answer with clickable URLs.

## Config

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVER_TOP_K` | 10 | Max local docs |
| `RETRIEVER_DENSE_WEIGHT` | 0.6 | Dense score weight |
| `WEB_SEARCH_MAX_RESULTS` | 4 | DuckDuckGo count |
