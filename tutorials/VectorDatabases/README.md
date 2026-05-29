# Vector Databases Tutorial

In-depth coverage of the most popular vector database systems: FAISS, ChromaDB, and Qdrant.
Learn how to store, index, and retrieve high-dimensional embeddings efficiently.

## Curriculum

```
01-basics/
  01_embeddings_primer.py      — What embeddings are, how to create them
  02_faiss_basics.py           — FAISS: flat index, IVF index, HNSW

02-intermediate/
  01_chroma.py                 — ChromaDB: collections, metadata, filtering
  02_qdrant.py                 — Qdrant: payloads, filtering, named vectors

03-advanced/
  01_index_comparison.py       — FAISS vs Chroma vs Qdrant benchmark

04-UseCases/
  01_semantic_search.py        — Production-grade semantic search engine
  02_product_recommendation.py — Recommendation system with vector similarity
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python 01-basics/01_embeddings_primer.py
```

## Vector DB Comparison

| Feature | FAISS | ChromaDB | Qdrant |
|---------|-------|----------|--------|
| Persistence | Save/load files | Built-in SQLite | Built-in |
| Metadata filtering | No | Yes | Yes (advanced) |
| Deployment | In-process | In-process / Server | Client/Server |
| Best for | Speed, research | Simple apps | Production |

## Prerequisites
- Python 3.10+
- `OPENAI_API_KEY` in `.env`
