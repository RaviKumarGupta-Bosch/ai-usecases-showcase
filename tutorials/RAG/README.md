# RAG Tutorial — Retrieval-Augmented Generation (Deep Dive)

A comprehensive tutorial covering every RAG pattern from basic to production-grade,
including naive RAG, advanced retrieval strategies, hybrid search, re-ranking, and
self-correcting RAG patterns.

## Curriculum

```
01-basics/
  01_naive_rag.py              — Basic chunk → embed → retrieve → generate pipeline
  02_document_processing.py   — Chunking strategies: fixed, recursive, semantic
  03_vector_stores.py          — FAISS and Chroma in-memory retrieval

02-intermediate/
  01_advanced_retrieval.py    — MMR, self-query, multi-query retrieval
  02_hybrid_search.py         — BM25 + vector search combined
  03_reranking.py             — Cross-encoder reranking pipeline

03-advanced/
  01_hyde.py                  — HyDE: Hypothetical Document Embeddings
  02_corrective_rag.py        — CRAG: self-corrective RAG
  03_adaptive_rag.py          — Adaptive RAG with routing

04-UseCases/
  01_qa_over_documents.py     — Q&A over a PDF/text corpus
  02_conversational_rag.py    — Multi-turn RAG with chat history
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python 01-basics/01_naive_rag.py
```

## Key Patterns

| Pattern | When to use |
|---------|-------------|
| Naive RAG | Simple Q&A, small corpus |
| MMR | Reduce redundancy in retrieved chunks |
| Multi-query | When single query misses relevant chunks |
| Hybrid Search | Keyword + semantic coverage |
| Cross-encoder rerank | Maximum precision, latency tolerant |
| HyDE | Sparse corpora, weak initial queries |
| CRAG | When hallucination is unacceptable |

## Prerequisites
- Python 3.10+
- `OPENAI_API_KEY`
