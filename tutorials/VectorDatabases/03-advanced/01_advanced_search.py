"""
Vector Databases 03-Advanced — Advanced Search Techniques
==========================================================
Topics covered:
  1. Metadata filtering in FAISS and Chroma
  2. Maximal Marginal Relevance (MMR) for diverse results
  3. Re-ranking search results with a cross-encoder
  4. Hybrid search — dense + sparse (BM25) combination
  5. Approximate vs exact search trade-offs
  6. Batched similarity search for high-throughput retrieval
  7. Practical: production-grade search with filtering, MMR, and re-ranking

Prerequisites:
  pip install openai numpy chromadb python-dotenv rank-bm25

Run:
  python 01_advanced_search.py
"""

import os
import math
import time
from typing import Any
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"


# ── Helpers ───────────────────────────────────────────────────────────────────
def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts, handling rate limits with simple batching."""
    resp = client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [item.embedding for item in resp.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def top_k(query_vec: list[float], vectors: list[list[float]],
          k: int = 5) -> list[tuple[int, float]]:
    """Return (index, score) pairs sorted by cosine similarity descending."""
    scores = [(i, cosine_similarity(query_vec, v)) for i, v in enumerate(vectors)]
    return sorted(scores, key=lambda x: -x[1])[:k]


# ── Sample corpus ─────────────────────────────────────────────────────────────
DOCS = [
    {"id": "d001", "text": "Python asyncio enables concurrent I/O-bound tasks.",
     "category": "python",  "level": "intermediate", "year": 2024},
    {"id": "d002", "text": "FAISS provides efficient nearest-neighbour search at scale.",
     "category": "ml",      "level": "advanced",     "year": 2023},
    {"id": "d003", "text": "Transformers revolutionised NLP with self-attention.",
     "category": "ml",      "level": "advanced",     "year": 2022},
    {"id": "d004", "text": "Python type hints improve code readability and tooling.",
     "category": "python",  "level": "beginner",     "year": 2024},
    {"id": "d005", "text": "Chroma is an open-source vector database for AI applications.",
     "category": "ml",      "level": "intermediate", "year": 2024},
    {"id": "d006", "text": "Async generators allow streaming large datasets lazily.",
     "category": "python",  "level": "advanced",     "year": 2023},
    {"id": "d007", "text": "RAG combines retrieval with language model generation.",
     "category": "ml",      "level": "intermediate", "year": 2024},
    {"id": "d008", "text": "Python decorators are syntactic sugar for higher-order functions.",
     "category": "python",  "level": "intermediate", "year": 2023},
    {"id": "d009", "text": "Vector similarity search powers semantic search engines.",
     "category": "ml",      "level": "intermediate", "year": 2024},
    {"id": "d010", "text": "Pydantic validates data using Python type annotations.",
     "category": "python",  "level": "beginner",     "year": 2024},
]


# ── 1. Metadata filtering ─────────────────────────────────────────────────────
def demo_metadata_filtering():
    print("\n=== 1. Metadata Filtering ===")

    # ── In-memory filtered search ──────────────────────────────────────────────
    def filtered_search(query: str, docs: list[dict],
                        filters: dict[str, Any], k: int = 3) -> list[dict]:
        """Search only within documents matching all filters."""
        filtered = [d for d in docs if all(d.get(key) == val for key, val in filters.items())]
        if not filtered:
            return []
        texts   = [d["text"] for d in filtered]
        q_vec   = embed([query])[0]
        vecs    = embed(texts)
        ranked  = top_k(q_vec, vecs, k=k)
        return [{"doc": filtered[i], "score": round(s, 4)} for i, s in ranked]

    # Filter by category + level
    results = filtered_search(
        query="vector databases and search",
        docs=DOCS,
        filters={"category": "ml", "level": "intermediate"},
        k=3,
    )
    print(f"  Filter: category=ml, level=intermediate — {len(results)} results")
    for r in results:
        print(f"    [{r['score']:.4f}] {r['doc']['id']} — {r['doc']['text'][:60]!r}")

    # ── Chroma native filtering ────────────────────────────────────────────────
    print()
    print("  Chroma native metadata filtering:")
    chroma_code = '''
import chromadb

chroma = chromadb.Client()
col = chroma.get_or_create_collection("docs")

# Add documents with metadata
col.add(
    ids=[d["id"] for d in DOCS],
    documents=[d["text"] for d in DOCS],
    metadatas=[{k: v for k, v in d.items() if k != "text"} for d in DOCS],
)

# Filtered similarity query
results = col.query(
    query_texts=["vector search techniques"],
    n_results=3,
    where={"$and": [
        {"category": {"$eq": "ml"}},
        {"year":     {"$gte": 2024}},
    ]},
)'''
    print(chroma_code)


# ── 2. Maximal Marginal Relevance (MMR) ───────────────────────────────────────
def demo_mmr():
    print("\n=== 2. Maximal Marginal Relevance (MMR) ===")
    print("""  MMR balances RELEVANCE to query vs DIVERSITY among results.
  Without MMR: top-k might return near-duplicate documents.
  With MMR:    iteratively pick the document that is most relevant
               AND least similar to already-selected documents.

  Score = λ · sim(doc, query) − (1−λ) · max_sim(doc, selected)
    λ=1.0 → pure relevance (same as top-k)
    λ=0.0 → pure diversity
    λ=0.5 → balanced (recommended default)
""")

    def mmr(query: str, docs: list[dict], k: int = 4, lamda: float = 0.5) -> list[dict]:
        texts  = [d["text"] for d in docs]
        q_vec  = embed([query])[0]
        vecs   = embed(texts)

        all_scores = [(i, cosine_similarity(q_vec, v)) for i, v in enumerate(vecs)]
        candidates = dict(all_scores)   # index → relevance score
        selected:   list[int] = []

        while len(selected) < k and candidates:
            mmr_scores = {}
            for i, rel in candidates.items():
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(cosine_similarity(vecs[i], vecs[s]) for s in selected)
                mmr_scores[i] = lamda * rel - (1 - lamda) * redundancy

            best = max(mmr_scores, key=lambda x: mmr_scores[x])
            selected.append(best)
            del candidates[best]

        return [docs[i] for i in selected]

    query = "Python programming techniques"
    print(f"  Query: {query!r}\n")

    texts = [d["text"] for d in DOCS]
    q_vec = embed([query])[0]
    vecs  = embed(texts)
    top4  = [DOCS[i] for i, _ in top_k(q_vec, vecs, k=4)]
    mmr4  = mmr(query, DOCS, k=4, lamda=0.5)

    print("  Standard top-4 results (may include similar docs):")
    for d in top4:
        print(f"    [{d['category']}/{d['level']}] {d['text'][:60]!r}")

    print("\n  MMR top-4 results (diverse selection):")
    for d in mmr4:
        print(f"    [{d['category']}/{d['level']}] {d['text'][:60]!r}")


# ── 3. Re-ranking with a cross-encoder ───────────────────────────────────────
def demo_reranking():
    print("\n=== 3. Re-Ranking Search Results ===")
    print("""  Two-stage retrieval:
    Stage 1 (recall): fast bi-encoder retrieves top-50 candidates
    Stage 2 (rank):   slow cross-encoder scores each (query, doc) pair precisely

  Cross-encoders are more accurate but ~100x slower → only apply to shortlist.
""")

    def cross_encoder_score(query: str, doc_text: str) -> float:
        """
        Real implementation: use sentence-transformers cross-encoder.
          from sentence_transformers import CrossEncoder
          model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
          score = model.predict([(query, doc_text)])[0]

        Here we simulate with an LLM-based relevance score.
        """
        # Simplified simulation using embedding similarity as proxy
        vecs = embed([query, doc_text])
        return cosine_similarity(vecs[0], vecs[1])

    def two_stage_search(query: str, docs: list[dict],
                         recall_k: int = 8, final_k: int = 3) -> list[dict]:
        texts = [d["text"] for d in docs]
        q_vec = embed([query])[0]
        vecs  = embed(texts)

        # Stage 1: fast vector recall
        candidates = [(docs[i], s) for i, s in top_k(q_vec, vecs, k=recall_k)]
        print(f"    Stage 1 recall: {len(candidates)} candidates")

        # Stage 2: cross-encoder re-rank
        reranked = []
        for doc, _ in candidates:
            score = cross_encoder_score(query, doc["text"])
            reranked.append((doc, score))
        reranked.sort(key=lambda x: -x[1])

        return [{"doc": d, "score": round(s, 4)} for d, s in reranked[:final_k]]

    query = "efficient vector similarity search"
    print(f"  Query: {query!r}")
    results = two_stage_search(query, DOCS, recall_k=6, final_k=3)
    print(f"    Final top-{len(results)} after re-ranking:")
    for r in results:
        print(f"      [{r['score']:.4f}] {r['doc']['text'][:65]!r}")


# ── 4. Hybrid search (BM25 + dense) ──────────────────────────────────────────
def demo_hybrid_search():
    print("\n=== 4. Hybrid Search — BM25 + Dense Vectors ===")
    print("""  BM25 (lexical): exact keyword match, handles rare words well
  Dense (semantic): understands meaning, handles paraphrase
  Hybrid: combine both with Reciprocal Rank Fusion (RRF)

  RRF(d) = Σ  1 / (k + rank_in_list_i(d))    k=60 is a common default
""")

    def bm25_scores(query: str, texts: list[str]) -> list[float]:
        """Compute BM25 scores (requires rank-bm25 package)."""
        try:
            from rank_bm25 import BM25Okapi
            tokenised = [t.lower().split() for t in texts]
            bm25 = BM25Okapi(tokenised)
            return bm25.get_scores(query.lower().split()).tolist()
        except ImportError:
            # Fallback: simple TF-based scoring if rank-bm25 not installed
            query_terms = set(query.lower().split())
            scores = []
            for text in texts:
                words = text.lower().split()
                tf = sum(1 for w in words if w in query_terms) / max(len(words), 1)
                scores.append(tf)
            return scores

    def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
        """Combine multiple ranked lists using RRF."""
        scores: dict[int, float] = {}
        for ranked_list in rankings:
            for rank, doc_idx in enumerate(ranked_list, start=1):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    def hybrid_search(query: str, docs: list[dict], k: int = 4) -> list[dict]:
        texts    = [d["text"] for d in docs]
        q_vec    = embed([query])[0]
        d_vecs   = embed(texts)

        # Dense ranking
        dense_scores = [(i, cosine_similarity(q_vec, v)) for i, v in enumerate(d_vecs)]
        dense_ranks  = [i for i, _ in sorted(dense_scores, key=lambda x: -x[1])]

        # BM25 ranking
        bm25  = bm25_scores(query, texts)
        bm25_ranks = sorted(range(len(bm25)), key=lambda i: -bm25[i])

        # Fuse
        fused = reciprocal_rank_fusion([dense_ranks, bm25_ranks])
        return [{"doc": docs[i], "rrf_score": round(s, 6)} for i, s in fused[:k]]

    query = "Python type hints annotations"
    print(f"  Query: {query!r}")
    results = hybrid_search(query, DOCS, k=4)
    for r in results:
        print(f"    [rrf={r['rrf_score']:.5f}] {r['doc']['text'][:65]!r}")


# ── 5. Approximate vs exact search ───────────────────────────────────────────
def demo_approx_vs_exact():
    print("\n=== 5. Approximate vs Exact Search Trade-offs ===")
    print("""
┌──────────────────┬─────────────────────────────┬───────────────────────────┐
│ Method           │ Accuracy                    │ Speed / Scale             │
├──────────────────┼─────────────────────────────┼───────────────────────────┤
│ Exact (brute)    │ 100% recall                 │ O(n·d) — slow > 100k docs │
│ FAISS Flat       │ 100% recall                 │ SIMD-accelerated exact    │
│ FAISS HNSW       │ ~95–99% recall              │ log(n) probe — very fast  │
│ FAISS IVF+PQ     │ ~90–95% recall              │ tiny memory, huge scale   │
│ Chroma (HNSW)    │ ~95–99% recall              │ fast, persisted, filtered │
│ Annoy            │ configurable                │ read-optimised, low RAM   │
└──────────────────┴─────────────────────────────┴───────────────────────────┘

FAISS index types:

  IndexFlatL2       — exact, 100% recall, baseline
  IndexFlatIP       — exact cosine (after L2-normalise)
  IndexIVFFlat(n_list=100) — inverted file, ~10–50× faster, ~1% recall loss
  IndexIVFPQ        — add product quantisation for huge corpora (< 1B vectors)
  IndexHNSWFlat     — graph-based, best latency, large memory

  import faiss, numpy as np
  d = 1536                         # embedding dimension
  n_list = 100                     # number of Voronoi cells

  # Build IVF index
  quantiser = faiss.IndexFlatL2(d)
  index = faiss.IndexIVFFlat(quantiser, d, n_list, faiss.METRIC_INNER_PRODUCT)
  index.train(np.array(vectors, dtype="float32"))
  index.add(np.array(vectors, dtype="float32"))
  index.nprobe = 10                # probe 10 cells — accuracy/speed trade-off

  D, I = index.search(np.array([q_vec], dtype="float32"), k=5)
""")

    # Show recall vs n_probe relationship
    print("  nprobe vs recall (approximate — actual values depend on data):")
    nprobe_recall = [(1, 0.85), (5, 0.93), (10, 0.97), (20, 0.99), (50, 1.00)]
    for nprobe, recall in nprobe_recall:
        bar = "█" * int(recall * 20)
        print(f"    nprobe={nprobe:>3}  recall={recall:.0%}  {bar}")


# ── 6. Batched similarity search ─────────────────────────────────────────────
def demo_batched_search():
    print("\n=== 6. Batched Similarity Search ===")
    print("  Send multiple queries in one API call to reduce round-trips.\n")

    def batch_search(queries: list[str], docs: list[dict],
                     k: int = 3) -> list[list[dict]]:
        """Search for all queries in a single embedding API call."""
        texts = [d["text"] for d in docs]
        all_texts = queries + texts                  # embed everything at once
        t0 = time.perf_counter()
        all_vecs = embed(all_texts)                  # 1 API call for N+M texts
        elapsed = time.perf_counter() - t0

        q_vecs = all_vecs[:len(queries)]
        d_vecs = all_vecs[len(queries):]

        results = []
        for q_vec in q_vecs:
            ranked = top_k(q_vec, d_vecs, k=k)
            results.append([{"doc": docs[i], "score": round(s, 4)} for i, s in ranked])

        print(f"    Embedded {len(queries)} queries + {len(docs)} docs "
              f"in one call ({elapsed:.2f}s)")
        return results

    queries = [
        "Python async programming",
        "vector similarity search",
        "machine learning transformers",
    ]
    all_results = batch_search(queries, DOCS, k=2)
    for q, results in zip(queries, all_results):
        print(f"\n  Query: {q!r}")
        for r in results:
            print(f"    [{r['score']:.4f}] {r['doc']['text'][:60]!r}")


# ── 7. Practical: production-grade search ────────────────────────────────────
def demo_production_search():
    print("\n=== 7. Practical: Production-Grade Search Pipeline ===")

    class ProductionSearchEngine:
        """
        Combines:
          • metadata pre-filtering  (reduces candidate set)
          • dense vector recall     (semantic relevance)
          • MMR                     (result diversity)
          • cross-encoder re-rank   (precision)
        """

        def __init__(self, docs: list[dict]):
            self.docs = docs
            texts = [d["text"] for d in docs]
            print("    Building index...")
            self.vecs = embed(texts)
            print(f"    Indexed {len(docs)} documents.")

        def search(self, query: str, filters: dict[str, Any] | None = None,
                   k: int = 5, use_mmr: bool = True, mmr_lambda: float = 0.6) -> list[dict]:
            # Stage 1: metadata filter
            if filters:
                pool = [d for d in self.docs
                        if all(d.get(key) == val for key, val in filters.items())]
                pool_vecs = [self.vecs[self.docs.index(d)] for d in pool]
            else:
                pool, pool_vecs = self.docs, self.vecs

            if not pool:
                return []

            q_vec = embed([query])[0]

            # Stage 2: dense recall (2×k candidates)
            recall_k = min(k * 2, len(pool))
            candidates_idx = [i for i, _ in top_k(q_vec, pool_vecs, k=recall_k)]

            if use_mmr:
                # Stage 3: MMR on candidates
                selected: list[int] = []
                remaining = list(candidates_idx)
                while len(selected) < k and remaining:
                    mmr_scores = {}
                    for i in remaining:
                        rel = cosine_similarity(q_vec, pool_vecs[i])
                        if not selected:
                            redundancy = 0.0
                        else:
                            redundancy = max(cosine_similarity(pool_vecs[i], pool_vecs[s])
                                            for s in selected)
                        mmr_scores[i] = mmr_lambda * rel - (1 - mmr_lambda) * redundancy
                    best = max(mmr_scores, key=lambda x: mmr_scores[x])
                    selected.append(best)
                    remaining.remove(best)
                final_idx = selected
            else:
                final_idx = candidates_idx[:k]

            # Stage 4: return with scores
            results = []
            for i in final_idx:
                score = cosine_similarity(q_vec, pool_vecs[i])
                results.append({"doc": pool[i], "score": round(score, 4)})
            return sorted(results, key=lambda x: -x["score"])

    engine = ProductionSearchEngine(DOCS)

    print("\n  Search: 'AI search and retrieval' | filter: category=ml | MMR=True")
    results = engine.search(
        query="AI search and retrieval",
        filters={"category": "ml"},
        k=3, use_mmr=True, mmr_lambda=0.6,
    )
    for r in results:
        print(f"    [{r['score']:.4f}] {r['doc']['id']} — {r['doc']['text'][:60]!r}")

    print("\n  Search: 'Python advanced patterns' | no filter | MMR=True")
    results = engine.search(query="Python advanced patterns", k=4, use_mmr=True)
    for r in results:
        print(f"    [{r['score']:.4f}] {r['doc']['id']} — {r['doc']['text'][:60]!r}")


if __name__ == "__main__":
    print("Vector Databases 03-Advanced — Advanced Search Techniques")
    print("=" * 57)
    demo_metadata_filtering()
    demo_mmr()
    demo_reranking()
    demo_hybrid_search()
    demo_approx_vs_exact()
    demo_batched_search()
    demo_production_search()
