"""
Vector Databases 01-basics — FAISS: Flat, IVF, and HNSW Indexes
================================================================
Topics covered:
  1. FAISS flat index (exact nearest neighbour search)
  2. IVF (Inverted File) index for approximate search
  3. HNSW (Hierarchical Navigable Small World) index
  4. Saving and loading FAISS indexes
  5. Performance vs accuracy trade-offs

Run:
  python 02_faiss_basics.py
"""

import os
import time
import tempfile
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_DIM = 1536

# Sample knowledge base
DOCUMENTS = [
    "Python is a high-level, dynamically typed programming language known for readability.",
    "NumPy provides efficient array operations and is the foundation of scientific Python.",
    "Pandas provides DataFrame structures for data analysis and manipulation.",
    "Scikit-learn is the standard machine learning library for Python.",
    "PyTorch is a deep learning framework favoured for research flexibility.",
    "TensorFlow is a deep learning framework from Google used in production.",
    "FastAPI is a modern, high-performance Python web framework for building APIs.",
    "LangChain is a framework for building LLM-powered applications.",
    "Hugging Face Transformers provides pre-trained NLP models.",
    "FAISS is Facebook AI's library for efficient similarity search of dense vectors.",
    "ChromaDB is an open-source vector database for AI applications.",
    "Qdrant is a vector similarity search engine for production deployments.",
    "OpenAI provides GPT models via API for text generation and embeddings.",
    "LlamaIndex is a framework for connecting LLMs to external data sources.",
    "AutoGen is a framework from Microsoft for building multi-agent AI systems.",
]


def get_embeddings(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    vecs = np.array([item.embedding for item in response.data], dtype="float32")
    faiss.normalize_L2(vecs)   # Normalise for cosine similarity via inner product
    return vecs


def query_vector(text: str) -> np.ndarray:
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    vec = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


# ── 1. Flat index (exact search) ──────────────────────────────────────────────
def demo_flat_index():
    print("\n=== 1. FAISS Flat Index (Exact Search) ===")

    vectors = get_embeddings(DOCUMENTS)

    # IndexFlatIP = Inner Product (cosine similarity after L2 normalisation)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    print(f"Index type: {type(index).__name__}")
    print(f"Total vectors: {index.ntotal}")
    print(f"Vector dimension: {index.d}")

    queries = [
        "best framework for deep learning research",
        "how to query a vector database",
        "Python for data analysis",
    ]

    for q in queries:
        qvec = query_vector(q)
        scores, indices = index.search(qvec, k=3)
        print(f"\nQuery: \"{q}\"")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            print(f"  {rank}. [score={score:.4f}] {DOCUMENTS[idx]}")


# ── 2. IVF index (approximate, faster for large datasets) ─────────────────────
def demo_ivf_index():
    print("\n=== 2. FAISS IVF Index (Approximate Search) ===")
    print("IVF partitions the space into clusters (Voronoi cells).")
    print("At query time, only nearby clusters are searched — faster but approximate.\n")

    vectors = get_embeddings(DOCUMENTS)

    # IndexIVFFlat requires a quantizer and training
    nlist = 4        # number of clusters (for small datasets, use sqrt(N))
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index_ivf = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_INNER_PRODUCT)

    # IVF indexes must be trained before adding vectors
    index_ivf.train(vectors)
    index_ivf.add(vectors)

    print(f"Index type: {type(index_ivf).__name__}")
    print(f"Number of partitions (nlist): {nlist}")
    print(f"Trained: {index_ivf.is_trained}")
    print(f"Total vectors: {index_ivf.ntotal}")

    # nprobe: how many clusters to search — higher = more accurate but slower
    for nprobe in [1, 2, 4]:
        index_ivf.nprobe = nprobe
        qvec = query_vector("deep learning framework")
        t0 = time.perf_counter()
        scores, indices = index_ivf.search(qvec, k=3)
        elapsed = (time.perf_counter() - t0) * 1000
        top = DOCUMENTS[indices[0][0]][:60] + "..."
        print(f"\nnprobe={nprobe}: {elapsed:.2f}ms | top result: {top}")


# ── 3. HNSW index (fast, production-ready ANN) ────────────────────────────────
def demo_hnsw_index():
    print("\n=== 3. FAISS HNSW Index (Hierarchical Navigable Small World) ===")
    print("HNSW builds a multi-layer graph for logarithmic-time search.")
    print("No training required. Excellent recall at high speed.\n")

    vectors = get_embeddings(DOCUMENTS)

    # M = number of connections per node (higher = better quality, more memory)
    M = 32
    index_hnsw = faiss.IndexHNSWFlat(EMBEDDING_DIM, M, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.add(vectors)

    print(f"Index type: {type(index_hnsw).__name__}")
    print(f"M (connections per node): {M}")
    print(f"Total vectors: {index_hnsw.ntotal}")
    print(f"Trained (not required): {index_hnsw.is_trained}")

    # efSearch: search-time exploration factor (higher = better recall, slower)
    for ef_search in [16, 32, 64]:
        index_hnsw.hnsw.efSearch = ef_search
        qvec = query_vector("multi-agent AI framework")
        t0 = time.perf_counter()
        scores, indices = index_hnsw.search(qvec, k=3)
        elapsed = (time.perf_counter() - t0) * 1000
        top = DOCUMENTS[indices[0][0]][:60] + "..."
        print(f"efSearch={ef_search:3d}: {elapsed:.2f}ms | top: {top}")


# ── 4. Adding IDs and saving/loading ─────────────────────────────────────────
def demo_ids_and_persistence():
    print("\n=== 4. FAISS with Custom IDs and Persistence ===")

    vectors = get_embeddings(DOCUMENTS)

    # IndexIDMap wraps another index to support custom integer IDs
    base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index_with_ids = faiss.IndexIDMap(base_index)

    # Use 1-based IDs
    ids = np.arange(1, len(DOCUMENTS) + 1, dtype="int64")
    index_with_ids.add_with_ids(vectors, ids)

    qvec = query_vector("Python library for machine learning")
    scores, returned_ids = index_with_ids.search(qvec, k=3)

    print("Search with custom IDs:")
    for score, doc_id in zip(scores[0], returned_ids[0]):
        print(f"  ID={doc_id}: [score={score:.4f}] {DOCUMENTS[doc_id - 1]}")

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "my_index.faiss")
        faiss.write_index(index_with_ids, index_path)
        size_kb = os.path.getsize(index_path) / 1024
        print(f"\nSaved index to disk: {size_kb:.1f} KB")

        loaded_index = faiss.read_index(index_path)
        scores2, ids2 = loaded_index.search(qvec, k=1)
        print(f"Loaded index — top result ID: {ids2[0][0]}")
        print(f"  → {DOCUMENTS[ids2[0][0] - 1]}")


# ── 5. Index comparison summary ───────────────────────────────────────────────
def demo_index_comparison():
    print("\n=== 5. FAISS Index Comparison ===")
    print("""
  Index Type    │ Accuracy │ Speed     │ Memory    │ Training │ Best For
  ──────────────┼──────────┼───────────┼───────────┼──────────┼──────────────────────
  IndexFlatL2   │ Exact    │ Slow O(N) │ Low       │ No       │ Small datasets (<50k)
  IndexFlatIP   │ Exact    │ Slow O(N) │ Low       │ No       │ Cosine similarity
  IndexIVFFlat  │ ~98%+    │ Fast      │ Low       │ Yes      │ Medium datasets
  IndexHNSWFlat │ ~99%+    │ Very fast │ Moderate  │ No       │ Production, large sets
  IndexIVFPQ    │ ~95%+    │ Fastest   │ Very low  │ Yes      │ Billion-scale
""")
    print("Tip: For most applications, HNSW with M=32 is the best default choice.")


if __name__ == "__main__":
    demo_flat_index()
    demo_ivf_index()
    demo_hnsw_index()
    demo_ids_and_persistence()
    demo_index_comparison()
