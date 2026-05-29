"""
Vector Databases 01-basics — Embeddings Primer
===============================================
Topics covered:
  1. What are vector embeddings and why they matter
  2. Creating embeddings with OpenAI
  3. Cosine similarity and distance metrics
  4. Visualising embedding space concepts
  5. Practical embedding dimensions and trade-offs

Run:
  python 01_embeddings_primer.py
"""

import os
import math
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed a list of texts and return a list of vectors."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# ── 1. What is an embedding? ──────────────────────────────────────────────────
def demo_what_is_embedding():
    print("\n=== 1. What is a Vector Embedding? ===")
    print("""
An embedding converts text (or images, audio, etc.) into a fixed-length
numerical vector where semantic similarity corresponds to geometric proximity.

   "cat"  → [0.12, -0.87, 0.34, 0.56, ...]  (1536 dimensions)
   "kitten" → [0.14, -0.83, 0.31, 0.52, ...]  (close to "cat")
   "Python" → [0.88, 0.21, -0.44, 0.09, ...]  (far from "cat")
""")

    texts = ["cat", "kitten", "dog", "Python programming", "machine learning"]
    vectors = embed(texts)

    print(f"Embedding model: text-embedding-3-small")
    print(f"Vector dimensions: {len(vectors[0])}")
    print(f"\nFirst 8 dimensions of 'cat': {[round(x, 4) for x in vectors[0][:8]]}")

    print("\nSimilarity matrix (cosine):")
    header = f"{'':18}" + "".join(f"{t:18}" for t in texts)
    print(header)
    for i, t1 in enumerate(texts):
        row = f"{t1:18}"
        for j, _ in enumerate(texts):
            sim = cosine_similarity(vectors[i], vectors[j])
            row += f"{sim:18.4f}"
        print(row)


# ── 2. Semantic similarity intuition ─────────────────────────────────────────
def demo_semantic_similarity():
    print("\n=== 2. Semantic Similarity ===")

    anchor = "I love programming in Python"
    comparisons = [
        "Python is my favourite coding language",    # very similar
        "I enjoy writing software",                  # similar
        "Machine learning with Python is fun",       # partially similar
        "Cooking pasta is relaxing",                 # unrelated
        "The stock market crashed today",            # completely unrelated
    ]

    all_texts = [anchor] + comparisons
    vectors = embed(all_texts)
    anchor_vec = vectors[0]

    print(f"Anchor: \"{anchor}\"\n")
    results = []
    for i, text in enumerate(comparisons):
        sim = cosine_similarity(anchor_vec, vectors[i + 1])
        results.append((sim, text))

    for sim, text in sorted(results, reverse=True):
        bar = "█" * int(sim * 40)
        print(f"  {sim:.4f} {bar}")
        print(f"         \"{text}\"")


# ── 3. Distance metrics comparison ───────────────────────────────────────────
def demo_distance_metrics():
    print("\n=== 3. Distance Metrics ===")
    print("""
Three common metrics for vector similarity:

  Cosine Similarity:  Angle between vectors (0=perpendicular, 1=same direction)
                      Best for semantic similarity — ignores magnitude
  
  Euclidean Distance: Straight-line distance (0=identical)
                      Sensitive to vector magnitude
  
  Dot Product:        Cosine × magnitude of both vectors
                      Used when magnitude carries meaning (e.g. importance weights)
""")

    pairs = [
        ("Python is great", "I love Python"),
        ("Python is great", "JavaScript is awesome"),
        ("Python is great", "The weather is nice today"),
    ]

    all_texts = list(set(t for pair in pairs for t in pair))
    vecs = dict(zip(all_texts, embed(all_texts)))

    print(f"{'Text A':30} {'Text B':30} {'Cosine':8} {'L2 dist':8}")
    print("-" * 80)
    for a, b in pairs:
        cos = cosine_similarity(vecs[a], vecs[b])
        l2  = euclidean_distance(vecs[a], vecs[b])
        print(f"{a[:28]:30} {b[:28]:30} {cos:8.4f} {l2:8.4f}")


# ── 4. Embedding models comparison ───────────────────────────────────────────
def demo_embedding_models():
    print("\n=== 4. OpenAI Embedding Models ===")
    print("""
  Model                      Dimensions  Cost (per 1M tokens)
  ─────────────────────────────────────────────────────────────
  text-embedding-3-small     1536        $0.02   (recommended)
  text-embedding-3-large     3072        $0.13   (higher quality)
  text-embedding-ada-002     1536        $0.10   (legacy)
""")

    text = "Vector embeddings enable semantic search and AI memory"
    small_vec  = embed([text], model="text-embedding-3-small")[0]
    large_vec  = embed([text], model="text-embedding-3-large")[0]

    print(f"text-embedding-3-small → {len(small_vec)} dims, first 5: {[round(x, 4) for x in small_vec[:5]]}")
    print(f"text-embedding-3-large → {len(large_vec)} dims, first 5: {[round(x, 4) for x in large_vec[:5]]}")

    print("\nNote: 3-small supports 'dimensions' parameter for truncation:")
    small_256 = client.embeddings.create(
        input=[text], model="text-embedding-3-small", dimensions=256
    ).data[0].embedding
    print(f"  256-dim vector: {len(small_256)} dims — smaller, faster for low-latency apps")


# ── 5. Nearest neighbour search (manual) ─────────────────────────────────────
def demo_nearest_neighbour():
    print("\n=== 5. Manual Nearest Neighbour Search ===")
    print("(This is what vector databases automate at scale)")

    corpus = [
        "How do I reverse a string in Python?",
        "What is object-oriented programming?",
        "Explain gradient descent in machine learning",
        "How does a neural network learn?",
        "What are list comprehensions in Python?",
        "Difference between supervised and unsupervised learning",
        "How to handle exceptions in Python?",
        "What is backpropagation?",
        "How to use decorators in Python?",
        "Explain the transformer architecture",
    ]

    query = "How does deep learning train itself?"
    print(f"\nQuery: \"{query}\"")

    all_texts = [query] + corpus
    all_vecs  = embed(all_texts)
    query_vec, corpus_vecs = all_vecs[0], all_vecs[1:]

    scores = [(cosine_similarity(query_vec, cv), txt) for cv, txt in zip(corpus_vecs, corpus)]
    scores.sort(reverse=True)

    print("\nTop 5 nearest neighbours:")
    for rank, (score, text) in enumerate(scores[:5], 1):
        print(f"  {rank}. [{score:.4f}] {text}")


if __name__ == "__main__":
    demo_what_is_embedding()
    demo_semantic_similarity()
    demo_distance_metrics()
    demo_embedding_models()
    demo_nearest_neighbour()
