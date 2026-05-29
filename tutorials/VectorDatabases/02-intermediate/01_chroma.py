"""
Vector Databases 02-intermediate — ChromaDB: Collections, Metadata & Filtering
===============================================================================
Topics covered:
  1. Creating and managing Chroma collections
  2. Adding documents with metadata
  3. Semantic search with filters
  4. Updating and deleting documents
  5. Persistent client (data survives restarts)

Run:
  python 01_chroma.py
"""

import os
import tempfile
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Reusable embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

# Sample dataset: tech blog posts with metadata
BLOG_POSTS = [
    {
        "id": "post_001",
        "text": "Python decorators are a powerful way to modify function behaviour without changing function code. The @property decorator creates managed attributes.",
        "metadata": {"category": "Python", "difficulty": "intermediate", "year": 2024, "likes": 245},
    },
    {
        "id": "post_002",
        "text": "Docker containers package applications with dependencies for consistent deployment. Use multi-stage builds to create small production images.",
        "metadata": {"category": "DevOps", "difficulty": "beginner", "year": 2024, "likes": 180},
    },
    {
        "id": "post_003",
        "text": "Vector databases store high-dimensional embeddings for fast similarity search. FAISS, Chroma, and Qdrant are the most popular choices.",
        "metadata": {"category": "AI/ML", "difficulty": "intermediate", "year": 2024, "likes": 312},
    },
    {
        "id": "post_004",
        "text": "React hooks like useState and useEffect simplify state management and side effects in functional components.",
        "metadata": {"category": "JavaScript", "difficulty": "beginner", "year": 2023, "likes": 198},
    },
    {
        "id": "post_005",
        "text": "Transformer attention mechanisms process all tokens in parallel, enabling better long-range dependencies than RNNs.",
        "metadata": {"category": "AI/ML", "difficulty": "advanced", "year": 2023, "likes": 420},
    },
    {
        "id": "post_006",
        "text": "Kubernetes orchestrates container deployment, scaling, and management across clusters of machines.",
        "metadata": {"category": "DevOps", "difficulty": "advanced", "year": 2024, "likes": 290},
    },
    {
        "id": "post_007",
        "text": "Python async/await with asyncio enables concurrent I/O operations. Use aiohttp for async HTTP requests.",
        "metadata": {"category": "Python", "difficulty": "intermediate", "year": 2024, "likes": 175},
    },
    {
        "id": "post_008",
        "text": "RAG (Retrieval-Augmented Generation) combines vector search with LLMs to answer questions grounded in external documents.",
        "metadata": {"category": "AI/ML", "difficulty": "intermediate", "year": 2024, "likes": 510},
    },
]


# ── 1. In-memory Chroma collection ────────────────────────────────────────────
def demo_in_memory():
    print("\n=== 1. In-Memory ChromaDB Collection ===")

    chroma = chromadb.Client()
    collection = chroma.create_collection(
        name="blog_posts",
        embedding_function=openai_ef,
    )

    # Add documents
    collection.add(
        ids=[p["id"] for p in BLOG_POSTS],
        documents=[p["text"] for p in BLOG_POSTS],
        metadatas=[p["metadata"] for p in BLOG_POSTS],
    )

    print(f"Collection: {collection.name}")
    print(f"Total documents: {collection.count()}")

    # Basic semantic search
    results = collection.query(
        query_texts=["how do neural networks learn?"],
        n_results=3,
    )

    print("\nQuery: 'how do neural networks learn?'")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"  [{dist:.4f}] [{meta['category']}] {doc[:80]}...")


# ── 2. Metadata filtering ─────────────────────────────────────────────────────
def demo_metadata_filtering():
    print("\n=== 2. Metadata Filtering ===")

    chroma = chromadb.Client()
    collection = chroma.create_collection(name="filtered_posts", embedding_function=openai_ef)
    collection.add(
        ids=[p["id"] for p in BLOG_POSTS],
        documents=[p["text"] for p in BLOG_POSTS],
        metadatas=[p["metadata"] for p in BLOG_POSTS],
    )

    # Filter: only AI/ML posts
    print("\nQuery: 'machine learning' — only AI/ML category")
    results = collection.query(
        query_texts=["machine learning models"],
        n_results=3,
        where={"category": "AI/ML"},
    )
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [{meta['category']}] [{meta['difficulty']}] {doc[:80]}...")

    # Filter: popular posts (likes > 300) in 2024
    print("\nQuery: 'deployment' — likes > 300 and year = 2024")
    results = collection.query(
        query_texts=["production deployment"],
        n_results=3,
        where={"$and": [{"likes": {"$gt": 300}}, {"year": {"$eq": 2024}}]},
    )
    if results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"  [likes={meta['likes']}] [{meta['year']}] {doc[:80]}...")
    else:
        print("  No results matching filters")

    # Filter: beginner OR intermediate
    print("\nQuery: 'Python programming' — beginner or intermediate only")
    results = collection.query(
        query_texts=["Python programming"],
        n_results=3,
        where={"difficulty": {"$in": ["beginner", "intermediate"]}},
    )
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [{meta['difficulty']}] {doc[:80]}...")


# ── 3. Persistent Chroma client ───────────────────────────────────────────────
def demo_persistence():
    print("\n=== 3. Persistent ChromaDB (survives restarts) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Persisting to: {tmpdir}")

        # First session — create and populate
        client1 = chromadb.PersistentClient(path=tmpdir)
        col = client1.get_or_create_collection("persistent_posts", embedding_function=openai_ef)
        col.add(
            ids=[p["id"] for p in BLOG_POSTS[:4]],
            documents=[p["text"] for p in BLOG_POSTS[:4]],
            metadatas=[p["metadata"] for p in BLOG_POSTS[:4]],
        )
        print(f"Session 1: added {col.count()} documents")

        # Second session — reload without re-embedding
        client2 = chromadb.PersistentClient(path=tmpdir)
        col2 = client2.get_collection("persistent_posts", embedding_function=openai_ef)
        print(f"Session 2: reloaded {col2.count()} documents")

        results = col2.query(query_texts=["async programming"], n_results=2)
        print("Search after reload:")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"  [{meta['category']}] {doc[:80]}...")


# ── 4. Update and delete documents ───────────────────────────────────────────
def demo_update_delete():
    print("\n=== 4. Updating and Deleting Documents ===")

    chroma = chromadb.Client()
    collection = chroma.create_collection(name="mutable_posts", embedding_function=openai_ef)
    collection.add(
        ids=[p["id"] for p in BLOG_POSTS[:4]],
        documents=[p["text"] for p in BLOG_POSTS[:4]],
        metadatas=[p["metadata"] for p in BLOG_POSTS[:4]],
    )
    print(f"Initial count: {collection.count()}")

    # Update a document
    collection.update(
        ids=["post_001"],
        documents=["Python decorators are functions that wrap other functions. @property, @staticmethod, and @classmethod are built-in decorators. Custom decorators use functools.wraps."],
        metadatas=[{"category": "Python", "difficulty": "intermediate", "year": 2024, "likes": 298}],
    )
    updated = collection.get(ids=["post_001"])
    print(f"\nUpdated post_001: {updated['documents'][0][:80]}...")
    print(f"New likes: {updated['metadatas'][0]['likes']}")

    # Delete a document
    collection.delete(ids=["post_002"])
    print(f"\nAfter deleting post_002: {collection.count()} documents")

    # Upsert (insert or update)
    collection.upsert(
        ids=["post_002", "post_009"],
        documents=[
            "Docker Compose orchestrates multi-container applications. Define services in docker-compose.yml.",
            "GraphQL provides a flexible query language for APIs, allowing clients to request exactly the data they need.",
        ],
        metadatas=[
            {"category": "DevOps", "difficulty": "beginner", "year": 2024, "likes": 200},
            {"category": "Backend", "difficulty": "intermediate", "year": 2024, "likes": 155},
        ],
    )
    print(f"After upsert (re-add post_002 + new post_009): {collection.count()} documents")


# ── 5. Chroma get by metadata (no semantic query) ─────────────────────────────
def demo_metadata_only_query():
    print("\n=== 5. Metadata-Only Queries (No Embedding Needed) ===")

    chroma = chromadb.Client()
    collection = chroma.create_collection(name="meta_query", embedding_function=openai_ef)
    collection.add(
        ids=[p["id"] for p in BLOG_POSTS],
        documents=[p["text"] for p in BLOG_POSTS],
        metadatas=[p["metadata"] for p in BLOG_POSTS],
    )

    # Get all AI/ML posts — no semantic query, just filter
    results = collection.get(where={"category": "AI/ML"})
    print(f"All AI/ML posts: {len(results['ids'])}")
    for doc_id, meta in zip(results["ids"], results["metadatas"]):
        print(f"  {doc_id}: likes={meta['likes']}, difficulty={meta['difficulty']}")

    # Get posts by specific IDs
    results = collection.get(ids=["post_001", "post_008"])
    print(f"\nFetched by ID: {results['ids']}")


if __name__ == "__main__":
    demo_in_memory()
    demo_metadata_filtering()
    demo_persistence()
    demo_update_delete()
    demo_metadata_only_query()
