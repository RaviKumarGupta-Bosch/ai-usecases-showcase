"""
Vector Databases 04-UseCases — Semantic Search Engine
======================================================
Topics covered:
  1. Building a production-grade semantic search engine
  2. ChromaDB as the vector store with metadata
  3. Query expansion for better recall
  4. Result deduplication and re-ranking
  5. Search analytics and relevance feedback

Run:
  python 01_semantic_search.py
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)

# Knowledge base: developer documentation articles
ARTICLES = [
    {"id": "a001", "title": "Python List Comprehensions",       "text": "List comprehensions provide a concise way to create lists. Syntax: [expr for item in iterable if condition]. They are faster than equivalent for loops.",                              "tags": ["python", "basics"],      "type": "tutorial"},
    {"id": "a002", "title": "Python Generators",                "text": "Generators produce values lazily using yield. They are memory-efficient for large datasets. Generator expressions use () syntax. Use next() or for loop to consume.",                   "tags": ["python", "advanced"],    "type": "tutorial"},
    {"id": "a003", "title": "Async Python with asyncio",        "text": "asyncio enables concurrent I/O in Python. async def defines coroutines, await pauses them. asyncio.run() starts the event loop. gather() runs multiple coroutines concurrently.",       "tags": ["python", "async"],       "type": "tutorial"},
    {"id": "a004", "title": "Python Type Hints Guide",          "text": "Type hints improve code readability and IDE support. Use Optional[T], Union[A,B], List[T], Dict[K,V]. The typing module provides Callable, TypeVar, Generic. Run mypy to check types.", "tags": ["python", "typing"],      "type": "reference"},
    {"id": "a005", "title": "REST API Design Best Practices",   "text": "RESTful APIs use HTTP methods: GET (read), POST (create), PUT (update), DELETE (remove). Use plural nouns for resources. Return appropriate status codes. Version your API.",            "tags": ["api", "backend"],        "type": "guide"},
    {"id": "a006", "title": "FastAPI Tutorial",                 "text": "FastAPI is a Python framework for building APIs with automatic OpenAPI docs. Use Pydantic models for validation. Async support is built in. Deploy with uvicorn or gunicorn.",             "tags": ["api", "python"],         "type": "tutorial"},
    {"id": "a007", "title": "Docker for Developers",            "text": "Docker containers package apps with all dependencies. Dockerfile defines the image. docker-compose.yml orchestrates multiple services. Use volumes for persistent data.",                  "tags": ["devops", "containers"],  "type": "tutorial"},
    {"id": "a008", "title": "Kubernetes Fundamentals",          "text": "Kubernetes orchestrates containers across a cluster. Pods run containers, Deployments manage replicas, Services expose pods. Use kubectl to manage resources.",                            "tags": ["devops", "k8s"],         "type": "tutorial"},
    {"id": "a009", "title": "Git Branching Strategy",           "text": "GitFlow uses feature/, release/, and hotfix/ branches. Trunk-based development uses short-lived feature branches. Always use pull requests for code review.",                             "tags": ["git", "workflow"],       "type": "guide"},
    {"id": "a010", "title": "SQL Query Optimisation",           "text": "Use indexes on columns in WHERE and JOIN clauses. EXPLAIN ANALYZE shows the query plan. Avoid SELECT *. Use CTEs for readability. Partition large tables.",                              "tags": ["sql", "database"],       "type": "reference"},
    {"id": "a011", "title": "PostgreSQL vs MySQL",              "text": "PostgreSQL supports advanced features: JSON, full-text search, window functions, CTEs. MySQL is simpler and faster for read-heavy workloads. PostgreSQL is preferred for complex queries.", "tags": ["sql", "database"],       "type": "guide"},
    {"id": "a012", "title": "Redis Caching Patterns",           "text": "Redis is an in-memory data store used for caching, sessions, and pub/sub. Cache-aside pattern: check cache first, set on miss. Use TTL to expire stale data.",                          "tags": ["database", "caching"],   "type": "tutorial"},
    {"id": "a013", "title": "React Hooks Explained",            "text": "useState manages local state. useEffect handles side effects and lifecycle. useCallback memoises functions. useMemo memoises values. Custom hooks extract reusable logic.",                "tags": ["javascript", "react"],   "type": "tutorial"},
    {"id": "a014", "title": "LLM Prompt Engineering",           "text": "Zero-shot prompts rely on model knowledge. Few-shot includes examples. Chain-of-thought prompts improve reasoning. System messages set model behaviour. Be specific and structured.",       "tags": ["ai", "llm"],             "type": "guide"},
    {"id": "a015", "title": "RAG Architecture Guide",           "text": "RAG combines retrieval with generation. Chunk documents, embed them, store in vector DB. At query time: retrieve relevant chunks, inject into prompt, generate grounded answer.",          "tags": ["ai", "rag"],             "type": "guide"},
]


def build_search_engine():
    """Create and populate the vector store."""
    chroma = chromadb.Client()
    collection = chroma.create_collection("dev_docs", embedding_function=openai_ef)
    collection.add(
        ids=[a["id"] for a in ARTICLES],
        documents=[a["text"] for a in ARTICLES],
        metadatas=[{"title": a["title"], "type": a["type"], "tags": ",".join(a["tags"])} for a in ARTICLES],
    )
    return collection


# ── 1. Basic semantic search ──────────────────────────────────────────────────
def demo_basic_search(collection):
    print("\n=== 1. Basic Semantic Search ===")

    queries = [
        "how to make Python code run faster",
        "deploying applications with containers",
        "building an AI question answering system",
    ]

    for q in queries:
        results = collection.query(query_texts=[q], n_results=3)
        print(f"\nQuery: \"{q}\"")
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            print(f"  [{dist:.4f}] {meta['title']} [{meta['type']}]")


# ── 2. Faceted search (filter by type/tag) ────────────────────────────────────
def demo_faceted_search(collection):
    print("\n=== 2. Faceted Search (Filtered by Category) ===")

    print("\nSearch 'database' — only guides:")
    results = collection.query(
        query_texts=["database query performance"],
        n_results=3,
        where={"type": "guide"},
    )
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  [{dist:.4f}] {meta['title']}")

    print("\nSearch 'async concurrent' — only tutorials:")
    results = collection.query(
        query_texts=["async concurrent programming"],
        n_results=3,
        where={"type": "tutorial"},
    )
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  [{dist:.4f}] {meta['title']}")


# ── 3. Query expansion ────────────────────────────────────────────────────────
def demo_query_expansion(collection):
    print("\n=== 3. Query Expansion (LLM-powered) ===")
    print("Generates alternative query formulations to improve recall.\n")

    original_query = "how to speed up my code"

    prompt = f"""Given the search query: "{original_query}"
Generate 3 alternative phrasings of this query that might match relevant documentation.
Return as JSON array of strings. Example: ["phrasing 1", "phrasing 2", "phrasing 3"]"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    expanded = json.loads(response.choices[0].message.content)
    queries = [original_query] + (expanded.get("queries") or list(expanded.values())[0])[:3]

    print(f"Original: {original_query}")
    print(f"Expanded to {len(queries)} queries:")
    for q in queries:
        print(f"  • {q}")

    # Collect results from all query variants
    seen_ids: set[str] = set()
    all_results = []
    for q in queries:
        results = collection.query(query_texts=[q], n_results=3)
        for doc_id, meta, dist in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append((dist, meta["title"]))

    all_results.sort()
    print(f"\nMerged unique results ({len(all_results)}):")
    for dist, title in all_results[:5]:
        print(f"  [{dist:.4f}] {title}")


# ── 4. Search with LLM summarisation ─────────────────────────────────────────
def demo_search_with_answer(collection):
    print("\n=== 4. Search → LLM Answer Generation ===")

    question = "What is the best way to handle asynchronous operations in Python?"
    print(f"Question: {question}\n")

    results = collection.query(query_texts=[question], n_results=4)
    context_parts = []
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        context_parts.append(f"[{meta['title']}]\n{doc}")
    context = "\n\n".join(context_parts)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context. Cite article titles."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    print(f"Answer: {response.choices[0].message.content}")
    print(f"\nSources used:")
    for meta in results["metadatas"][0]:
        print(f"  • {meta['title']}")


if __name__ == "__main__":
    col = build_search_engine()
    demo_basic_search(col)
    demo_faceted_search(col)
    demo_query_expansion(col)
    demo_search_with_answer(col)
