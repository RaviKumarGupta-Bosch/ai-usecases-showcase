"""
LlamaIndex 01-basics — VectorStoreIndex and Basic Querying
===========================================================
Topics covered:
  1. Creating an index from in-memory documents
  2. Querying with the default query engine
  3. Response modes: compact, refine, tree_summarize
  4. Controlling retrieval: top_k, similarity threshold
  5. Inspecting retrieved source nodes

Run:
  python 01_simple_index.py
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# Global settings — used by all LlamaIndex components
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
Settings.chunk_size = 256
Settings.chunk_overlap = 20

# Sample corpus: Python programming guide
PYTHON_DOCS = [
    Document(text="Python uses dynamic typing. Variables don't need type declarations. Type hints are optional annotations: x: int = 42. Use isinstance() for runtime type checking.", metadata={"topic": "types", "level": "beginner"}),
    Document(text="Python functions support default arguments, *args (tuple of positional), and **kwargs (dict of keyword arguments). Lambda creates anonymous one-line functions.", metadata={"topic": "functions", "level": "beginner"}),
    Document(text="Python classes: __init__ is the constructor. self is the instance reference. Use @property for computed attributes. __str__ defines str() representation.", metadata={"topic": "classes", "level": "beginner"}),
    Document(text="Python list operations: append, extend, insert, pop, remove, sort, reverse. Slicing: lst[start:stop:step]. List comprehensions: [x*2 for x in lst if x > 0].", metadata={"topic": "lists", "level": "beginner"}),
    Document(text="Python decorators modify functions. @wraps preserves metadata. Common decorators: @property, @staticmethod, @classmethod, @cache (lru_cache). Stacking decorators applies them bottom-up.", metadata={"topic": "decorators", "level": "intermediate"}),
    Document(text="Python generators use yield to produce values lazily. yield from delegates to sub-generators. Generator expressions: (x**2 for x in range(100)). Much more memory-efficient than lists.", metadata={"topic": "generators", "level": "intermediate"}),
    Document(text="Python async/await: async def defines coroutines, await pauses execution. asyncio.gather() runs coroutines concurrently. asyncio.create_task() schedules background tasks.", metadata={"topic": "async", "level": "intermediate"}),
    Document(text="Python exception handling: try/except/else/finally. Custom exceptions subclass Exception. Use raise from e to chain exceptions. Contextlib.suppress() silences specific exceptions.", metadata={"topic": "exceptions", "level": "beginner"}),
    Document(text="Python dataclasses (@dataclass) auto-generate __init__, __repr__, __eq__. field() customises defaults. frozen=True makes instances immutable. Works well with type hints.", metadata={"topic": "dataclasses", "level": "intermediate"}),
    Document(text="Python testing with pytest: test functions start with test_. assert statements for assertions. Fixtures with @pytest.fixture. Parametrise with @pytest.mark.parametrize. Mock with unittest.mock.", metadata={"topic": "testing", "level": "intermediate"}),
]


def build_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_documents(PYTHON_DOCS)


# ── 1. Basic query ────────────────────────────────────────────────────────────
def demo_basic_query(index: VectorStoreIndex):
    print("\n=== 1. Basic Query ===")

    engine = index.as_query_engine(similarity_top_k=3)

    questions = [
        "How do I create a class in Python?",
        "What is a generator and when should I use one?",
        "How does async/await work?",
    ]

    for q in questions:
        response = engine.query(q)
        print(f"\nQ: {q}")
        print(f"A: {response}")


# ── 2. Response modes ─────────────────────────────────────────────────────────
def demo_response_modes(index: VectorStoreIndex):
    print("\n=== 2. Response Modes ===")

    query = "Explain Python decorators and how to use them"

    for mode in ["compact", "refine", "tree_summarize"]:
        engine = index.as_query_engine(response_mode=mode, similarity_top_k=3)
        response = engine.query(query)
        print(f"\nMode: {mode}")
        print(f"Response: {str(response)[:200]}...")


# ── 3. Source node inspection ─────────────────────────────────────────────────
def demo_source_nodes(index: VectorStoreIndex):
    print("\n=== 3. Inspecting Source Nodes ===")

    engine = index.as_query_engine(similarity_top_k=4)
    response = engine.query("What are the different ways to handle errors in Python?")

    print(f"Answer: {response}\n")
    print(f"Source nodes ({len(response.source_nodes)}):")
    for i, node in enumerate(response.source_nodes, 1):
        print(f"\n  Node {i} [score={node.score:.4f}]:")
        print(f"    Text: {node.text[:100]}...")
        print(f"    Metadata: {node.metadata}")


# ── 4. Retriever with threshold ───────────────────────────────────────────────
def demo_retriever(index: VectorStoreIndex):
    print("\n=== 4. Retriever with Similarity Threshold ===")

    # Use the retriever directly without generation
    retriever = index.as_retriever(
        similarity_top_k=5,
    )

    query = "Python memory efficiency techniques"
    nodes = retriever.retrieve(query)

    print(f"Query: \"{query}\"")
    print(f"Retrieved {len(nodes)} nodes:")
    for n in nodes:
        print(f"  [{n.score:.4f}] [{n.metadata.get('topic', 'unknown')}] {n.text[:80]}...")


# ── 5. Index persistence ──────────────────────────────────────────────────────
def demo_index_persistence():
    print("\n=== 5. Index Persistence (Save & Load) ===")
    import tempfile
    from llama_index.core import StorageContext, load_index_from_storage

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Saving index to: {tmpdir}")

        # Build and save
        idx = VectorStoreIndex.from_documents(PYTHON_DOCS[:4])
        idx.storage_context.persist(persist_dir=tmpdir)
        print("Index saved.")

        # Reload
        storage_ctx = StorageContext.from_defaults(persist_dir=tmpdir)
        loaded_idx = load_index_from_storage(storage_ctx)
        print("Index reloaded.")

        # Query the reloaded index
        engine = loaded_idx.as_query_engine()
        response = engine.query("How do default arguments work in Python functions?")
        print(f"Answer from reloaded index: {response}")


if __name__ == "__main__":
    print("Building VectorStoreIndex from documents...")
    idx = build_index()
    print(f"Index built with {len(PYTHON_DOCS)} documents.")

    demo_basic_query(idx)
    demo_response_modes(idx)
    demo_source_nodes(idx)
    demo_retriever(idx)
    demo_index_persistence()
