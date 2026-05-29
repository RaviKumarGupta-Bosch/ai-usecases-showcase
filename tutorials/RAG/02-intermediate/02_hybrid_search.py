"""
RAG Intermediate 02 — Hybrid Search (BM25 + Vector)
=====================================================
Topics covered:
  1. BM25 keyword retrieval
  2. Vector (semantic) retrieval
  3. Hybrid search with RRF (Reciprocal Rank Fusion)
  4. Comparing keyword vs semantic vs hybrid
  5. Score-based weighting

BM25 excels at exact keyword matches.
Vector search excels at semantic similarity.
Hybrid gets the best of both worlds.

Run:
  python 02_hybrid_search.py
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

DOCUMENTS = [
    Document(page_content="Python list comprehension: [x**2 for x in range(10) if x % 2 == 0]", metadata={"topic": "python", "level": "beginner"}),
    Document(page_content="Python decorators use @functools.wraps to preserve function metadata when wrapping functions.", metadata={"topic": "python", "level": "intermediate"}),
    Document(page_content="Python asyncio: use async def and await for coroutines. Run with asyncio.run().", metadata={"topic": "python", "level": "intermediate"}),
    Document(page_content="Python generators use yield to produce values lazily, saving memory for large sequences.", metadata={"topic": "python", "level": "intermediate"}),
    Document(page_content="JavaScript Promises handle async operations. Use .then()/.catch() or async/await syntax.", metadata={"topic": "javascript", "level": "intermediate"}),
    Document(page_content="JavaScript ES6 arrow functions: const add = (a, b) => a + b; are shorter and lexically bind 'this'.", metadata={"topic": "javascript", "level": "beginner"}),
    Document(page_content="SQL window functions: SELECT name, salary, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) FROM employees.", metadata={"topic": "sql", "level": "advanced"}),
    Document(page_content="SQL CTEs (Common Table Expressions) use WITH clause to create readable sub-queries: WITH cte AS (SELECT ...)", metadata={"topic": "sql", "level": "intermediate"}),
    Document(page_content="Docker multi-stage builds reduce image size: use FROM python:3.11 AS builder then FROM python:3.11-slim.", metadata={"topic": "docker", "level": "intermediate"}),
    Document(page_content="Kubernetes HPA (Horizontal Pod Autoscaler) scales pods based on CPU or custom metrics automatically.", metadata={"topic": "kubernetes", "level": "advanced"}),
    Document(page_content="The @dataclass decorator in Python automatically generates __init__, __repr__, and __eq__ methods.", metadata={"topic": "python", "level": "intermediate"}),
    Document(page_content="Python type hints with generics: def get_items(container: list[str]) -> Iterator[str]: ...", metadata={"topic": "python", "level": "intermediate"}),
]


def build_retrievers(docs: list[Document]):
    """Build BM25, vector, and hybrid retrievers."""
    # BM25 — keyword based
    bm25 = BM25Retriever.from_documents(docs, k=3)

    # Vector — semantic
    vs = FAISS.from_documents(docs, embeddings)
    vector = vs.as_retriever(search_kwargs={"k": 3})

    # Hybrid — EnsembleRetriever uses Reciprocal Rank Fusion
    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.4, 0.6],   # 40% BM25 + 60% vector
    )

    return bm25, vector, hybrid


# ── 1. Compare three retrieval methods ───────────────────────────────────────
def demo_compare_retrieval_methods():
    print("\n=== 1. BM25 vs Vector vs Hybrid ===")

    bm25, vector, hybrid = build_retrievers(DOCUMENTS)

    queries = [
        "asyncio coroutines",           # semantic — needs understanding of async
        "@dataclass decorator",         # keyword — exact term present in docs
        "automatic memory management",  # out-of-corpus semantic query
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")

        bm25_docs = bm25.invoke(query)
        vec_docs = vector.invoke(query)
        hyb_docs = hybrid.invoke(query)

        print(f"  BM25   ({len(bm25_docs)} docs): {[d.page_content[:50] for d in bm25_docs[:2]]}")
        print(f"  Vector ({len(vec_docs)} docs): {[d.page_content[:50] for d in vec_docs[:2]]}")
        print(f"  Hybrid ({len(hyb_docs)} docs): {[d.page_content[:50] for d in hyb_docs[:2]]}")


# ── 2. Hybrid search with weight tuning ──────────────────────────────────────
def demo_weight_tuning():
    print("\n=== 2. Weight Tuning for Different Query Types ===")

    bm25 = BM25Retriever.from_documents(DOCUMENTS, k=3)
    vs = FAISS.from_documents(DOCUMENTS, embeddings)
    vector = vs.as_retriever(search_kwargs={"k": 3})

    configs = [
        ("Keyword-heavy  (BM25 0.8, Vector 0.2)", 0.8, 0.2),
        ("Balanced       (BM25 0.5, Vector 0.5)", 0.5, 0.5),
        ("Semantic-heavy (BM25 0.2, Vector 0.8)", 0.2, 0.8),
    ]

    query = "Python lazy evaluation techniques"
    print(f"\nQuery: '{query}'")

    for label, w_bm25, w_vec in configs:
        retriever = EnsembleRetriever(
            retrievers=[bm25, vector],
            weights=[w_bm25, w_vec],
        )
        docs = retriever.invoke(query)
        top = docs[0].page_content[:70] if docs else "No results"
        print(f"\n  {label}")
        print(f"  Top result: {top}...")


# ── 3. Hybrid RAG end-to-end ──────────────────────────────────────────────────
def demo_hybrid_rag():
    print("\n=== 3. Hybrid RAG Pipeline ===")

    bm25 = BM25Retriever.from_documents(DOCUMENTS, k=4)
    vs = FAISS.from_documents(DOCUMENTS, embeddings)
    vector = vs.as_retriever(search_kwargs={"k": 4})
    hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.35, 0.65])

    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY the provided context. If not in context, say so.

Context:
{context}

Question: {question}"""
    )

    chain = (
        {"context": hybrid | (lambda docs: "\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "How do I write a Python function that runs asynchronously?",
        "What SQL feature allows reusable sub-queries?",
        "How can I reduce Docker image size?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {chain.invoke(q)}")


if __name__ == "__main__":
    demo_compare_retrieval_methods()
    demo_weight_tuning()
    demo_hybrid_rag()
