"""
03 - Embeddings & Vector Stores
=================================
Embeddings turn text into dense numerical vectors.
Vector stores index those vectors for fast similarity search.

Topics covered:
  1. OpenAIEmbeddings — embed text and documents
  2. FAISS — in-memory, fast, no server required
  3. Chroma — persistent on-disk vector store
  4. Similarity search — top-k by cosine distance
  5. Similarity search with scores
  6. Max Marginal Relevance (MMR) — diversity-aware retrieval
  7. Metadata filtering
  8. Save and reload FAISS index
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample knowledge base — AI topics
DOCUMENTS = [
    Document(page_content="LangChain is a framework for building LLM-powered applications. It provides chains, agents, memory, and retrieval components.", metadata={"source": "langchain_docs", "topic": "framework"}),
    Document(page_content="LangGraph extends LangChain with graph-based workflows for complex agentic behaviour. Nodes represent computation; edges define flow.", metadata={"source": "langgraph_docs", "topic": "agents"}),
    Document(page_content="LangSmith is an observability platform for LLM applications. It enables tracing, evaluation, and dataset management.", metadata={"source": "langsmith_docs", "topic": "observability"}),
    Document(page_content="RAG (Retrieval-Augmented Generation) combines a retriever with an LLM. The retriever fetches relevant documents; the LLM generates the answer.", metadata={"source": "rag_paper", "topic": "retrieval"}),
    Document(page_content="Vector databases store high-dimensional embeddings. Similarity search finds the nearest neighbours in embedding space.", metadata={"source": "vector_db_guide", "topic": "storage"}),
    Document(page_content="FAISS (Facebook AI Similarity Search) is an efficient library for similarity search of dense vectors. It operates entirely in memory.", metadata={"source": "faiss_docs", "topic": "storage"}),
    Document(page_content="Chroma is an open-source embedding database optimised for AI applications. It supports persistent storage and metadata filtering.", metadata={"source": "chroma_docs", "topic": "storage"}),
    Document(page_content="OpenAI embeddings convert text to dense vector representations using models like text-embedding-3-small and text-embedding-3-large.", metadata={"source": "openai_docs", "topic": "embeddings"}),
    Document(page_content="Prompt engineering involves crafting effective instructions for LLMs. Techniques include few-shot prompting, chain-of-thought, and system prompts.", metadata={"source": "prompt_guide", "topic": "prompting"}),
    Document(page_content="ReAct is an agent pattern that interleaves reasoning (thought) with acting (tool use). It produces more interpretable and reliable agent behaviour.", metadata={"source": "react_paper", "topic": "agents"}),
]


# ── 1. Embedding text ────────────────────────────────────────────────────────
def demo_embed_text():
    query = "How does LangChain work?"
    vector = embeddings.embed_query(query)

    print("=== 1. Embed Text ===")
    print(f"Query    : {query}")
    print(f"Dimension: {len(vector)}")
    print(f"First 5  : {[round(v, 4) for v in vector[:5]]}")

    # Embed multiple texts at once (batch)
    texts = ["FAISS is fast", "Chroma is persistent", "Both are vector stores"]
    batch_vectors = embeddings.embed_documents(texts)
    print(f"Batch embedded {len(batch_vectors)} texts, each {len(batch_vectors[0])}D")


# ── 2. FAISS — build and search ──────────────────────────────────────────────
def demo_faiss_basic():
    db = FAISS.from_documents(DOCUMENTS, embeddings)

    print("\n=== 2. FAISS Basic Search ===")
    results = db.similarity_search("what is retrieval augmented generation?", k=3)
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content[:80]}...")


# ── 3. Similarity search with scores ────────────────────────────────────────
def demo_similarity_scores():
    db = FAISS.from_documents(DOCUMENTS, embeddings)

    print("\n=== 3. Similarity Search with Scores ===")
    results = db.similarity_search_with_score("vector database options", k=4)
    for doc, score in results:
        # Lower L2 distance = more similar
        print(f"  score={score:.4f}  [{doc.metadata['topic']}] {doc.page_content[:70]}...")


# ── 4. Max Marginal Relevance (MMR) — diversity-aware ────────────────────────
def demo_mmr():
    """
    MMR balances relevance with diversity.
    Avoids returning near-duplicate chunks.
    fetch_k: initial candidates, k: final diverse set returned.
    """
    db = FAISS.from_documents(DOCUMENTS, embeddings)

    print("\n=== 4. Max Marginal Relevance ===")
    results = db.max_marginal_relevance_search(
        "vector stores and databases",
        k=3,
        fetch_k=8,
        lambda_mult=0.5,   # 0 = max diversity, 1 = max relevance
    )
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content[:80]}...")


# ── 5. Metadata filtering ────────────────────────────────────────────────────
def demo_metadata_filter():
    db = FAISS.from_documents(DOCUMENTS, embeddings)

    print("\n=== 5. Metadata Filtering ===")
    # Only return documents where topic == "agents"
    results = db.similarity_search(
        "how do agents work?",
        k=5,
        filter={"topic": "agents"},
    )
    print(f"Found {len(results)} agent-related document(s):")
    for doc in results:
        print(f"  [{doc.metadata['source']}] {doc.page_content[:80]}...")


# ── 6. Save and reload FAISS index ──────────────────────────────────────────
def demo_faiss_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "faiss_index"

        # Build and save
        db = FAISS.from_documents(DOCUMENTS, embeddings)
        db.save_local(str(index_path))
        print(f"\n=== 6. FAISS Persistence ===")
        print(f"Saved index to {index_path}")

        # Reload and search
        db2 = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        results = db2.similarity_search("LangSmith tracing", k=2)
        print("Reloaded and searched:")
        for doc in results:
            print(f"  {doc.page_content[:80]}...")


# ── 7. Add documents to an existing index ───────────────────────────────────
def demo_faiss_add_documents():
    db = FAISS.from_documents(DOCUMENTS[:5], embeddings)
    print(f"\n=== 7. Add Documents to FAISS ===")
    print(f"Initial index size: {db.index.ntotal} vectors")

    new_docs = [
        Document(
            page_content="Pinecone is a managed vector database cloud service offering low-latency similarity search at scale.",
            metadata={"source": "pinecone_docs", "topic": "storage"},
        ),
        Document(
            page_content="Weaviate is an open-source vector database with built-in ML models and a GraphQL API.",
            metadata={"source": "weaviate_docs", "topic": "storage"},
        ),
    ]
    db.add_documents(new_docs)
    print(f"After adding 2 docs: {db.index.ntotal} vectors")

    results = db.similarity_search("cloud vector database", k=2)
    for doc in results:
        print(f"  {doc.page_content[:80]}...")


# ── 8. Chroma — persistent vector store ──────────────────────────────────────
def demo_chroma():
    with tempfile.TemporaryDirectory() as tmp:
        db = Chroma.from_documents(
            documents=DOCUMENTS,
            embedding=embeddings,
            persist_directory=str(tmp),
            collection_name="tutorial_docs",
        )

        print("\n=== 8. Chroma Vector Store ===")
        print(f"Collection count: {db._collection.count()}")

        results = db.similarity_search("agent reasoning patterns", k=2)
        for doc in results:
            print(f"  [{doc.metadata['topic']}] {doc.page_content[:80]}...")

        # Metadata filter with Chroma uses a `where` dict
        filtered = db.similarity_search(
            "database",
            k=5,
            filter={"topic": "storage"},
        )
        print(f"\nFiltered to 'storage' topic: {len(filtered)} result(s)")
        for doc in filtered:
            print(f"  {doc.metadata['source']}")


if __name__ == "__main__":
    demo_embed_text()
    demo_faiss_basic()
    demo_similarity_scores()
    demo_mmr()
    demo_metadata_filter()
    demo_faiss_persistence()
    demo_faiss_add_documents()
    demo_chroma()
