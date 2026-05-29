"""
RAG Intermediate 01 — Advanced Retrieval Strategies
======================================================
Topics covered:
  1. Multi-query retrieval — generate multiple query variants
  2. MMR (Maximal Marginal Relevance) — reduce redundancy
  3. Self-query retriever — natural language metadata filtering
  4. Contextual compression — trim retrieved docs to just the answer
  5. Ensemble retriever — combine multiple retrievers

Run:
  python 01_advanced_retrieval.py
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

CORPUS = [
    Document(page_content="Python is a dynamically-typed, interpreted language known for its readability. It supports OOP, functional, and procedural paradigms. Created by Guido van Rossum in 1991.", metadata={"language": "Python", "year": 1991, "type": "language"}),
    Document(page_content="JavaScript runs in web browsers and Node.js servers. It is the only language natively supported by all web browsers. It supports async/await for non-blocking I/O.", metadata={"language": "JavaScript", "year": 1995, "type": "language"}),
    Document(page_content="Rust is a systems language focused on memory safety without a garbage collector. It uses an ownership system to prevent data races at compile time. Created by Mozilla.", metadata={"language": "Rust", "year": 2010, "type": "language"}),
    Document(page_content="Go (Golang) was designed at Google for simplicity and fast compilation. It has built-in concurrency primitives (goroutines, channels) and a rich standard library.", metadata={"language": "Go", "year": 2009, "type": "language"}),
    Document(page_content="TypeScript is a typed superset of JavaScript developed by Microsoft. It adds static typing, interfaces, generics, and better IDE support to JavaScript.", metadata={"language": "TypeScript", "year": 2012, "type": "language"}),
    Document(page_content="Docker is a containerisation platform that packages applications and their dependencies into containers. Containers share the host OS kernel, making them more lightweight than VMs.", metadata={"tool": "Docker", "year": 2013, "type": "tool"}),
    Document(page_content="Kubernetes (K8s) is a container orchestration system. It automates deployment, scaling, and management of containerised applications across clusters of machines.", metadata={"tool": "Kubernetes", "year": 2014, "type": "tool"}),
    Document(page_content="PostgreSQL is a powerful, open-source relational database. It supports JSON, full-text search, window functions, CTEs, and advanced indexing strategies.", metadata={"tool": "PostgreSQL", "year": 1996, "type": "database"}),
]


def build_store() -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(CORPUS)
    return FAISS.from_documents(chunks, embeddings)


# ── 1. Multi-query retriever ──────────────────────────────────────────────────
def demo_multi_query():
    print("\n=== 1. Multi-Query Retrieval ===")
    print("(Generates multiple query variants to cast a wider retrieval net)")

    vs = build_store()
    retriever = vs.as_retriever(search_kwargs={"k": 2})

    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )

    query = "What programming languages are good for web development?"
    print(f"\nOriginal query: {query}")

    docs = multi_retriever.invoke(query)
    print(f"Retrieved {len(docs)} unique docs (more than single-query k=2):")
    for doc in docs:
        lang = doc.metadata.get("language") or doc.metadata.get("tool", "?")
        print(f"  - {lang}: {doc.page_content[:70]}...")


# ── 2. MMR (Maximal Marginal Relevance) ───────────────────────────────────────
def demo_mmr():
    print("\n=== 2. MMR — Reduce Redundancy in Retrieved Chunks ===")

    vs = build_store()

    query = "Which languages were created by large tech companies?"
    print(f"\nQuery: {query}")

    # Standard similarity search — may return redundant chunks
    standard = vs.similarity_search(query, k=4)
    print("\nStandard similarity (may have duplicates):")
    for doc in standard:
        print(f"  {doc.page_content[:80]}...")

    # MMR balances relevance AND diversity
    mmr_docs = vs.max_marginal_relevance_search(query, k=4, fetch_k=8, lambda_mult=0.5)
    print("\nMMR results (diverse and relevant):")
    for doc in mmr_docs:
        print(f"  {doc.page_content[:80]}...")


# ── 3. Contextual compression ─────────────────────────────────────────────────
def demo_contextual_compression():
    print("\n=== 3. Contextual Compression ===")
    print("(Extracts only the relevant portion from each retrieved chunk)")

    vs = build_store()
    base_retriever = vs.as_retriever(search_kwargs={"k": 3})

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    query = "Which languages support concurrency?"
    print(f"\nQuery: {query}")

    # Without compression
    raw_docs = base_retriever.invoke(query)
    print("\nRaw chunks (full text):")
    for doc in raw_docs:
        print(f"  {doc.page_content}")

    # With compression — only the relevant sentence is kept
    compressed = compression_retriever.invoke(query)
    print("\nAfter compression (only relevant parts):")
    for doc in compressed:
        print(f"  {doc.page_content}")


# ── 4. Full advanced RAG pipeline ─────────────────────────────────────────────
def demo_advanced_rag_pipeline():
    print("\n=== 4. Advanced RAG Pipeline (MMR + Compression) ===")

    vs = build_store()

    # MMR retriever for diversity
    mmr_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.6},
    )

    # Compress to extract only relevant parts
    compressor = LLMChainExtractor.from_llm(llm)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mmr_retriever,
    )

    prompt = ChatPromptTemplate.from_template(
        """Use ONLY the context to answer. Be concise.
Context: {context}
Question: {question}"""
    )

    chain = (
        {"context": final_retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What languages have memory safety as a design goal?",
        "Which tools help with running applications in containers?",
        "What databases support JSON natively?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {chain.invoke(q)}")


if __name__ == "__main__":
    demo_multi_query()
    demo_mmr()
    demo_contextual_compression()
    demo_advanced_rag_pipeline()
