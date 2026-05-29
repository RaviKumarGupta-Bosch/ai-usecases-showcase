"""
Use Case 02 — RAG Document Q&A System
========================================
A retrieval-augmented generation (RAG) system that:
- Ingests multiple text documents with FAISS and source metadata
- Supports conversational Q&A with follow-up questions
- Returns answers with [Source: filename] citations
- Uses history-aware retrieval (rewrites questions using chat history)

Components:
  - create_history_aware_retriever  — query rewriting with context
  - create_retrieval_chain          — full RAG pipeline
  - RunnableWithMessageHistory      — conversation memory
  - FAISS                           — vector store with metadata
  - Source citation formatting

Run:
  python main.py
"""

import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ── Knowledge base (simulated multi-document corpus) ─────────────────────────
DOCUMENTS = {
    "python_guide.txt": """
# Python Programming Guide

## Introduction
Python is a high-level, general-purpose programming language emphasising code readability.
Created by Guido van Rossum and first released in 1991, it supports multiple programming
paradigms: procedural, object-oriented, and functional.

## Key Features
- Dynamic typing and automatic memory management
- Comprehensive standard library ("batteries included")
- Interactive interpreter (REPL)
- Cross-platform compatibility
- Extensive third-party ecosystem (PyPI has 500k+ packages)

## Data Types
Python built-in types include: int, float, complex, str, list, tuple, dict, set, frozenset, bool, bytes.
Type hints (PEP 484) allow optional static typing with tools like mypy.

## Performance
Python is interpreted and slower than C/Java. Options for improvement:
- PyPy: JIT-compiled Python interpreter (4-8x faster on average)
- Cython: compile Python to C extensions
- Numba: JIT compiler for numerical code
- multiprocessing: bypass the GIL for CPU-bound tasks
""",
    "machine_learning.txt": """
# Machine Learning Reference

## What is Machine Learning?
Machine learning (ML) is a subset of AI where algorithms learn patterns from data
without being explicitly programmed. Tom Mitchell's formal definition:
"A computer program learns from experience E with respect to task T and performance P,
if its performance on T improves with experience E."

## Types of Machine Learning
1. Supervised Learning — labelled training data (classification, regression)
   Examples: linear regression, SVM, random forests, neural networks
2. Unsupervised Learning — unlabelled data, find patterns (clustering, dimensionality reduction)
   Examples: k-means, DBSCAN, PCA, autoencoders
3. Reinforcement Learning — agent learns from environment rewards
   Examples: Q-learning, PPO, AlphaGo

## Key Algorithms
- Linear/Logistic Regression: statistical baseline, highly interpretable
- Decision Trees & Random Forests: ensemble, handles non-linearity
- Gradient Boosting (XGBoost, LightGBM): state-of-art for tabular data
- Neural Networks: universal function approximators, deep learning foundation
- Transformer: attention-based architecture, dominates NLP and vision

## Evaluation Metrics
Classification: accuracy, precision, recall, F1, ROC-AUC
Regression: MAE, MSE, RMSE, R²
""",
    "cloud_computing.txt": """
# Cloud Computing Overview

## Definition
Cloud computing delivers computing services (servers, storage, databases, networking,
software, analytics) over the internet ("the cloud") for flexible resources and economies of scale.

## Service Models
- IaaS (Infrastructure as a Service): virtual machines, storage, networking
  Providers: AWS EC2, Azure VMs, Google Compute Engine
- PaaS (Platform as a Service): runtime, middleware, databases
  Providers: AWS Elastic Beanstalk, Azure App Service, Heroku
- SaaS (Software as a Service): end-user applications
  Examples: Gmail, Salesforce, Slack, Zoom

## Deployment Models
- Public Cloud: shared infrastructure, pay-as-you-go (AWS, Azure, GCP)
- Private Cloud: dedicated infrastructure, on-premises or hosted
- Hybrid Cloud: combination of public and private
- Multi-cloud: use multiple cloud providers to avoid lock-in

## Key Benefits
1. Scalability: scale up/down on demand
2. Cost efficiency: no upfront hardware investment, pay per use
3. Reliability: built-in redundancy, 99.9%+ SLA
4. Global reach: data centres worldwide
5. Security: enterprise-grade compliance (ISO 27001, SOC 2, GDPR)

## Major Providers
AWS (31% market share), Microsoft Azure (25%), Google Cloud (12%)
""",
    "database_systems.txt": """
# Database Systems Reference

## Relational Databases (RDBMS)
Use structured tables with SQL. ACID transactions ensure data integrity.
Popular: PostgreSQL, MySQL, Oracle Database, SQL Server.

Key concepts: normalization (1NF, 2NF, 3NF), indexes, foreign keys,
joins (INNER, LEFT, RIGHT, FULL OUTER, CROSS), stored procedures, views, triggers.

## NoSQL Databases
Designed for scalability and flexible schemas:
- Document stores: MongoDB, CouchDB — JSON-like documents
- Key-value stores: Redis, DynamoDB — fast lookups, caching
- Wide-column: Cassandra, HBase — time-series, analytics
- Graph: Neo4j, Amazon Neptune — relationship-heavy data

## Vector Databases
Optimised for storing and searching high-dimensional vectors (embeddings).
Used in AI/ML for semantic search and RAG applications.
Popular: Pinecone, Weaviate, Qdrant, Chroma, FAISS (library, not a DB server).

## CAP Theorem
A distributed database can guarantee at most 2 of 3:
- Consistency: all nodes see the same data simultaneously
- Availability: every request receives a response
- Partition Tolerance: system works despite network partitions

## Query Optimisation
1. Use indexes on frequently queried columns
2. Avoid SELECT * — specify needed columns
3. Use EXPLAIN/EXPLAIN ANALYZE to inspect query plans
4. Partition large tables by date or category
5. Use connection pooling (pgBouncer, HikariCP)
""",
}


# ── Build FAISS vector store ───────────────────────────────────────────────────
def build_vectorstore(documents: dict[str, str]) -> FAISS:
    """Convert document dict to Document objects and build FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_chunks = []
    for filename, content in documents.items():
        doc = Document(page_content=content, metadata={"source": filename})
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["source"] = filename  # preserve source in each chunk
        all_chunks.extend(chunks)

    print(f"  Indexed {len(all_chunks)} chunks from {len(documents)} documents")
    return FAISS.from_documents(all_chunks, embeddings)


# ── Format docs with source citations ────────────────────────────────────────
def format_docs_with_sources(docs: list[Document]) -> str:
    """Format retrieved documents with source filenames."""
    parts = []
    seen_sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content.strip()}")
        seen_sources.add(source)
    return "\n\n---\n\n".join(parts)


# ── Build the RAG pipeline ────────────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Step 1: History-aware retriever — rewrites the question using chat history
    history_aware_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human",
         "Given the conversation history, reformulate the question as a standalone question "
         "that can be understood without the history context. "
         "If it's already standalone, return it unchanged."),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_aware_prompt
    )

    # Step 2: QA chain with source-aware context
    qa_system_prompt = """You are an expert knowledge assistant. Answer questions based ONLY on the provided context.

Rules:
- Answer concisely and accurately (2-4 sentences)
- Always mention the source document(s) in your answer: [Source: filename]
- If the answer is not in the context, say "I don't have information about that in my knowledge base."
- Do not make up information

Context:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Step 3: Full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Step 4: Wrap with session memory
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag, get_session_history


# ── Run a Q&A session ─────────────────────────────────────────────────────────
def ask(chain, question: str, session_id: str) -> str:
    """Send a question and return the answer with source citations."""
    result = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    return result["answer"]


def run_demo():
    print("=" * 60)
    print("RAG Document Q&A System")
    print("=" * 60)
    print("\nLoading knowledge base...")

    vectorstore = build_vectorstore(DOCUMENTS)
    rag_chain, get_history = build_rag_chain(vectorstore)

    session_id = "demo-session"

    qa_pairs = [
        ("python",   "What programming paradigms does Python support?"),
        ("ml",       "What are the three types of machine learning?"),
        ("followup", "What algorithms are typically used in the first type?"),  # follow-up
        ("cloud",    "What are the three cloud service models?"),
        ("db",       "What is the CAP theorem?"),
        ("followup2","What are the recommendations for query optimisation?"),   # follow-up
        ("mixed",    "How do vector databases relate to Python and ML?"),       # multi-doc
        ("unknown",  "What is the current price of Bitcoin?"),                  # out-of-scope
    ]

    for label, question in qa_pairs:
        print(f"\nQ [{label}]: {question}")
        answer = ask(rag_chain, question, session_id)
        print(f"A: {answer}")

    history = get_history(session_id)
    print(f"\n{'='*60}")
    print(f"Conversation turns: {len(history.messages) // 2}")


def run_interactive():
    """Interactive CLI Q&A session."""
    print("=" * 60)
    print("RAG Document Q&A — Interactive Mode")
    print(f"Knowledge base: {list(DOCUMENTS.keys())}")
    print("Type 'quit' to exit | 'clear' to reset history")
    print("=" * 60)

    vectorstore = build_vectorstore(DOCUMENTS)
    rag_chain, get_history = build_rag_chain(vectorstore)
    session_id = "interactive"

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "clear":
            get_history(session_id).clear()
            print("History cleared.")
            continue
        if not question:
            continue

        answer = ask(rag_chain, question, session_id)
        print(f"AI: {answer}")


if __name__ == "__main__":
    run_demo()
    # Uncomment for interactive mode:
    # run_interactive()
