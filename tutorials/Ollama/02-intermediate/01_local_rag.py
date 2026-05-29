"""
Ollama 02-Intermediate — Fully Local RAG
==========================================
Topics covered:
  1. Build a local knowledge base from plain-text documents
  2. Embed documents with OllamaEmbeddings (nomic-embed-text)
  3. Store and search with FAISS — zero cloud dependency
  4. Retrieve relevant chunks and generate answers with OllamaLLM
  5. Evaluate retrieval quality (MMR vs similarity search)

This entire pipeline runs offline. No API keys. No internet needed after
model download.

Prerequisites:
  - Ollama running: `ollama serve`
  - Models pulled:
      ollama pull llama3.2
      ollama pull nomic-embed-text

Run:
  python 01_local_rag.py
"""

import os
import sys
import math
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MODEL       = os.getenv("OLLAMA_MODEL",       "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")

# ── Knowledge base ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    # Python
    Document(page_content="Python is an interpreted, high-level, general-purpose language created by Guido van Rossum in 1991.",   metadata={"topic": "python", "subtopic": "history"}),
    Document(page_content="Python uses indentation to define code blocks instead of curly braces.",                                 metadata={"topic": "python", "subtopic": "syntax"}),
    Document(page_content="Python's package manager pip allows installing third-party libraries from PyPI.",                        metadata={"topic": "python", "subtopic": "tooling"}),
    Document(page_content="Virtual environments in Python isolate project dependencies using venv or conda.",                        metadata={"topic": "python", "subtopic": "tooling"}),
    Document(page_content="Python decorators are functions that modify the behaviour of other functions using the @ syntax.",        metadata={"topic": "python", "subtopic": "advanced"}),
    Document(page_content="Python's GIL (Global Interpreter Lock) prevents multiple threads from executing Python code at once.",   metadata={"topic": "python", "subtopic": "concurrency"}),
    Document(page_content="Async/await in Python enables cooperative concurrency without threads, using the asyncio library.",      metadata={"topic": "python", "subtopic": "concurrency"}),
    # Machine Learning
    Document(page_content="Machine learning is a subset of AI where models learn patterns from data without explicit programming.", metadata={"topic": "ml", "subtopic": "basics"}),
    Document(page_content="Supervised learning trains models on labelled examples to predict outputs for unseen inputs.",           metadata={"topic": "ml", "subtopic": "types"}),
    Document(page_content="Unsupervised learning discovers hidden patterns in unlabelled data, e.g. clustering.",                   metadata={"topic": "ml", "subtopic": "types"}),
    Document(page_content="Gradient descent is an optimisation algorithm that minimises a loss function by adjusting model weights.",metadata={"topic": "ml", "subtopic": "training"}),
    Document(page_content="Overfitting occurs when a model memorises training data and fails to generalise to new data.",           metadata={"topic": "ml", "subtopic": "problems"}),
    Document(page_content="Cross-validation splits data into folds to give a more reliable estimate of model performance.",        metadata={"topic": "ml", "subtopic": "evaluation"}),
    # DevOps
    Document(page_content="Docker packages applications with all dependencies into portable containers.",                           metadata={"topic": "devops", "subtopic": "containers"}),
    Document(page_content="A Dockerfile defines the instructions to build a Docker image layer by layer.",                          metadata={"topic": "devops", "subtopic": "containers"}),
    Document(page_content="Kubernetes (k8s) automates deployment, scaling, and management of containerised applications.",         metadata={"topic": "devops", "subtopic": "orchestration"}),
    Document(page_content="CI/CD pipelines automate testing and deployment, reducing manual steps and human error.",                metadata={"topic": "devops", "subtopic": "automation"}),
    Document(page_content="Infrastructure as Code (IaC) tools like Terraform manage cloud resources via configuration files.",     metadata={"topic": "devops", "subtopic": "iac"}),
]


def build_vectorstore(docs: list[Document], embeddings: OllamaEmbeddings) -> FAISS:
    print(f"  Embedding {len(docs)} documents with '{EMBED_MODEL}'...")
    return FAISS.from_documents(docs, embeddings)


def rag_answer(question: str, vectorstore: FAISS, llm: OllamaLLM, k: int = 3) -> tuple[str, list[Document]]:
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n".join(f"- {d.page_content}" for d in docs)

    prompt = PromptTemplate.from_template(
        "You are a knowledgeable assistant. Use ONLY the context below to answer the question.\n"
        "If the context does not contain enough information, say 'I don't know based on the provided context'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (2-3 sentences):"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer.strip(), docs


# ── 1. Build index ────────────────────────────────────────────────────────────
def demo_build_index(embeddings, llm):
    print("\n=== 1. Build Local Knowledge Base ===")
    vectorstore = build_vectorstore(KNOWLEDGE_BASE, embeddings)
    print(f"  Index built. {len(KNOWLEDGE_BASE)} documents indexed.")
    return vectorstore


# ── 2. Basic RAG Q&A ──────────────────────────────────────────────────────────
def demo_basic_rag(vectorstore, llm):
    print("\n=== 2. Basic RAG — Question & Answer ===")
    questions = [
        "How does Python handle concurrency?",
        "What is overfitting in machine learning?",
        "How does Kubernetes help with application deployment?",
    ]
    for q in questions:
        answer, sources = rag_answer(q, vectorstore, llm)
        print(f"\n  Q: {q}")
        print(f"  A: {answer}")
        print(f"  Sources ({len(sources)}):")
        for doc in sources:
            topic = doc.metadata.get("topic", "?")
            subtopic = doc.metadata.get("subtopic", "?")
            print(f"    [{topic}/{subtopic}] {doc.page_content[:80]}...")


# ── 3. Similarity search with scores ─────────────────────────────────────────
def demo_similarity_scores(vectorstore):
    print("\n=== 3. Similarity Search with Scores ===")
    query = "How are Python packages installed?"
    results = vectorstore.similarity_search_with_score(query, k=5)

    print(f"  Query: {query}\n  Top 5 matches (lower L2 distance = more similar):")
    for doc, score in results:
        topic = doc.metadata.get("topic", "?")
        print(f"    [{score:.4f}] [{topic}] {doc.page_content[:90]}")


# ── 4. MMR retrieval (diversity) ──────────────────────────────────────────────
def demo_mmr_retrieval(vectorstore):
    print("\n=== 4. MMR Retrieval — Diverse Results ===")
    query = "What are Python best practices?"

    print("  [Standard similarity — may return near-duplicate docs]")
    sim_docs = vectorstore.similarity_search(query, k=3)
    for doc in sim_docs:
        print(f"    {doc.page_content[:90]}")

    print("\n  [MMR — maximises relevance AND diversity]")
    mmr_docs = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)
    for doc in mmr_docs:
        print(f"    {doc.page_content[:90]}")


# ── 5. Out-of-scope question (graceful handling) ──────────────────────────────
def demo_out_of_scope(vectorstore, llm):
    print("\n=== 5. Out-of-Scope Question Handling ===")
    question = "What is the capital of France?"
    answer, sources = rag_answer(question, vectorstore, llm)
    print(f"  Q: {question}")
    print(f"  A: {answer}")
    print("  (Knowledge base has no geography docs — model should say it doesn't know)")


if __name__ == "__main__":
    print("Ollama 02-Intermediate — Fully Local RAG")
    print("=" * 45)
    print(f"  LLM:        {MODEL}")
    print(f"  Embeddings: {EMBED_MODEL}")
    print("  No API keys required — fully offline pipeline\n")

    try:
        import ollama as _ollama
        _ollama.list()
    except Exception:
        print("  ERROR: Cannot connect to Ollama. Start with: ollama serve")
        sys.exit(1)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)
    llm = OllamaLLM(model=MODEL, temperature=0, base_url=BASE_URL)

    vectorstore = demo_build_index(embeddings, llm)
    demo_basic_rag(vectorstore, llm)
    demo_similarity_scores(vectorstore)
    demo_mmr_retrieval(vectorstore)
    demo_out_of_scope(vectorstore, llm)
