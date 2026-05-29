"""
Ollama 01-Basics — LangChain Integration
==========================================
Topics covered:
  1. OllamaLLM — drop-in LangChain LLM using local Ollama
  2. OllamaEmbeddings — local embeddings (no OpenAI key needed)
  3. LCEL chain: PromptTemplate | OllamaLLM | StrOutputParser
  4. Chaining multiple steps (translate → summarise)
  5. FAISS vector store with Ollama embeddings (fully local)

Prerequisites:
  - Ollama running: `ollama serve`
  - Models pulled: `ollama pull llama3.2` and `ollama pull nomic-embed-text`
  - pip install langchain-ollama langchain-community faiss-cpu

Run:
  python 03_langchain_integration.py
"""

import os
import sys
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

MODEL       = os.getenv("OLLAMA_MODEL",       "llama3.2")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")


def get_llm(temperature: float = 0) -> OllamaLLM:
    return OllamaLLM(model=MODEL, temperature=temperature, base_url=BASE_URL)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)


# ── 1. Basic OllamaLLM invoke ─────────────────────────────────────────────────
def demo_basic_llm():
    print("\n=== 1. OllamaLLM — Basic Invoke ===")
    llm = get_llm()
    question = "What is the difference between a process and a thread?"
    print(f"  Q: {question}")
    answer = llm.invoke(question)
    print(f"  A: {answer.strip()[:300]}")


# ── 2. OllamaEmbeddings ───────────────────────────────────────────────────────
def demo_embeddings():
    print("\n=== 2. OllamaEmbeddings — Local Vectors ===")
    embeddings = get_embeddings()

    texts = [
        "Python is a high-level programming language.",
        "Machine learning models learn from data.",
        "The Eiffel Tower is in Paris, France.",
    ]

    vectors = embeddings.embed_documents(texts)
    print(f"  Embedded {len(vectors)} documents")
    print(f"  Vector dimension: {len(vectors[0])}")

    query_vec = embeddings.embed_query("What programming language is popular for AI?")
    print(f"  Query vector dimension: {len(query_vec)}")

    # Cosine similarity (manual)
    import math
    def cosine(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0

    print("\n  Similarity to query 'What programming language is popular for AI?':")
    for text, vec in zip(texts, vectors):
        sim = cosine(query_vec, vec)
        print(f"    {sim:.3f}  {text}")


# ── 3. LCEL chain ─────────────────────────────────────────────────────────────
def demo_lcel_chain():
    print("\n=== 3. LCEL Chain: PromptTemplate | LLM | Parser ===")
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Answer in exactly 2 sentences.\n\nQuestion: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    questions = [
        "What is Docker?",
        "What is a REST API?",
    ]
    for q in questions:
        answer = chain.invoke({"question": q})
        print(f"\n  Q: {q}")
        print(f"  A: {answer.strip()}")


# ── 4. Multi-step chain ───────────────────────────────────────────────────────
def demo_multi_step_chain():
    print("\n=== 4. Multi-Step Chain (summarise → bullet points) ===")
    llm = get_llm()

    # Step 1: summarise
    summarise_prompt = PromptTemplate.from_template(
        "Summarise the following text in 2 sentences:\n\n{text}"
    )
    # Step 2: convert to bullets
    bullets_prompt = PromptTemplate.from_template(
        "Convert this summary into 3 concise bullet points:\n\n{summary}"
    )

    summarise_chain = summarise_prompt | llm | StrOutputParser()
    bullets_chain   = bullets_prompt   | llm | StrOutputParser()

    # Compose: text → summary → bullets
    full_chain = (
        summarise_chain
        | (lambda summary: {"summary": summary})
        | bullets_chain
    )

    article = """
    Transformer models have revolutionised natural language processing. Introduced in 2017 
    by Vaswani et al. in the paper 'Attention Is All You Need', the transformer architecture 
    uses self-attention mechanisms to process sequences in parallel rather than sequentially. 
    This made training significantly faster and enabled models to learn long-range dependencies 
    more effectively. BERT, GPT, and T5 are all built on this architecture. Today, transformers 
    are applied in computer vision, audio, protein folding, and beyond NLP.
    """

    result = full_chain.invoke({"text": article})
    print(f"  Input: {len(article.split())} words")
    print(f"\n  Bullet-point output:")
    for line in result.strip().splitlines():
        if line.strip():
            print(f"    {line.strip()}")


# ── 5. FAISS vector store with Ollama embeddings ─────────────────────────────
def demo_faiss_vector_store():
    print("\n=== 5. FAISS Vector Store with Ollama Embeddings ===")
    embeddings = get_embeddings()
    llm = get_llm()

    docs = [
        Document(page_content="Python was created by Guido van Rossum in 1991.", metadata={"topic": "python"}),
        Document(page_content="Python's 'Zen of Python' values simplicity and readability.", metadata={"topic": "python"}),
        Document(page_content="FastAPI is a modern web framework built on Python type hints.", metadata={"topic": "web"}),
        Document(page_content="Docker containers package apps with all their dependencies.", metadata={"topic": "devops"}),
        Document(page_content="Kubernetes orchestrates containers across a cluster of machines.", metadata={"topic": "devops"}),
        Document(page_content="LangChain provides abstractions for building LLM-powered apps.", metadata={"topic": "ai"}),
    ]

    print(f"  Building FAISS index from {len(docs)} documents (local embeddings)...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    query = "What tools are used to manage containerised applications?"
    print(f"\n  Query: {query}")
    results = vectorstore.similarity_search_with_score(query, k=3)

    print("  Top matches:")
    for doc, score in results:
        print(f"    [{score:.3f}] {doc.page_content}")

    # RAG-style answer
    context = "\n".join(d.page_content for d, _ in results)
    rag_prompt = PromptTemplate.from_template(
        "Use only the context below to answer the question in 1 sentence.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    print(f"\n  RAG answer: {answer.strip()}")


if __name__ == "__main__":
    print("Ollama 01-Basics — LangChain Integration")
    print("=" * 45)
    print(f"  LLM:       {MODEL}")
    print(f"  Embeddings: {EMBED_MODEL}")
    print(f"  Base URL:  {BASE_URL}")

    try:
        import ollama as _ollama
        _ollama.list()
    except Exception:
        print("\n  ERROR: Cannot connect to Ollama. Start with: ollama serve")
        sys.exit(1)

    demo_basic_llm()
    demo_embeddings()
    demo_lcel_chain()
    demo_multi_step_chain()
    demo_faiss_vector_store()
