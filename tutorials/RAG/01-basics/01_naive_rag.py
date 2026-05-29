"""
RAG Basics 01 — Naive RAG Pipeline
=====================================
Topics covered:
  1. Document loading and text splitting
  2. Embedding generation with OpenAI
  3. FAISS vector store creation
  4. Similarity search retrieval
  5. Prompt construction with retrieved context
  6. End-to-end Q&A chain

Run:
  python 01_naive_rag.py
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Sample corpus ─────────────────────────────────────────────────────────────
CORPUS = [
    """Python is a high-level, general-purpose programming language created by Guido van Rossum
and first released in 1991. Its design philosophy emphasises code readability with the use of
significant indentation. Python is dynamically typed and garbage-collected. It supports multiple
programming paradigms, including structured, object-oriented, and functional programming.
Python is consistently ranked among the most popular programming languages.""",

    """Machine learning is a subset of artificial intelligence that gives systems the ability to
automatically learn and improve from experience without being explicitly programmed. Machine
learning focuses on the development of computer programs that can access data and use it to
learn for themselves. The process begins with observations or data, such as examples, to look for
patterns in data and make better decisions in the future.""",

    """The transformer architecture was introduced in the 2017 paper 'Attention Is All You Need'
by Vaswani et al. It relies entirely on attention mechanisms, dispensing with recurrence and
convolutions entirely. Transformers have become the dominant architecture for natural language
processing tasks. The key innovation is the multi-head self-attention mechanism which allows the
model to attend to different positions of the input sequence simultaneously.""",

    """Large Language Models (LLMs) are deep learning models trained on massive text datasets.
They learn to predict the next token in a sequence, which allows them to generate coherent text.
GPT-4, Claude, and Gemini are prominent examples. LLMs can perform tasks like translation,
summarisation, question answering, and code generation without task-specific training data —
a capability known as zero-shot or few-shot learning.""",

    """Vector databases store data as high-dimensional vectors, enabling semantic similarity
search. Unlike traditional databases that match exact keywords, vector databases find the most
similar vectors using metrics like cosine similarity or dot product. Popular vector databases
include FAISS (Facebook AI Similarity Search), Chroma, Qdrant, Pinecone, and Weaviate.
They are essential infrastructure for RAG (Retrieval-Augmented Generation) systems.""",

    """Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with text generation. Instead of relying solely on a model's parametric knowledge, RAG retrieves
relevant documents from an external knowledge base and uses them as context for generation.
This reduces hallucinations and allows models to answer questions about private or recent data
that was not in their training set. RAG was introduced by Lewis et al. in 2020.""",
]


def build_vector_store(texts: list[str], chunk_size: int = 300, overlap: int = 50) -> FAISS:
    """Split texts, embed, and build a FAISS index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.create_documents(texts)
    print(f"Split into {len(docs)} chunks from {len(texts)} source documents")
    return FAISS.from_documents(docs, embeddings)


# ── 1. Manual retrieval and generation ───────────────────────────────────────
def demo_manual_rag():
    print("\n=== 1. Manual RAG (Step by Step) ===")

    vs = build_vector_store(CORPUS)

    query = "What is Retrieval-Augmented Generation and when was it introduced?"
    print(f"\nQuery: {query}")

    # Retrieve top-3 chunks
    results = vs.similarity_search(query, k=3)
    print(f"\nRetrieved {len(results)} chunks:")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:80]}...")

    # Build prompt
    context = "\n\n".join(doc.page_content for doc in results)
    prompt = f"""Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {query}
Answer:"""

    answer = llm.invoke(prompt).content
    print(f"\nAnswer: {answer}")


# ── 2. Similarity score threshold ────────────────────────────────────────────
def demo_similarity_scores():
    print("\n=== 2. Similarity Scores and Thresholds ===")

    vs = build_vector_store(CORPUS)

    queries = [
        "Who invented Python?",
        "What is the capital of France?",   # out-of-corpus
        "How do transformers use attention?",
    ]

    for query in queries:
        results = vs.similarity_search_with_score(query, k=2)
        print(f"\nQuery: {query}")
        for doc, score in results:
            # FAISS returns L2 distance (lower = more similar)
            print(f"  Score: {score:.4f} | {doc.page_content[:70]}...")

        # Only answer if best score is below threshold (< 0.8 for L2 distance)
        best_score = results[0][1] if results else 9999
        if best_score < 0.8:
            context = "\n\n".join(d.page_content for d, _ in results)
            answer = llm.invoke(
                f"Context:\n{context}\n\nAnswer briefly: {query}"
            ).content
            print(f"  → {answer}")
        else:
            print("  → Low confidence, skipping answer (query likely out-of-corpus)")


# ── 3. LangChain RAG chain (LCEL) ────────────────────────────────────────────
def demo_lcel_rag_chain():
    print("\n=== 3. LCEL RAG Chain ===")

    vs = build_vector_store(CORPUS)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    prompt_template = ChatPromptTemplate.from_template(
        """You are an AI assistant. Answer the question using ONLY the context provided.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(docs))

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    questions = [
        "What are the key features of the Transformer architecture?",
        "How do vector databases differ from traditional databases?",
        "What is few-shot learning?",
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")


# ── 4. RAG with source attribution ───────────────────────────────────────────
def demo_rag_with_sources():
    print("\n=== 4. RAG with Source Attribution ===")

    # Add metadata to documents
    docs_with_meta = [
        Document(page_content=text, metadata={"source": f"doc_{i}", "topic": topic})
        for i, (text, topic) in enumerate(zip(CORPUS, [
            "python", "machine-learning", "transformers",
            "llms", "vector-databases", "rag"
        ]))
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    split_docs = splitter.split_documents(docs_with_meta)
    vs = FAISS.from_documents(split_docs, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    query = "What models are examples of LLMs?"
    retrieved = retriever.invoke(query)

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in retrieved
    )

    prompt = f"""Answer the question and cite the sources used (e.g. [Source: doc_3]).

Context:
{context}

Question: {query}
Answer:"""

    answer = llm.invoke(prompt).content
    print(f"Q: {query}")
    print(f"A: {answer}")
    print(f"\nSources used: {[d.metadata.get('source') for d in retrieved]}")


if __name__ == "__main__":
    demo_manual_rag()
    demo_similarity_scores()
    demo_lcel_rag_chain()
    demo_rag_with_sources()
