"""
01 - Full RAG Pipeline
=======================
End-to-end Retrieval-Augmented Generation: ingest → split → embed → store → retrieve → generate.

Topics covered:
  1. Document ingestion (TextLoader + DirectoryLoader)
  2. RecursiveCharacterTextSplitter with metadata preservation
  3. FAISS vector store construction
  4. Custom RAG prompt with citation instructions
  5. LCEL RAG chain (context + question → answer with sources)
  6. Batch evaluation across multiple questions
  7. Displaying source citations
"""

import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ── Knowledge Base ────────────────────────────────────────────────────────────
KB_DOCUMENTS = {
    "langchain_overview.txt": """
LangChain is an open-source framework for building applications powered by large language models.
It was founded by Harrison Chase in 2022 and quickly became one of the most popular AI frameworks.
LangChain provides a standard interface for LLMs, along with modules for chains, agents, memory,
and retrieval. Its core abstraction is the "chain" — a sequence of calls to components.
The LangChain Expression Language (LCEL) uses the pipe operator (|) to compose chains declaratively,
enabling streaming, async, and batch execution out of the box.
""",
    "langgraph_overview.txt": """
LangGraph is an extension of LangChain that enables the creation of stateful, multi-actor applications.
It models workflows as directed graphs where nodes perform computation and edges define the flow.
LangGraph supports cycles (unlike pure chains), making it ideal for agent loops.
Key concepts: StateGraph (the graph builder), TypedDict state (shared data between nodes),
conditional edges (branching based on state), and MemorySaver (checkpoint persistence).
LangGraph is the recommended approach for building production-grade agentic applications in 2024+.
""",
    "langsmith_overview.txt": """
LangSmith is a developer platform for debugging, testing, evaluating, and monitoring LLM applications.
It provides full tracing of every LangChain call, including inputs, outputs, latency, and token usage.
LangSmith's evaluation framework lets you define metrics, create datasets, and run automated experiments.
The @traceable decorator integrates custom Python functions into the trace tree.
LangSmith connects to LangChain automatically when LANGCHAIN_TRACING_V2=true is set in the environment.
""",
    "rag_best_practices.txt": """
RAG (Retrieval-Augmented Generation) best practices:
1. Chunk size matters: 200-500 tokens typically works well. Use overlap of 10-20%.
2. Metadata: always store source, date, and domain in document metadata for filtering.
3. Hybrid search: combining BM25 keyword search with vector search improves recall.
4. Re-ranking: use a cross-encoder to re-rank retrieved results before passing to LLM.
5. Query expansion: generate multiple query variants to improve retrieval coverage.
6. Evaluation: measure recall@k and faithfulness using LangSmith or RAGAS.
7. Contextual compression: filter irrelevant passages to reduce noise in the context.
""",
    "vector_databases.txt": """
Popular vector databases for AI applications:
- FAISS (Facebook AI Similarity Search): In-memory, extremely fast, no server required.
  Best for prototyping and small-to-medium datasets (<10M vectors).
- Chroma: Open-source, local or cloud, easy Python integration, good for development.
- Pinecone: Fully managed cloud vector DB, scales to billions of vectors.
- Weaviate: Open-source, supports hybrid search natively, has built-in ML models.
- Qdrant: Rust-based, extremely fast, supports payload filtering, open-source with cloud option.
- pgvector: PostgreSQL extension, ideal if you already use Postgres.
Choosing criteria: scale, latency requirements, metadata filtering, cost, and team familiarity.
""",
}


# ── 1. Build the RAG pipeline ────────────────────────────────────────────────
def build_rag_pipeline():
    """Ingest docs → split → embed → store. Returns retriever."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        add_start_index=True,
    )

    all_docs = []
    for filename, content in KB_DOCUMENTS.items():
        doc = Document(
            page_content=content.strip(),
            metadata={"source": filename, "domain": "AI"},
        )
        chunks = splitter.split_documents([doc])
        all_docs.extend(chunks)

    print(f"Ingested {len(KB_DOCUMENTS)} documents → {len(all_docs)} chunks")

    db = FAISS.from_documents(all_docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever


# ── 2. Custom RAG prompt ─────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a knowledgeable AI assistant specialising in LangChain, LangGraph, and LangSmith.

Answer the user's question using ONLY the provided context.
If the information isn't in the context, say "I don't have enough information to answer that."

Rules:
- Be concise and factual
- Cite the source document(s) you used, e.g. [langchain_overview.txt]
- If multiple sources support your answer, cite all of them

Context:
{context}"""),
    ("human", "{question}"),
])


def format_docs_with_sources(docs: list[Document]) -> str:
    """Format documents into a numbered context block with source labels."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


# ── 3. Build the RAG chain ───────────────────────────────────────────────────
def build_rag_chain(retriever):
    """Returns a chain that accepts a question string and returns answer + source docs."""

    rag_chain_with_sources = RunnableParallel(
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
            "source_documents": retriever,
        }
    )

    def generate_answer(inputs):
        answer = (RAG_PROMPT | llm | StrOutputParser()).invoke({
            "context": inputs["context"],
            "question": inputs["question"],
        })
        return {
            "answer": answer,
            "source_documents": inputs["source_documents"],
        }

    return rag_chain_with_sources | generate_answer


# ── 4. Run evaluations ───────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    "What is LangChain and who created it?",
    "How does LangGraph differ from LangChain?",
    "What does LangSmith help with?",
    "What are the best practices for chunking documents in RAG?",
    "Which vector database should I use for a large-scale production system?",
    "What is the LCEL pipe operator used for?",
]


def run_rag_demo():
    print("=" * 60)
    print("         FULL RAG PIPELINE DEMO")
    print("=" * 60)

    retriever = build_rag_pipeline()
    rag_chain = build_rag_chain(retriever)

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n{'─' * 55}")
        print(f"Q{i}: {question}")
        result = rag_chain.invoke(question)
        print(f"\nAnswer:\n{result['answer']}")

        sources = {doc.metadata.get("source") for doc in result["source_documents"]}
        print(f"\nSources used: {', '.join(sorted(sources))}")

    print(f"\n{'=' * 60}")
    print("RAG pipeline demo complete.")


if __name__ == "__main__":
    run_rag_demo()
