"""
04 - Retrieval
==============
Retrievers fetch relevant documents for a query.
More sophisticated retrievers improve RAG accuracy significantly.

Topics covered:
  1. VectorStoreRetriever (baseline)
  2. MultiQueryRetriever — rephrases the query for better recall
  3. ContextualCompressionRetriever — filters / compresses results
  4. EnsembleRetriever — combine BM25 + vector (hybrid search)
  5. Self-query retriever — natural language metadata filtering
  6. create_retrieval_chain — production-ready RAG chain
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DOCS = [
    Document(page_content="Python is a high-level, interpreted programming language known for its readability. It supports multiple programming paradigms and has a large standard library.", metadata={"language": "Python", "category": "basics"}),
    Document(page_content="Python's pandas library provides fast, flexible data structures for data analysis. DataFrame and Series are its core abstractions.", metadata={"language": "Python", "category": "data_science"}),
    Document(page_content="NumPy provides efficient numerical operations in Python. It supports N-dimensional arrays and broadcasting.", metadata={"language": "Python", "category": "data_science"}),
    Document(page_content="FastAPI is a modern, fast Python framework for building APIs. It uses Pydantic for data validation and generates OpenAPI docs automatically.", metadata={"language": "Python", "category": "web"}),
    Document(page_content="Django is a high-level Python web framework that follows the model-template-view pattern. It includes an ORM, admin panel, and authentication.", metadata={"language": "Python", "category": "web"}),
    Document(page_content="JavaScript is the language of the web, running in browsers and on servers via Node.js. It supports event-driven, functional, and OOP styles.", metadata={"language": "JavaScript", "category": "basics"}),
    Document(page_content="React is a JavaScript library for building user interfaces. It uses a virtual DOM and component-based architecture.", metadata={"language": "JavaScript", "category": "web"}),
    Document(page_content="TypeScript extends JavaScript with static type checking. It compiles to plain JavaScript and is widely used in large codebases.", metadata={"language": "TypeScript", "category": "basics"}),
    Document(page_content="Rust is a systems programming language focused on safety, concurrency, and performance. It prevents memory errors at compile time.", metadata={"language": "Rust", "category": "systems"}),
    Document(page_content="Go (Golang) is a statically typed, compiled language designed for simplicity and performance. It excels at building concurrent services.", metadata={"language": "Go", "category": "systems"}),
    Document(page_content="Scikit-learn is a Python library for classical machine learning. It provides implementations of regression, classification, clustering, and more.", metadata={"language": "Python", "category": "machine_learning"}),
    Document(page_content="PyTorch is a deep learning framework developed by Meta AI. It uses dynamic computation graphs, making it popular for research.", metadata={"language": "Python", "category": "machine_learning"}),
]


def build_faiss_db():
    return FAISS.from_documents(DOCS, embeddings)


# ── 1. VectorStoreRetriever (baseline) ───────────────────────────────────────
def demo_basic_retriever():
    db = build_faiss_db()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    print("=== 1. VectorStoreRetriever (baseline) ===")
    results = retriever.invoke("Python libraries for data analysis")
    for doc in results:
        print(f"  [{doc.metadata['category']}] {doc.page_content[:80]}...")


# ── 2. MultiQueryRetriever ───────────────────────────────────────────────────
def demo_multi_query():
    """
    Generates 3–5 paraphrased versions of the query, fetches results for each,
    and de-duplicates. Increases recall for vague or ambiguous queries.
    """
    db = build_faiss_db()
    base_retriever = db.as_retriever(search_kwargs={"k": 3})

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    print("\n=== 2. MultiQueryRetriever ===")
    # A somewhat ambiguous query
    results = multi_query_retriever.invoke("best language for web services")
    print(f"Retrieved {len(results)} unique document(s) across all query variants:")
    for doc in results:
        print(f"  [{doc.metadata['language']}|{doc.metadata['category']}] {doc.page_content[:70]}...")


# ── 3. ContextualCompressionRetriever ────────────────────────────────────────
def demo_contextual_compression():
    """
    Filters documents that are not relevant to the query using the LLM.
    LLMChainFilter passes each doc through the LLM and discards irrelevant ones.
    """
    db = build_faiss_db()
    base_retriever = db.as_retriever(search_kwargs={"k": 5})

    compressor = LLMChainFilter.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    print("\n=== 3. ContextualCompressionRetriever ===")
    query = "memory safe systems programming"
    results = compression_retriever.invoke(query)
    print(f"Query: '{query}'")
    print(f"Compressed to {len(results)} relevant document(s):")
    for doc in results:
        print(f"  [{doc.metadata['language']}] {doc.page_content[:80]}...")


# ── 4. EnsembleRetriever — BM25 + vector (hybrid search) ────────────────────
def demo_ensemble_retriever():
    """
    Combines keyword search (BM25) with semantic search (vector).
    BM25 is excellent for exact keyword matches; vectors handle semantic similarity.
    weights sum to 1.0.
    """
    texts = [doc.page_content for doc in DOCS]
    metadatas = [doc.metadata for doc in DOCS]

    # BM25 — classic TF-IDF keyword retriever
    bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25_retriever.k = 3

    # Vector retriever
    db = FAISS.from_documents(DOCS, embeddings)
    vector_retriever = db.as_retriever(search_kwargs={"k": 3})

    # Ensemble: 40% BM25, 60% vector
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    print("\n=== 4. EnsembleRetriever (BM25 + Vector) ===")
    results = ensemble.invoke("Django web framework database ORM")
    print(f"Retrieved {len(results)} document(s):")
    for doc in results:
        print(f"  [{doc.metadata['language']}|{doc.metadata['category']}] {doc.page_content[:80]}...")


# ── 5. create_retrieval_chain — full RAG chain ───────────────────────────────
def demo_retrieval_chain():
    """
    create_stuff_documents_chain: stuffs all retrieved docs into the prompt.
    create_retrieval_chain: wires retriever + QA chain end-to-end.
    """
    db = build_faiss_db()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful programming assistant.\n\n"
         "Use only the following context to answer the question.\n"
         "If the answer isn't in the context, say you don't know.\n\n"
         "Context:\n{context}"),
        ("human", "{input}"),
    ])

    # Chain that formats retrieved docs into the prompt
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full RAG chain: query → retrieve → combine → answer
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("\n=== 5. create_retrieval_chain (RAG) ===")
    queries = [
        "What Python framework should I use for building a REST API?",
        "Which language is best for system-level programming with memory safety?",
    ]
    for query in queries:
        result = rag_chain.invoke({"input": query})
        print(f"\nQ: {query}")
        print(f"A: {result['answer']}")
        print(f"Sources: {[d.metadata['language'] for d in result['context']]}")


# ── 6. Custom format_docs helper ─────────────────────────────────────────────
def demo_custom_rag():
    """
    Manual LCEL RAG chain with custom document formatting and source citation.
    """
    db = build_faiss_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(
                f"[{i}] ({doc.metadata.get('language', 'N/A')}/{doc.metadata.get('category', 'N/A')})\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(formatted)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (mention the source numbers you used):"
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n=== 6. Custom LCEL RAG with Citations ===")
    answer = rag_chain.invoke("What are Python's main data science libraries?")
    print(answer)


if __name__ == "__main__":
    demo_basic_retriever()
    demo_multi_query()
    demo_contextual_compression()
    demo_ensemble_retriever()
    demo_retrieval_chain()
    demo_custom_rag()
