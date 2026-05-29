"""
RAG Advanced 01 — HyDE (Hypothetical Document Embeddings)
===========================================================
Topics covered:
  1. What is HyDE and when to use it
  2. Generating hypothetical documents with an LLM
  3. Embedding the hypothetical document for retrieval
  4. Comparing HyDE vs standard retrieval
  5. HyDE in a full RAG pipeline

HyDE works by: query → LLM generates a fake "ideal answer" → embed that → retrieve
The idea: a hypothetical answer is closer in embedding space to real answers than the query.

Run:
  python 01_hyde.py
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Specialised corpus: technical deep-dives (harder for naive RAG)
TECHNICAL_DOCS = [
    Document(page_content="The PageRank algorithm assigns a numerical weighting to each element of a hyperlinked set of documents. It measures the relative importance of a page based on how many pages link to it, and the importance of those linking pages. Originally developed by Larry Page and Sergey Brin at Stanford University.", metadata={"topic": "search"}),
    Document(page_content="B-tree indexes maintain sorted data and allow searches, sequential access, insertions, and deletions in O(log n) time. In PostgreSQL, B-trees are the default index type. They are optimal for equality and range queries on high-cardinality columns.", metadata={"topic": "databases"}),
    Document(page_content="The RAFT consensus algorithm is designed for understandability. Leader election: a node becomes a candidate if it doesn't hear from a leader; it requests votes from other nodes. Log replication: the leader receives client requests and replicates them to follower nodes before responding.", metadata={"topic": "distributed-systems"}),
    Document(page_content="Bloom filters use multiple hash functions and a bit array to test set membership. False positives are possible but false negatives are not. Space complexity: O(m) where m is the size of the bit array. Used in databases, caches, and network systems to avoid expensive lookups.", metadata={"topic": "data-structures"}),
    Document(page_content="Consistent hashing distributes data across nodes with minimal rehashing when nodes join or leave. Each node and key is mapped to a point on a virtual ring. A key is assigned to the nearest node clockwise. Used in distributed caches like DynamoDB and Cassandra.", metadata={"topic": "distributed-systems"}),
    Document(page_content="The Copy-on-Write (COW) mechanism delays copying data until a write occurs. When a process forks, both parent and child share the same memory pages. Only when one modifies a page is a copy made. This makes fork() fast and memory-efficient in Unix-based systems.", metadata={"topic": "operating-systems"}),
    Document(page_content="Write-Ahead Logging (WAL) ensures database durability. Changes are first written to a sequential log before being applied to data pages. On crash recovery, the database replays the WAL to restore a consistent state. PostgreSQL, SQLite, and most ACID databases use WAL.", metadata={"topic": "databases"}),
    Document(page_content="The ART (Adaptive Radix Tree) is a space-efficient trie variant used for in-memory indexing. It adapts its node type (4, 16, 48, 256 children) based on the number of children, reducing memory overhead. Used in ClickHouse and HyPer database systems.", metadata={"topic": "data-structures"}),
]


def build_store(docs: list[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


# ── 1. Manual HyDE demonstration ──────────────────────────────────────────────
def demo_manual_hyde():
    print("\n=== 1. Manual HyDE — Step by Step ===")

    vs = build_store(TECHNICAL_DOCS)

    query = "How does a database handle system crashes without losing data?"
    print(f"Original query: {query}")

    # Step 1: Standard retrieval
    standard_results = vs.similarity_search(query, k=2)
    print("\nStandard retrieval top results:")
    for doc in standard_results:
        print(f"  [{doc.metadata.get('topic')}] {doc.page_content[:80]}...")

    # Step 2: Generate hypothetical document (fake ideal answer)
    hyde_prompt = ChatPromptTemplate.from_template(
        """Write a 2-sentence technical paragraph that directly answers this question.
Be specific and use technical terminology.

Question: {question}

Answer:"""
    )
    hypothetical_doc = (hyde_prompt | llm | StrOutputParser()).invoke({"question": query})
    print(f"\nHypothetical document generated by LLM:")
    print(f"  {hypothetical_doc}")

    # Step 3: Retrieve using hypothetical doc as query
    hyde_results = vs.similarity_search(hypothetical_doc, k=2)
    print("\nHyDE retrieval top results:")
    for doc in hyde_results:
        print(f"  [{doc.metadata.get('topic')}] {doc.page_content[:80]}...")

    # Step 4: Generate final answer
    context = "\n\n".join(d.page_content for d in hyde_results)
    answer = llm.invoke(
        f"Context:\n{context}\n\nAnswer precisely: {query}"
    ).content
    print(f"\nFinal answer: {answer}")


# ── 2. HyDE as a LangChain retriever ─────────────────────────────────────────
def demo_hyde_retriever():
    print("\n=== 2. HyDE as a LangChain Chain ===")

    vs = build_store(TECHNICAL_DOCS)

    # HyDE generation chain
    hyde_gen_prompt = ChatPromptTemplate.from_template(
        """Generate a concise, factual technical paragraph (3-4 sentences) that would 
directly answer this question. Use precise technical language.

Question: {question}"""
    )
    hyde_gen = hyde_gen_prompt | llm | StrOutputParser()

    # Full HyDE RAG chain
    def hyde_retrieve(question: str) -> list[Document]:
        hypothetical = hyde_gen.invoke({"question": question})
        return vs.similarity_search(hypothetical, k=3)

    final_prompt = ChatPromptTemplate.from_template(
        """Use ONLY the context to answer. Be precise and cite technical details.

Context:
{context}

Question: {question}"""
    )

    hyde_rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: "\n\n".join(
                d.page_content for d in hyde_retrieve(x["question"])
            ))
        )
        | final_prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What algorithm ensures that no data is lost when a database server crashes?",
        "How can you quickly check if an element exists in a large set without storing all elements?",
        "How do distributed databases route requests to the correct server when servers are added or removed?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {hyde_rag_chain.invoke({'question': q})}")


# ── 3. Quantitative comparison ────────────────────────────────────────────────
def demo_comparison():
    print("\n=== 3. Comparing Standard vs HyDE Retrieval ===")

    vs = build_store(TECHNICAL_DOCS)

    hyde_gen_prompt = ChatPromptTemplate.from_template(
        "Write a 2-sentence technical answer to: {question}"
    )
    hyde_gen = hyde_gen_prompt | llm | StrOutputParser()

    test_cases = [
        {
            "query": "mechanism for distributing load across nodes when cluster size changes",
            "expected_topic": "distributed-systems",
        },
        {
            "query": "fast membership testing without storing every element",
            "expected_topic": "data-structures",
        },
        {
            "query": "how databases guarantee committed transactions survive power failures",
            "expected_topic": "databases",
        },
    ]

    for case in test_cases:
        q = case["query"]
        expected = case["expected_topic"]

        # Standard
        std_docs = vs.similarity_search(q, k=1)
        std_topic = std_docs[0].metadata.get("topic") if std_docs else "none"

        # HyDE
        hyp_doc = hyde_gen.invoke({"question": q})
        hyde_docs = vs.similarity_search(hyp_doc, k=1)
        hyde_topic = hyde_docs[0].metadata.get("topic") if hyde_docs else "none"

        std_hit = "✓" if std_topic == expected else "✗"
        hyde_hit = "✓" if hyde_topic == expected else "✗"

        print(f"\nQ: {q[:60]}...")
        print(f"  Standard: topic={std_topic} {std_hit}")
        print(f"  HyDE:     topic={hyde_topic} {hyde_hit}")


if __name__ == "__main__":
    demo_manual_hyde()
    demo_hyde_retriever()
    demo_comparison()
