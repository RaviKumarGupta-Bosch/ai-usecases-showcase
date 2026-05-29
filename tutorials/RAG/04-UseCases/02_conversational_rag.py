"""
RAG Use Case 01 — Conversational RAG (Multi-turn Q&A)
======================================================
Topics covered:
  1. Chat history management in RAG
  2. Query contextualisation — rewriting follow-up questions
  3. Conversational retrieval chain
  4. Session-based chat memory
  5. Multi-turn conversation with a document corpus

Run:
  python 02_conversational_rag.py
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Corpus: Python programming guide
PYTHON_DOCS = [
    Document(page_content="Python variables are dynamically typed. You don't need to declare types: x = 42 creates an integer. Type hints are optional: x: int = 42. Use type() to check the type at runtime."),
    Document(page_content="Python functions are defined with def. They support default arguments: def greet(name, greeting='Hello'). *args collects positional arguments into a tuple, **kwargs collects keyword arguments into a dict."),
    Document(page_content="Python classes use class keyword. __init__ is the constructor. self refers to the instance. Inheritance: class Dog(Animal): pass. Use super() to call parent methods."),
    Document(page_content="Python list operations: append(x) adds to end, insert(i, x) adds at index, pop() removes last, remove(x) removes first match. sort() sorts in-place, sorted() returns new list."),
    Document(page_content="Python dictionaries: dict.get(key, default) safely gets values. dict.items() returns key-value pairs. dict comprehension: {k: v for k, v in pairs}. Python 3.7+ dicts preserve insertion order."),
    Document(page_content="Python exceptions: try/except/else/finally. Raise with raise ValueError('message'). Custom exceptions inherit from Exception. Context managers use with statement and __enter__/__exit__."),
    Document(page_content="Python file I/O: open(file, mode) where mode is 'r', 'w', 'a', 'rb'. Use with statement for auto-close. read() reads all, readline() reads one line, readlines() returns list of lines."),
    Document(page_content="Python modules and packages: import module, from module import func. __init__.py makes a folder a package. __name__ == '__main__' guards script execution. pip installs packages."),
    Document(page_content="Python list comprehensions: [expr for item in iterable if condition]. Generator expressions use () instead of []. Dict comprehensions: {k: v for ...}. Set comprehensions: {x for ...}."),
    Document(page_content="Python decorators wrap functions using @decorator syntax. functools.wraps preserves the original function's metadata. Common built-in decorators: @property, @staticmethod, @classmethod, @abstractmethod."),
    Document(page_content="Python async programming: async def defines coroutines. await suspends execution. asyncio.run() starts the event loop. asyncio.gather() runs coroutines concurrently. Use aiohttp for async HTTP requests."),
    Document(page_content="Python testing with pytest: test functions start with test_. Use assert statements. Fixtures use @pytest.fixture. Parametrize with @pytest.mark.parametrize. Run with: pytest test_file.py -v."),
]


def build_rag_components():
    """Build vector store and retriever."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(PYTHON_DOCS)
    vs = FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    return retriever


# ── 1. Basic conversational RAG ───────────────────────────────────────────────
def demo_basic_conversational_rag():
    print("\n=== 1. Basic Conversational RAG ===")
    print("(Shows how follow-up questions fail without history contextualisation)")

    retriever = build_rag_components()

    def ask(query: str, history: list[BaseMessage]) -> tuple[str, list[BaseMessage]]:
        docs = retriever.invoke(query)
        context = "\n".join(d.page_content for d in docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Python tutor. Answer using ONLY the context provided."),
            MessagesPlaceholder("history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query, "history": history})

        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))
        return answer, history

    history: list[BaseMessage] = []
    conversation = [
        "How do I add an item to a Python list?",
        "What about removing an item?",            # ambiguous — needs history to work
        "Can you show me how to sort it too?",     # still about list
    ]

    for q in conversation:
        answer, history = ask(q, history)
        print(f"\nQ: {q}")
        print(f"A: {answer}")


# ── 2. Query contextualisation (standalone question reformulation) ─────────────
def demo_query_contextualisation():
    print("\n=== 2. Query Contextualisation ===")
    print("(Reformulates ambiguous follow-up questions into standalone queries)")

    contextualise_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Given a chat history and a follow-up question, rewrite the follow-up question 
to be a complete standalone question that can be understood without the history.
Return ONLY the rewritten question, nothing else.
If the question is already standalone, return it unchanged."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    contextualise_chain = contextualise_prompt | llm | StrOutputParser()

    retriever = build_rag_components()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the provided context.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])
    qa_chain = qa_prompt | llm | StrOutputParser()

    def conversational_rag(question: str, history: list[BaseMessage]) -> tuple[str, list[BaseMessage]]:
        # Contextualise only if there's history
        if history:
            standalone = contextualise_chain.invoke({
                "chat_history": history,
                "question": question,
            })
            print(f"  [Rewritten to]: {standalone}")
        else:
            standalone = question

        # Retrieve with standalone question
        docs = retriever.invoke(standalone)
        context = "\n".join(d.page_content for d in docs)

        # Answer with full history
        answer = qa_chain.invoke({
            "context": context,
            "question": question,
            "chat_history": history,
        })

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        return answer, history

    history: list[BaseMessage] = []
    conversation = [
        "What is a Python decorator?",
        "How do I create one?",                    # "one" → needs contextualisation
        "Can you show the @property example?",     # still about decorators
        "Now tell me about async functions",        # topic switch
        "How do I run them?",                      # "them" → async functions
    ]

    for q in conversation:
        print(f"\nQ: {q}")
        answer, history = conversational_rag(q, history)
        print(f"A: {answer}")


# ── 3. Multi-session RAG ──────────────────────────────────────────────────────
def demo_multi_session():
    print("\n=== 3. Multi-Session RAG ===")
    print("(Different users with separate conversation histories)")

    retriever = build_rag_components()
    store: dict[str, ChatMessageHistory] = {}

    def get_session(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python tutor. Context:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: "\n".join(
                d.page_content for d in retriever.invoke(x["question"])
            ))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    with_history = RunnableWithMessageHistory(
        chain,
        get_session,
        input_messages_key="question",
        history_messages_key="history",
    )

    # Two separate user sessions
    sessions = {
        "user_alice": [
            "How do I read a file in Python?",
            "What about writing to it?",
        ],
        "user_bob": [
            "Tell me about Python classes",
            "How does inheritance work?",
        ],
    }

    for session_id, questions in sessions.items():
        print(f"\n--- Session: {session_id} ---")
        for q in questions:
            answer = with_history.invoke(
                {"question": q},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"Q: {q}")
            print(f"A: {answer}\n")

    print(f"Total sessions: {len(store)}")
    for sid, hist in store.items():
        print(f"  {sid}: {len(hist.messages)} messages")


if __name__ == "__main__":
    demo_basic_conversational_rag()
    demo_query_contextualisation()
    demo_multi_session()
