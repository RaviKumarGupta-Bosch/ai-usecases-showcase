"""
02 - Conversational RAG
========================
A full chatbot that retrieves relevant documents AND remembers the conversation.
The retriever is history-aware: it rephrases the follow-up question using history.

Topics covered:
  1. create_history_aware_retriever — rephrase query using chat history
  2. create_stuff_documents_chain — QA chain that formats docs into prompt
  3. create_retrieval_chain — wire retriever + QA chain
  4. RunnableWithMessageHistory — inject message history automatically
  5. Multi-turn conversation demo
  6. Inspecting retrieved sources per turn
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── Knowledge base about a fictional SaaS product ────────────────────────────
KB_TEXTS = [
    "CloudSync Pro is a cloud storage and file synchronisation platform launched in 2023. It supports Windows, macOS, Linux, iOS, and Android.",
    "CloudSync Pro offers three pricing tiers: Free (5GB), Standard ($9.99/mo, 100GB), and Business ($29.99/mo, unlimited storage with admin controls).",
    "The Business tier includes advanced features: team collaboration, audit logs, SSO (Single Sign-On), custom branding, and priority 24/7 support.",
    "CloudSync Pro uses AES-256 encryption at rest and TLS 1.3 in transit. All data is stored in ISO 27001 certified data centres.",
    "File versioning is available on Standard and Business plans. Standard plan keeps 30 days of history; Business plan keeps 365 days.",
    "CloudSync Pro integrates natively with Google Workspace, Microsoft 365, Slack, and Zapier. The REST API is available for Business customers.",
    "Customer support: Free users get community forum access only. Standard users get email support with 48h response SLA. Business users get live chat and phone support.",
    "CloudSync Pro has a 14-day free trial for both Standard and Business tiers. No credit card is required to start the trial.",
    "Data migration: CloudSync Pro provides a migration tool to import from Google Drive, Dropbox, Box, and OneDrive. Large migrations (>1TB) require Business plan.",
    "Compliance: CloudSync Pro is GDPR compliant and SOC 2 Type II certified. EU data residency is available for Business customers at no extra cost.",
]

KB_METADATAS = [
    {"topic": "overview"},
    {"topic": "pricing"},
    {"topic": "business_features"},
    {"topic": "security"},
    {"topic": "versioning"},
    {"topic": "integrations"},
    {"topic": "support"},
    {"topic": "trial"},
    {"topic": "migration"},
    {"topic": "compliance"},
]


def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    docs = splitter.create_documents(KB_TEXTS, metadatas=KB_METADATAS)
    return FAISS.from_documents(docs, embeddings)


# ── 1. Build history-aware retriever ─────────────────────────────────────────
def build_conversational_rag():
    db = build_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Prompt that rephrases the follow-up question into a standalone search query
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question which might reference context "
         "in the chat history, formulate a standalone question which can be understood "
         "without the chat history. Do NOT answer the question, just reformulate it if needed "
         "and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # This retriever rephrases the query using history before searching
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA prompt that uses retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful product support assistant for CloudSync Pro.\n\n"
         "Answer the user's question using ONLY the context below.\n"
         "If the context doesn't contain the answer, say you don't know.\n"
         "Be concise and friendly.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Chain: format retrieved docs → generate answer
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full RAG chain: retriever + QA chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Wrap with message history management
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag, store


# ── 2. Multi-turn conversation demo ──────────────────────────────────────────
def run_conversation_demo():
    print("=" * 60)
    print("   CONVERSATIONAL RAG — CloudSync Pro Support Bot")
    print("=" * 60)

    rag_chain, store = build_conversational_rag()
    session_id = "demo-session"
    config = {"configurable": {"session_id": session_id}}

    # Simulated multi-turn conversation
    turns = [
        "What is CloudSync Pro?",
        "What are the pricing options?",
        "What do I get with the Business plan specifically?",
        "Is there a free trial for it?",
        "How secure is the storage?",
        "Can I import my files from Google Drive?",
        "Does it support SSO?",          # references Business plan from earlier
        "What's the support SLA for Standard users?",
    ]

    for i, question in enumerate(turns, 1):
        result = rag_chain.invoke({"input": question}, config)
        print(f"\n[Turn {i}] User: {question}")
        print(f"         Bot : {result['answer']}")
        if result.get("context"):
            topics = [d.metadata.get("topic", "?") for d in result["context"]]
            print(f"         Src : {topics}")

    print(f"\n{'─' * 55}")
    print(f"Total messages in history: {len(store[session_id].messages)}")


# ── 3. Interactive CLI ────────────────────────────────────────────────────────
def interactive_support_bot():
    print("\n=== CloudSync Pro Support Bot (type 'quit' to exit) ===")
    rag_chain, _ = build_conversational_rag()
    config = {"configurable": {"session_id": "interactive"}}

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        result = rag_chain.invoke({"input": user_input}, config)
        print(f"Bot: {result['answer']}")


if __name__ == "__main__":
    run_conversation_demo()
    # Uncomment to try interactive mode:
    # interactive_support_bot()
