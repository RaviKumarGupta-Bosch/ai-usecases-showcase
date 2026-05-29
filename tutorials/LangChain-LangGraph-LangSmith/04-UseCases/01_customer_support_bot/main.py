"""
Use Case 01 — Customer Support Bot
=====================================
A production-grade customer support chatbot for "TechFlow SaaS" with:
- Product FAQ retrieval using FAISS vector search
- Multi-turn conversation memory (RunnableWithMessageHistory)
- Escalation detection when the bot cannot resolve the issue
- LangSmith tracing for monitoring

Architecture (LangGraph):
  greet → retrieve_context → generate_answer → check_escalation
                                                  ├─ escalate → END
                                                  └─ respond  → END

Run:
  python main.py
"""

import os
import operator
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langsmith import traceable
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── Model setup ───────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── TechFlow SaaS knowledge base ──────────────────────────────────────────────
FAQ_DOCUMENTS = [
    Document(page_content="""
TechFlow Pricing Plans:
- Starter: $29/month — 5 users, 10 GB storage, email support
- Professional: $99/month — 25 users, 100 GB storage, priority support, API access
- Enterprise: $299/month — Unlimited users, 1 TB storage, 24/7 support, SLA, custom integrations
Annual billing gives 20% discount on all plans.
    """, metadata={"category": "pricing", "topic": "plans"}),

    Document(page_content="""
TechFlow Account & Billing:
- Upgrade or downgrade anytime from Settings > Billing
- Invoices are sent on the 1st of each month
- Accepted payment methods: Visa, Mastercard, PayPal, Wire Transfer
- Refunds: Pro-rated refund within 14 days for annual plans
- To cancel: Settings > Account > Cancel Subscription
    """, metadata={"category": "billing", "topic": "account"}),

    Document(page_content="""
TechFlow API Integration:
- REST API with OpenAPI 3.0 spec at api.techflow.io/docs
- Authentication: Bearer token (generate in Settings > API Keys)
- Rate limits: 100 req/min (Starter), 1000 req/min (Pro), Unlimited (Enterprise)
- SDKs available: Python, JavaScript, Java, Ruby, Go
- Webhooks supported for real-time event notifications
    """, metadata={"category": "technical", "topic": "api"}),

    Document(page_content="""
TechFlow Data & Security:
- Data stored in AWS us-east-1 and eu-west-1 (GDPR compliant)
- AES-256 encryption at rest, TLS 1.3 in transit
- SOC 2 Type II certified, ISO 27001 compliant
- GDPR DPA available for EU customers
- Data export: CSV/JSON from Settings > Data Export
- Data deletion: completed within 30 days of account closure
    """, metadata={"category": "security", "topic": "compliance"}),

    Document(page_content="""
TechFlow Troubleshooting:
- Login issues: Clear cache, try incognito mode, reset password at /forgot-password
- Slow performance: Check status.techflow.io for incidents, try different browser
- Data sync errors: Force sync via Settings > Data > Force Sync
- Integration failures: Verify API key permissions, check webhook URL is HTTPS
- If issue persists >2 hours: Contact support@techflow.io with error screenshots
    """, metadata={"category": "troubleshooting", "topic": "technical_issues"}),

    Document(page_content="""
TechFlow Support Channels:
- Email: support@techflow.io (response within 24 hours for Starter, 4 hours for Pro)
- Live chat: Available 9am-6pm EST Monday-Friday (Pro and Enterprise only)
- Phone: +1-800-TECHFLOW (Enterprise only, 24/7)
- Knowledge base: docs.techflow.io
- Community forum: community.techflow.io
- Emergency escalation: escalations@techflow.io
    """, metadata={"category": "support", "topic": "contact"}),

    Document(page_content="""
TechFlow Features:
- Project management: Kanban boards, Gantt charts, sprint planning
- Collaboration: Real-time editing, @mentions, comments, notifications
- Reporting: Custom dashboards, scheduled reports, CSV/PDF export
- Automation: If-then workflows, triggers, scheduled actions
- Integrations: Slack, Jira, GitHub, Salesforce, Zapier (200+ apps)
- Mobile apps: iOS and Android, full feature parity
    """, metadata={"category": "features", "topic": "product"}),

    Document(page_content="""
TechFlow Onboarding & Setup:
- Free onboarding call with Pro and Enterprise plans
- Setup wizard guides through workspace configuration
- Import data from: CSV, Trello, Asana, Monday.com, Notion
- Video tutorials at techflow.io/tutorials
- Typical setup time: 30 minutes for small teams, 2-4 hours for enterprise
    """, metadata={"category": "onboarding", "topic": "getting_started"}),
]

# Build FAISS index from FAQ documents
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(FAQ_DOCUMENTS)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Graph state ────────────────────────────────────────────────────────────────
class SupportState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    session_id:     str
    user_query:     str
    retrieved_docs: list[str]
    answer:         str
    should_escalate: bool
    escalation_reason: str


# ── Chat history store (for RunnableWithMessageHistory) ───────────────────────
_chat_histories: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _chat_histories:
        _chat_histories[session_id] = InMemoryChatMessageHistory()
    return _chat_histories[session_id]


# ── Nodes ─────────────────────────────────────────────────────────────────────
SUPPORT_SYSTEM = """You are Aria, a friendly and knowledgeable customer support agent for TechFlow SaaS.

Your role:
- Answer questions accurately based on the provided context
- Be empathetic and professional
- If you cannot fully resolve the issue, offer to escalate
- Keep answers concise (2-4 sentences)

Context from knowledge base:
{context}

If you don't know the answer or the issue requires manual intervention, say:
"I'd like to escalate this to our specialist team for the best resolution."
"""


def retrieve_context_node(state: SupportState) -> dict:
    """Retrieve relevant FAQ content for the user's query."""
    docs = retriever.invoke(state["user_query"])
    context_texts = [d.page_content.strip() for d in docs]
    return {"retrieved_docs": context_texts}


def generate_answer_node(state: SupportState) -> dict:
    """Generate a response using retrieved context and conversation history."""
    context = "\n\n".join(state["retrieved_docs"])
    history = get_session_history(state["session_id"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPPORT_SYSTEM),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # Build history messages
    hist_messages = history.messages

    answer = chain.invoke({
        "context":  context,
        "history":  hist_messages,
        "question": state["user_query"],
    })

    # Update history
    history.add_user_message(state["user_query"])
    history.add_ai_message(answer)

    return {
        "answer":   answer,
        "messages": [
            HumanMessage(content=state["user_query"]),
            AIMessage(content=answer),
        ],
    }


class EscalationDecision(BaseModel):
    should_escalate: bool = Field(description="True if issue needs human agent")
    reason: str = Field(description="Reason for escalation decision")


def check_escalation_node(state: SupportState) -> dict:
    """Detect if the bot's answer suggests escalation is needed."""
    escalation_llm = llm.with_structured_output(EscalationDecision)

    result = escalation_llm.invoke(
        f"""Review this support interaction and decide if escalation to a human agent is needed.

User question: {state['user_query']}
Bot answer: {state['answer']}

Escalate if:
- The answer is vague/uncertain
- The issue requires manual action (refunds, account recovery, billing disputes)
- The bot explicitly mentioned escalation
- The user expresses high frustration"""
    )

    return {
        "should_escalate":  result.should_escalate,
        "escalation_reason": result.reason,
    }


def escalate_node(state: SupportState) -> dict:
    """Escalation pathway — notify the user and log the issue."""
    escalation_msg = (
        "I've escalated your case to our specialist team. "
        "You'll receive an email at your registered address within 2 hours. "
        f"Reference: ESC-{hash(state['user_query']) % 100000:05d}. "
        "Is there anything else I can help you with in the meantime?"
    )
    history = get_session_history(state["session_id"])
    history.add_ai_message(escalation_msg)
    return {
        "answer":   escalation_msg,
        "messages": [AIMessage(content=escalation_msg)],
    }


def route_after_escalation_check(state: SupportState) -> str:
    return "escalate" if state["should_escalate"] else END


# ── Build the graph ────────────────────────────────────────────────────────────
def build_support_graph():
    memory = MemorySaver()
    graph = StateGraph(SupportState)

    graph.add_node("retrieve",    retrieve_context_node)
    graph.add_node("generate",    generate_answer_node)
    graph.add_node("check_esc",   check_escalation_node)
    graph.add_node("escalate",    escalate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check_esc")
    graph.add_conditional_edges(
        "check_esc",
        route_after_escalation_check,
        {"escalate": "escalate", END: END},
    )
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=memory)


@traceable(name="customer_support_session", tags=["production", "support"])
def handle_message(app, session_id: str, user_message: str) -> str:
    """Process a single customer message."""
    config = {"configurable": {"thread_id": session_id}}
    result = app.invoke(
        {
            "messages":        [],
            "session_id":      session_id,
            "user_query":      user_message,
            "retrieved_docs":  [],
            "answer":          "",
            "should_escalate": False,
            "escalation_reason": "",
        },
        config=config,
    )
    escalated = result.get("should_escalate", False)
    prefix = "🚨 [ESCALATED] " if escalated else ""
    return f"{prefix}{result['answer']}"


# ── Demo conversation ─────────────────────────────────────────────────────────
def run_demo():
    print("=" * 60)
    print("TechFlow Customer Support Bot — Demo")
    print("=" * 60)

    app = build_support_graph()
    session_id = "demo-session-001"

    conversations = [
        "Hi! I'm interested in TechFlow. What plans do you offer?",
        "How much does the Professional plan cost per year with the discount?",
        "I tried to set up the API integration but I keep getting 401 errors.",
        "I've been having this billing issue for 3 months and nobody has helped me. I'm very frustrated!",
        "Thanks. Can I export my data if I decide to leave?",
    ]

    for i, message in enumerate(conversations, 1):
        print(f"\nUser [{i}]: {message}")
        response = handle_message(app, session_id, message)
        print(f"Aria : {response}")


def run_interactive():
    """Interactive CLI mode."""
    print("=" * 60)
    print("TechFlow Customer Support Bot — Interactive")
    print("Type 'quit' to exit")
    print("=" * 60)

    app = build_support_graph()
    session_id = "interactive-session"

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Aria: Thank you for contacting TechFlow support. Goodbye!")
            break
        if not user_input:
            continue
        response = handle_message(app, session_id, user_input)
        print(f"Aria: {response}")


if __name__ == "__main__":
    run_demo()
    # Uncomment for interactive mode:
    # run_interactive()
