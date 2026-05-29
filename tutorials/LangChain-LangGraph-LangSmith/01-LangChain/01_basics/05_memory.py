"""
05 - Memory & Chat History
===========================
Memory lets your chain/agent remember past messages across turns.
Modern LangChain uses RunnableWithMessageHistory — explicit and composable.

Topics covered:
  1. Manual in-memory history (simplest pattern)
  2. InMemoryChatMessageHistory
  3. RunnableWithMessageHistory (production pattern)
  4. Window memory — keep only the last N messages
  5. Per-session (multi-user) memory with session_id
  6. Summary memory pattern — compress long histories
  7. Persistent memory with file-based store
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. Manual history (simplest) ─────────────────────────────────────────────
def demo_manual_history():
    """Build history yourself — full control, no abstractions."""
    history = []

    def chat(user_message: str) -> str:
        history.append(HumanMessage(content=user_message))
        response = llm.invoke(history)
        history.append(AIMessage(content=response.content))
        return response.content

    print("=== 1. Manual History ===")
    print("Bot:", chat("Hi! My name is Alice and I'm a data scientist."))
    print("Bot:", chat("What's my name and what do I do?"))
    print("Bot:", chat("Suggest one Python library for my work."))


# ── 2. InMemoryChatMessageHistory + RunnableWithMessageHistory ────────────────
def demo_runnable_with_history():
    """
    RunnableWithMessageHistory wires history into any LCEL chain.
    The chain must accept a `history` placeholder in its prompt.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful cooking assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # Store: maps session_id → InMemoryChatMessageHistory
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "cooking-session-1"}}

    print("\n=== 2. RunnableWithMessageHistory ===")
    r1 = chain_with_history.invoke({"input": "I want to make pasta. I have tomatoes and garlic."}, cfg)
    print("Bot:", r1)
    r2 = chain_with_history.invoke({"input": "What sauce can I make with those ingredients?"}, cfg)
    print("Bot:", r2)
    r3 = chain_with_history.invoke({"input": "How long should I cook it?"}, cfg)
    print("Bot:", r3)


# ── 3. Multi-session (multi-user) memory ─────────────────────────────────────
def demo_multi_session():
    """Different session_ids keep conversations isolated."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a personal fitness coach."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session(sid: str) -> InMemoryChatMessageHistory:
        if sid not in store:
            store[sid] = InMemoryChatMessageHistory()
        return store[sid]

    chain_with_history = RunnableWithMessageHistory(
        chain, get_session,
        input_messages_key="input", history_messages_key="history",
    )

    # User Alice — wants to lose weight
    alice_cfg = {"configurable": {"session_id": "alice"}}
    chain_with_history.invoke({"input": "I want to lose 10kg. I can train 3x a week."}, alice_cfg)

    # User Bob — different goal, separate memory
    bob_cfg = {"configurable": {"session_id": "bob"}}
    chain_with_history.invoke({"input": "I want to build muscle. I'm a beginner."}, bob_cfg)

    # Each user continues their own thread
    alice_reply = chain_with_history.invoke({"input": "Remind me of my goal."}, alice_cfg)
    bob_reply   = chain_with_history.invoke({"input": "What did I tell you about my experience?"}, bob_cfg)

    print("\n=== 3. Multi-Session Memory ===")
    print("Alice:", alice_reply)
    print("Bob  :", bob_reply)


# ── 4. Window memory — limit to last N turns ─────────────────────────────────
def demo_window_memory():
    """Keep only the last N messages to limit token usage."""
    MAX_MESSAGES = 6  # keep last 3 human+AI pairs

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer concisely."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    def trim_to_window(messages):
        return messages[-MAX_MESSAGES:] if len(messages) > MAX_MESSAGES else messages

    chain = (
        {
            "history": lambda x: trim_to_window(x["history"]),
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session(sid: str):
        if sid not in store:
            store[sid] = InMemoryChatMessageHistory()
        return store[sid]

    chain_with_history = RunnableWithMessageHistory(
        chain, get_session,
        input_messages_key="input", history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "window-demo"}}

    print("\n=== 4. Window Memory (last 3 pairs) ===")
    turns = [
        "My favourite colour is blue.",
        "I live in Berlin.",
        "I work as a software engineer.",
        "I enjoy hiking on weekends.",  # this will push out the first turn
        "What is my favourite colour?",  # may not remember — windowed!
    ]
    for turn in turns:
        response = chain_with_history.invoke({"input": turn}, cfg)
        print(f"User: {turn}")
        print(f"Bot : {response}\n")


# ── 5. Summary memory pattern ────────────────────────────────────────────────
def demo_summary_memory():
    """
    Instead of keeping all messages, periodically summarise the conversation
    and keep only the summary + recent messages. Greatly reduces token usage
    for long conversations.
    """
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarise this conversation history in 2–3 sentences:\n\n{history}"
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    conversation = [
        ("human", "Hi, I'm Alice, a 30-year-old product manager at a fintech startup."),
        ("ai",    "Nice to meet you Alice! How can I help you today?"),
        ("human", "I'm struggling with prioritisation on my team."),
        ("ai",    "Happy to help! What's the main challenge — too many requests or unclear priorities?"),
        ("human", "Both. We have 20 feature requests and no clear framework."),
        ("ai",    "Consider using the RICE framework: Reach, Impact, Confidence, Effort."),
        ("human", "That sounds useful. Can you explain Impact in more detail?"),
    ]

    # Build history string for summarisation
    history_text = "\n".join(
        f"{role.upper()}: {msg}" for role, msg in conversation
    )
    summary = summarize_chain.invoke({"history": history_text})

    print("\n=== 5. Summary Memory ===")
    print("Conversation summary:")
    print(summary)

    # Now continue with just the summary as context
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coach. Conversation so far:\n{summary}"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "summary": summary,
        "input": "Based on our conversation, what framework should I use?",
    })
    print("\nContinued response:", response)


# ── Interactive demo — try it yourself ───────────────────────────────────────
def interactive_chat():
    """A simple interactive CLI chatbot with persistent memory."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly and knowledgeable assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session(sid):
        if sid not in store:
            store[sid] = InMemoryChatMessageHistory()
        return store[sid]

    chain_with_history = RunnableWithMessageHistory(
        chain, get_session,
        input_messages_key="input", history_messages_key="history",
    )

    print("\n=== Interactive Chat (type 'quit' to exit) ===")
    cfg = {"configurable": {"session_id": "interactive"}}
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        response = chain_with_history.invoke({"input": user_input}, cfg)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    demo_manual_history()
    demo_runnable_with_history()
    demo_multi_session()
    demo_window_memory()
    demo_summary_memory()
    # Uncomment to try the interactive CLI:
    # interactive_chat()
