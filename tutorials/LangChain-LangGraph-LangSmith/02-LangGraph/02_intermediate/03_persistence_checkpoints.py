"""
03 - Persistence & Checkpoints (LangGraph Intermediate)
========================================================
Checkpointers save graph state after every node, enabling:
- Conversation memory that survives restarts
- Resume-from-checkpoint after interrupts
- Time-travel debugging (replay from any past state)

Topics covered:
  1. MemorySaver — in-memory checkpointer (dev/testing)
  2. SqliteSaver — persistent checkpointer (production-ready)
  3. thread_id — isolate separate conversation sessions
  4. get_state_history() — list all checkpoints
  5. Time-travel: replay from a specific checkpoint
  6. Multi-user sessions with different thread_ids
  7. Checkpoint metadata
"""

import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Shared chatbot graph ──────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn: int


SYSTEM = SystemMessage(content=
    "You are a friendly AI assistant. Keep answers concise (1-2 sentences). "
    "Remember the conversation context."
)


def chat_node(state: ChatState) -> dict:
    messages = [SYSTEM] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response], "turn": state["turn"] + 1}


def build_chatbot(checkpointer):
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)


# ── 1. MemorySaver — in-memory checkpoints ────────────────────────────────────
def demo_memory_saver():
    """MemorySaver stores checkpoints in a Python dict. Lost when process exits."""
    memory = MemorySaver()
    app = build_chatbot(memory)

    config = {"configurable": {"thread_id": "session-001"}}

    turns = [
        "Hi! My name is Alex and I'm learning LangGraph.",
        "What was my name again?",
        "What am I learning?",
    ]

    print("=== 1. MemorySaver — Persistent Conversation ===")
    for turn in turns:
        result = app.invoke(
            {"messages": [HumanMessage(content=turn)], "turn": 0},
            config=config,
        )
        print(f"  User: {turn}")
        print(f"  Bot : {result['messages'][-1].content}")
        print()


# ── 2. SqliteSaver — persistent disk checkpoints ─────────────────────────────
def demo_sqlite_saver():
    """
    SqliteSaver stores checkpoints in a SQLite database file.
    Survives process restarts. Use for production chatbots.
    """
    import tempfile, os
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Create a temp DB file (in production, use a fixed path)
    db_path = os.path.join(tempfile.gettempdir(), "langgraph_tutorial.db")

    print(f"\n=== 2. SqliteSaver — Persistent to Disk ===")
    print(f"  DB path: {db_path}")

    config = {"configurable": {"thread_id": "sqlite-session-001"}}

    # First "session" — write some turns
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_chatbot(checkpointer)
        app.invoke(
            {"messages": [HumanMessage(content="Remember: my favourite colour is blue.")], "turn": 0},
            config=config,
        )
        app.invoke(
            {"messages": [HumanMessage(content="Also, I work as a data scientist.")], "turn": 0},
            config=config,
        )
        print("  [Session 1] Saved 2 turns to SQLite.")

    # Second "session" — load from disk, ask about earlier context
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_chatbot(checkpointer)
        result = app.invoke(
            {"messages": [HumanMessage(content="What's my favourite colour and job?")], "turn": 0},
            config=config,
        )
        print(f"  [Session 2] Bot recalls: {result['messages'][-1].content}")

    # Clean up
    os.remove(db_path)
    print(f"  [Cleaned up {db_path}]")


# ── 3. Multiple thread_ids — isolated sessions ────────────────────────────────
def demo_multi_thread():
    """
    Each thread_id is an isolated conversation.
    Different users can run concurrently without interfering.
    """
    memory = MemorySaver()
    app = build_chatbot(memory)

    # Two different users
    alice_config = {"configurable": {"thread_id": "user-alice"}}
    bob_config   = {"configurable": {"thread_id": "user-bob"}}

    # Alice's conversation
    app.invoke({"messages": [HumanMessage(content="I love Python programming.")], "turn": 0}, config=alice_config)
    app.invoke({"messages": [HumanMessage(content="My favourite library is Pandas.")], "turn": 0}, config=alice_config)

    # Bob's conversation
    app.invoke({"messages": [HumanMessage(content="I prefer JavaScript and React.")], "turn": 0}, config=bob_config)

    # Ask each about their preferences
    alice_result = app.invoke({"messages": [HumanMessage(content="What language do I like?")], "turn": 0}, config=alice_config)
    bob_result   = app.invoke({"messages": [HumanMessage(content="What language do I like?")], "turn": 0}, config=bob_config)

    print("\n=== 3. Multi-Thread Isolation ===")
    print(f"  Alice's bot: {alice_result['messages'][-1].content}")
    print(f"  Bob's bot  : {bob_result['messages'][-1].content}")


# ── 4. get_state_history — all checkpoints ────────────────────────────────────
def demo_state_history():
    """
    get_state_history() returns all saved checkpoints for a thread.
    Each checkpoint has: config (with checkpoint_id), values, next, metadata.
    """
    memory = MemorySaver()
    app = build_chatbot(memory)

    config = {"configurable": {"thread_id": "history-demo"}}

    messages = [
        "The Eiffel Tower is in Paris.",
        "The Great Wall is in China.",
        "The Colosseum is in Rome.",
    ]

    for msg in messages:
        app.invoke(
            {"messages": [HumanMessage(content=msg)], "turn": 0},
            config=config,
        )

    print("\n=== 4. State History ===")
    history = list(app.get_state_history(config))
    print(f"  Total checkpoints: {len(history)}")

    for i, checkpoint in enumerate(history):
        msg_count  = len(checkpoint.values.get("messages", []))
        turn       = checkpoint.values.get("turn", 0)
        next_nodes = checkpoint.next
        print(f"  [{i}] turn={turn} | messages={msg_count} | next={next_nodes}")


# ── 5. Time-travel: replay from a specific checkpoint ─────────────────────────
def demo_time_travel():
    """
    Resume from any past checkpoint to explore alternate execution paths.
    Useful for debugging and testing "what-if" scenarios.
    """
    memory = MemorySaver()
    app = build_chatbot(memory)

    config = {"configurable": {"thread_id": "time-travel-demo"}}

    # Build some history
    app.invoke({"messages": [HumanMessage(content="My name is Jordan.")], "turn": 0}, config=config)
    app.invoke({"messages": [HumanMessage(content="I live in Berlin.")], "turn": 0}, config=config)
    app.invoke({"messages": [HumanMessage(content="I work in finance.")], "turn": 0}, config=config)

    # Capture all checkpoints
    history = list(app.get_state_history(config))
    print(f"\n=== 5. Time Travel ===")
    print(f"  Total checkpoints: {len(history)}")

    if len(history) >= 2:
        # Go back to second-to-last checkpoint (after turn 1)
        past_checkpoint = history[-2]  # oldest = last in list
        past_config = past_checkpoint.config
        past_turn = past_checkpoint.values.get("turn", 0)
        print(f"  Rewinding to turn {past_turn}...")

        # Resume from past checkpoint with a different question
        result = app.invoke(
            {"messages": [HumanMessage(content="What have I told you so far?")], "turn": 0},
            config=past_config,
        )
        print(f"  Bot from past state: {result['messages'][-1].content}")


# ── 6. Checkpoint metadata ────────────────────────────────────────────────────
def demo_checkpoint_metadata():
    """Show the full metadata structure of a checkpoint."""
    memory = MemorySaver()
    app = build_chatbot(memory)

    config = {"configurable": {"thread_id": "metadata-demo"}}
    app.invoke({"messages": [HumanMessage(content="Hello!")], "turn": 0}, config=config)

    snapshot = app.get_state(config)

    print("\n=== 6. Checkpoint Metadata ===")
    print(f"  Thread ID    : {snapshot.config['configurable']['thread_id']}")
    print(f"  Checkpoint ID: {snapshot.config['configurable'].get('checkpoint_id', 'N/A')[:16]}...")
    print(f"  Next nodes   : {snapshot.next}")
    print(f"  Created at   : {snapshot.metadata.get('created_at', 'N/A')}")
    print(f"  Step number  : {snapshot.metadata.get('step', 'N/A')}")
    print(f"  State keys   : {list(snapshot.values.keys())}")
    print(f"  Message count: {len(snapshot.values.get('messages', []))}")


if __name__ == "__main__":
    demo_memory_saver()
    demo_sqlite_saver()
    demo_multi_thread()
    demo_state_history()
    demo_time_travel()
    demo_checkpoint_metadata()
