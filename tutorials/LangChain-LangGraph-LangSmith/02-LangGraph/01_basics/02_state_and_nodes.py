"""
02 - State & Nodes (LangGraph Basics)
=======================================
Deep dive into how state is defined, read, and updated in LangGraph.
Understanding reducers is the key to correct multi-step state management.

Topics covered:
  1. TypedDict state — basic field types
  2. Annotated reducers — operator.add for lists
  3. add_messages reducer for conversation history
  4. Nodes that read multiple state fields
  5. Partial state updates (nodes return only changed fields)
  6. State snapshots — inspecting state mid-graph
  7. Complex nested state
  8. Default values with TypedDict
"""

import operator
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. Plain TypedDict state ──────────────────────────────────────────────────
class BasicState(TypedDict):
    name: str
    age: int
    score: float
    active: bool
    tags: list[str]


def node_a(state: BasicState) -> dict:
    """Increment score and mark active."""
    return {
        "score": state["score"] + 10.0,
        "active": True,
    }


def node_b(state: BasicState) -> dict:
    """Append a tag."""
    # Without a reducer, returning a list REPLACES the existing list
    existing = state.get("tags") or []
    return {"tags": existing + ["processed"]}


def demo_basic_state():
    graph = StateGraph(BasicState)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    app = graph.compile()

    initial: BasicState = {"name": "Alice", "age": 30, "score": 0.0, "active": False, "tags": []}
    result = app.invoke(initial)

    print("=== 1. Basic TypedDict State ===")
    print(f"  Name  : {result['name']}")
    print(f"  Score : {result['score']}  (was 0.0)")
    print(f"  Active: {result['active']} (was False)")
    print(f"  Tags  : {result['tags']}")


# ── 2. Annotated reducer — operator.add ──────────────────────────────────────
class AccumulatorState(TypedDict):
    total: int
    log: Annotated[list[str], operator.add]   # appends, never replaces


def add_five_node(state: AccumulatorState) -> dict:
    return {"total": state["total"] + 5, "log": ["added 5"]}


def multiply_two_node(state: AccumulatorState) -> dict:
    return {"total": state["total"] * 2, "log": ["multiplied by 2"]}


def negate_node(state: AccumulatorState) -> dict:
    return {"total": -state["total"], "log": ["negated"]}


def demo_reducer_state():
    graph = StateGraph(AccumulatorState)
    graph.add_node("add_five",    add_five_node)
    graph.add_node("multiply",    multiply_two_node)
    graph.add_node("negate",      negate_node)
    graph.set_entry_point("add_five")
    graph.add_edge("add_five",    "multiply")
    graph.add_edge("multiply",    "negate")
    graph.add_edge("negate",      END)
    app = graph.compile()

    result = app.invoke({"total": 10, "log": []})

    print("\n=== 2. Annotated Reducer (operator.add) ===")
    print(f"  Final total: {result['total']}")  # ((10+5)*2)*-1 = -30
    print(f"  Log (all steps accumulated): {result['log']}")


# ── 3. add_messages reducer ───────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


def chat_node(state: ChatState) -> dict:
    """Call the LLM and append its response to messages."""
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],        # add_messages will append, not replace
        "turn_count": state["turn_count"] + 1,
    }


def demo_messages_reducer():
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    system = SystemMessage(content="You are a Socratic philosophy teacher. Always answer with a question.")

    # Turn 1
    state = {"messages": [system, HumanMessage(content="What is truth?")], "turn_count": 0}
    state = app.invoke(state)
    print("\n=== 3. add_messages Reducer ===")
    print(f"Turn 1 — Bot: {state['messages'][-1].content}")

    # Turn 2 — previous messages preserved automatically
    state["messages"].append(HumanMessage(content="That's a good question. But what do YOU think?"))
    state = app.invoke(state)
    print(f"Turn 2 — Bot: {state['messages'][-1].content}")
    print(f"Total turns: {state['turn_count']}")
    print(f"Total messages in history: {len(state['messages'])}")


# ── 4. Complex nested state ───────────────────────────────────────────────────
class TaskItem(TypedDict):
    id: int
    description: str
    done: bool


class ProjectState(TypedDict):
    project_name: str
    tasks: list[TaskItem]
    completed_count: int
    notes: Annotated[list[str], operator.add]
    status: str


def add_tasks_node(state: ProjectState) -> dict:
    """Add initial tasks to the project."""
    tasks = [
        {"id": 1, "description": "Define requirements",   "done": False},
        {"id": 2, "description": "Design architecture",   "done": False},
        {"id": 3, "description": "Implement core module", "done": False},
        {"id": 4, "description": "Write tests",           "done": False},
    ]
    return {"tasks": tasks, "notes": ["Tasks loaded from backlog"]}


def execute_tasks_node(state: ProjectState) -> dict:
    """Mark all tasks as done (simulate execution)."""
    completed = [dict(t, done=True) for t in state["tasks"]]
    return {
        "tasks": completed,
        "completed_count": len(completed),
        "notes": [f"Completed {len(completed)} tasks"],
    }


def close_project_node(state: ProjectState) -> dict:
    all_done = all(t["done"] for t in state["tasks"])
    status = "CLOSED" if all_done else "BLOCKED"
    return {"status": status, "notes": [f"Project status set to {status}"]}


def demo_complex_state():
    graph = StateGraph(ProjectState)
    graph.add_node("add_tasks",    add_tasks_node)
    graph.add_node("execute",      execute_tasks_node)
    graph.add_node("close",        close_project_node)
    graph.set_entry_point("add_tasks")
    graph.add_edge("add_tasks",    "execute")
    graph.add_edge("execute",      "close")
    graph.add_edge("close",        END)
    app = graph.compile()

    initial: ProjectState = {
        "project_name": "LangGraph Tutorial",
        "tasks": [],
        "completed_count": 0,
        "notes": ["Project initialised"],
        "status": "OPEN",
    }
    result = app.invoke(initial)

    print("\n=== 4. Complex Nested State ===")
    print(f"  Project        : {result['project_name']}")
    print(f"  Status         : {result['status']}")
    print(f"  Tasks completed: {result['completed_count']}")
    print("  Notes timeline :")
    for n in result["notes"]:
        print(f"    • {n}")


# ── 5. State inspection with MemorySaver ──────────────────────────────────────
def demo_state_inspection():
    """
    Compile with a checkpointer to enable get_state() at any point.
    Each invocation is identified by a thread_id in the config.
    """
    class CountState(TypedDict):
        count: int
        history: Annotated[list[int], operator.add]

    def increment(state: CountState) -> dict:
        new = state["count"] + 1
        return {"count": new, "history": [new]}

    def double(state: CountState) -> dict:
        new = state["count"] * 2
        return {"count": new, "history": [new]}

    graph = StateGraph(CountState)
    graph.add_node("increment", increment)
    graph.add_node("double",    double)
    graph.set_entry_point("increment")
    graph.add_edge("increment", "double")
    graph.add_edge("double",    END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "demo-thread"}}

    result = app.invoke({"count": 5, "history": [5]}, config=config)

    print("\n=== 5. State Inspection with Checkpointer ===")
    print(f"  Final count  : {result['count']}")   # (5+1)*2 = 12
    print(f"  History      : {result['history']}")

    # Inspect the persisted state
    state_snapshot = app.get_state(config)
    print(f"\n  Snapshot values: {state_snapshot.values}")
    print(f"  Next nodes     : {state_snapshot.next}")  # empty = finished


if __name__ == "__main__":
    demo_basic_state()
    demo_reducer_state()
    demo_messages_reducer()
    demo_complex_state()
    demo_state_inspection()
