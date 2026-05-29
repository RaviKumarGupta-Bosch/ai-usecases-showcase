"""
01 - Simple Graph (LangGraph Basics)
=====================================
LangGraph models stateful workflows as directed graphs.
Nodes = functions that read/write state.
Edges = define execution order.

Topics covered:
  1. StateGraph and TypedDict state definition
  2. add_node, add_edge, set_entry_point
  3. Compile and invoke
  4. END sentinel — terminating the graph
  5. Multiple nodes in sequence
  6. Inspecting graph structure with Mermaid
  7. Streaming graph execution
"""

import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. Minimal Graph — single node ───────────────────────────────────────────
class SimpleState(TypedDict):
    message: str
    result: str


def greet_node(state: SimpleState) -> SimpleState:
    """A simple node that transforms the state."""
    name = state["message"]
    return {"result": f"Hello, {name}! Welcome to LangGraph."}


def demo_minimal_graph():
    graph = StateGraph(SimpleState)
    graph.add_node("greet", greet_node)
    graph.set_entry_point("greet")
    graph.add_edge("greet", END)

    app = graph.compile()

    print("=== 1. Minimal Graph ===")
    result = app.invoke({"message": "Alice", "result": ""})
    print(f"Input : Alice")
    print(f"Output: {result['result']}")


# ── 2. Multi-node sequential pipeline ────────────────────────────────────────
class PipelineState(TypedDict):
    raw_text: str
    cleaned_text: str
    word_count: int
    summary: str


def clean_node(state: PipelineState) -> dict:
    """Remove extra whitespace and lowercase the text."""
    cleaned = " ".join(state["raw_text"].split()).lower()
    return {"cleaned_text": cleaned}


def count_node(state: PipelineState) -> dict:
    """Count words in the cleaned text."""
    count = len(state["cleaned_text"].split())
    return {"word_count": count}


def summarise_node(state: PipelineState) -> dict:
    """Summarise the text using the LLM."""
    response = llm.invoke(f"Summarise in one sentence: {state['cleaned_text']}")
    return {"summary": response.content}


def demo_pipeline_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("clean",     clean_node)
    graph.add_node("count",     count_node)
    graph.add_node("summarise", summarise_node)

    graph.set_entry_point("clean")
    graph.add_edge("clean",     "count")
    graph.add_edge("count",     "summarise")
    graph.add_edge("summarise", END)

    app = graph.compile()

    raw = """
    LangGraph is a library for building stateful, multi-actor applications with LLMs,
    used to create agent and multi-agent workflows. Compared to other LLM frameworks,
    it offers these core benefits: cycles, controllability, and persistence.
    LangGraph allows you to define flows that involve cycles, which are essential for
    most agentic architectures, differentiating it from DAG-based solutions.
    """

    print("\n=== 2. Multi-Node Pipeline ===")
    result = app.invoke({
        "raw_text": raw,
        "cleaned_text": "",
        "word_count": 0,
        "summary": "",
    })
    print(f"Word count : {result['word_count']}")
    print(f"Summary    : {result['summary']}")


# ── 3. Message-based graph (ChatBot pattern) ──────────────────────────────────
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot_node(state: MessagesState) -> dict:
    """Invoke the LLM with the current message history."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def demo_message_graph():
    """
    Messages state uses the `add_messages` reducer:
    instead of replacing the list, new messages are appended.
    This is the foundation of every LangGraph chatbot.
    """
    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot_node)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", END)

    app = graph.compile()

    print("\n=== 3. Message-Based Graph ===")
    result = app.invoke({
        "messages": [HumanMessage(content="What are the three laws of robotics?")]
    })
    print("Bot:", result["messages"][-1].content)

    # Multi-turn: add a follow-up with the full history
    result2 = app.invoke({
        "messages": result["messages"] + [
            HumanMessage(content="Who created those laws?")
        ]
    })
    print("\nBot (follow-up):", result2["messages"][-1].content)


# ── 4. Graph with multiple entry paths ───────────────────────────────────────
class CounterState(TypedDict):
    value: int
    steps: Annotated[list[str], operator.add]


def add_ten_node(state: CounterState) -> dict:
    return {"value": state["value"] + 10, "steps": ["added 10"]}


def double_node(state: CounterState) -> dict:
    return {"value": state["value"] * 2, "steps": ["doubled"]}


def subtract_one_node(state: CounterState) -> dict:
    return {"value": state["value"] - 1, "steps": ["subtracted 1"]}


def demo_chained_transforms():
    graph = StateGraph(CounterState)

    graph.add_node("add_ten",      add_ten_node)
    graph.add_node("double",       double_node)
    graph.add_node("subtract_one", subtract_one_node)

    graph.set_entry_point("add_ten")
    graph.add_edge("add_ten",      "double")
    graph.add_edge("double",       "subtract_one")
    graph.add_edge("subtract_one", END)

    app = graph.compile()

    print("\n=== 4. Chained Transforms: 5 → +10 → ×2 → -1 ===")
    result = app.invoke({"value": 5, "steps": []})
    print(f"Result  : {result['value']}")   # ((5+10)*2)-1 = 29
    print(f"Steps   : {result['steps']}")


# ── 5. Streaming graph execution ─────────────────────────────────────────────
def demo_streaming_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("clean",     clean_node)
    graph.add_node("count",     count_node)
    graph.add_node("summarise", summarise_node)
    graph.set_entry_point("clean")
    graph.add_edge("clean",     "count")
    graph.add_edge("count",     "summarise")
    graph.add_edge("summarise", END)
    app = graph.compile()

    print("\n=== 5. Streaming Graph Execution ===")
    initial_state = {
        "raw_text":     "Machine learning is a branch of AI. It learns from data. Models improve with experience.",
        "cleaned_text": "",
        "word_count":   0,
        "summary":      "",
    }

    # stream() yields state after each node executes
    for step in app.stream(initial_state):
        node_name = next(iter(step.keys()))
        print(f"  After '{node_name}': {step[node_name]}")


# ── 6. Get graph structure ────────────────────────────────────────────────────
def demo_graph_structure():
    graph = StateGraph(PipelineState)
    graph.add_node("clean",     clean_node)
    graph.add_node("count",     count_node)
    graph.add_node("summarise", summarise_node)
    graph.set_entry_point("clean")
    graph.add_edge("clean",     "count")
    graph.add_edge("count",     "summarise")
    graph.add_edge("summarise", END)
    app = graph.compile()

    print("\n=== 6. Graph Structure (Mermaid) ===")
    try:
        print(app.get_graph().draw_mermaid())
    except Exception:
        print("  [Install: pip install grandalf to render Mermaid diagrams]")
        print("  Nodes:", list(graph.nodes.keys()))


if __name__ == "__main__":
    demo_minimal_graph()
    demo_pipeline_graph()
    demo_message_graph()
    demo_chained_transforms()
    demo_streaming_graph()
    demo_graph_structure()
