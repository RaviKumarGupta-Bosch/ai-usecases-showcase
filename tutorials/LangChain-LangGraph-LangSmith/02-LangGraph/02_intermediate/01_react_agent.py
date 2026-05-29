"""
01 - ReAct Agent from Scratch (LangGraph Intermediate)
=======================================================
Build a full ReAct (Reason + Act) agent using LangGraph primitives.
This is what create_react_agent does under the hood.

Topics covered:
  1. Define tools with @tool decorator
  2. Bind tools to LLM with llm.bind_tools()
  3. ToolNode — automatic tool execution
  4. tools_condition — route based on tool_calls presence
  5. Full ReAct graph: agent → tools → agent loop
  6. Streaming agent execution
  7. Inspecting tool calls and results
"""

import math
import json
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Define tools ──────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, abs, round.
    Example: calculator("sqrt(144) + 2**3")
    """
    safe_names = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "abs": abs, "round": round, "pi": math.pi, "e": math.e,
        "floor": math.floor, "ceil": math.ceil,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)
        return str(result)
    except Exception as ex:
        return f"Error evaluating expression: {ex}"


@tool
def get_word_stats(text: str) -> str:
    """
    Analyse text and return statistics: word count, sentence count, avg word length.
    Example: get_word_stats("Hello world. This is a test.")
    """
    words = text.split()
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    return json.dumps({
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": round(avg_len, 2),
        "char_count": len(text),
    })


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    Supported: celsius↔fahrenheit, km↔miles, kg↔pounds, meters↔feet.
    Example: unit_converter(100, "celsius", "fahrenheit")
    """
    conversions = {
        ("celsius",    "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"):    lambda x: (x - 32) * 5/9,
        ("km",         "miles"):      lambda x: x * 0.621371,
        ("miles",      "km"):         lambda x: x * 1.60934,
        ("kg",         "pounds"):     lambda x: x * 2.20462,
        ("pounds",     "kg"):         lambda x: x * 0.453592,
        ("meters",     "feet"):       lambda x: x * 3.28084,
        ("feet",       "meters"):     lambda x: x * 0.3048,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"
    return f"Unsupported conversion: {from_unit} → {to_unit}"


@tool
def string_manipulator(text: str, operation: str) -> str:
    """
    Perform string operations on text.
    Operations: upper, lower, reverse, title, count_vowels, palindrome_check.
    Example: string_manipulator("Hello World", "reverse")
    """
    ops = {
        "upper":          lambda t: t.upper(),
        "lower":          lambda t: t.lower(),
        "reverse":        lambda t: t[::-1],
        "title":          lambda t: t.title(),
        "count_vowels":   lambda t: str(sum(c in "aeiouAEIOU" for c in t)),
        "palindrome_check": lambda t: str(t.lower().replace(" ", "") == t.lower().replace(" ", "")[::-1]),
    }
    if operation in ops:
        return ops[operation](text)
    return f"Unknown operation: {operation}. Choose from: {list(ops.keys())}"


TOOLS = [calculator, get_word_stats, unit_converter, string_manipulator]

# Bind tools to LLM — this attaches the JSON tool schemas to every API call
llm_with_tools = llm.bind_tools(TOOLS)


# ── Graph state ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Agent node ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant with access to several tools:
- calculator: for math operations
- get_word_stats: to analyse text statistics
- unit_converter: to convert between units
- string_manipulator: for text transformations

Use tools whenever calculations or data lookups are needed.
Think step by step. Use multiple tools if the question requires it."""


def agent_node(state: AgentState) -> dict:
    """LLM decides whether to call a tool or respond to the user."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ── 1. Build the ReAct graph ──────────────────────────────────────────────────
def build_react_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))   # auto-executes tool calls

    graph.set_entry_point("agent")

    # tools_condition: if last message has tool_calls → "tools", else → END
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")           # always return to agent after tools

    return graph.compile()


def demo_react_basic():
    app = build_react_graph()

    print("=== 1. ReAct Agent — Basic Queries ===")

    questions = [
        "What is the square root of 2025, multiplied by 5?",
        "Convert 100 kilometers to miles, then tell me how many feet that is.",
        "Is the word 'racecar' a palindrome?",
    ]

    for q in questions:
        result = app.invoke({"messages": [HumanMessage(content=q)]})
        final = result["messages"][-1].content
        tool_calls_made = sum(
            1 for m in result["messages"]
            if hasattr(m, "tool_calls") and m.tool_calls
        )
        print(f"\nQ: {q}")
        print(f"Tool calls: {tool_calls_made}")
        print(f"A: {final}")


# ── 2. Multi-step tool use ─────────────────────────────────────────────────────
def demo_multi_step():
    app = build_react_graph()

    question = (
        "I have a text: 'The quick brown fox jumps over the lazy dog'. "
        "First get its word stats. Then convert the word count to binary (use calculator). "
        "Finally reverse the original text."
    )

    print("\n=== 2. Multi-Step Tool Use ===")
    result = app.invoke({"messages": [HumanMessage(content=question)]})

    print("Execution trace:")
    for i, msg in enumerate(result["messages"]):
        name = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [{i}] {name} → tool call: {tc['name']}({tc['args']})")
        elif hasattr(msg, "name") and msg.name:
            content_preview = str(msg.content)[:60]
            print(f"  [{i}] ToolMessage({msg.name}) → {content_preview}")
        elif name == "HumanMessage":
            print(f"  [{i}] Human → {str(msg.content)[:60]}")
        else:
            print(f"  [{i}] {name} → {str(msg.content)[:80]}")

    print(f"\nFinal answer: {result['messages'][-1].content}")


# ── 3. Streaming the agent ────────────────────────────────────────────────────
def demo_streaming_agent():
    app = build_react_graph()

    print("\n=== 3. Streaming Agent Execution ===")
    question = "What is 15% of 840? Then convert that result from kg to pounds."

    print(f"Question: {question}\n")
    for step in app.stream({"messages": [HumanMessage(content=question)]}):
        node_name = next(iter(step.keys()))
        node_output = step[node_name]
        if "messages" in node_output:
            last_msg = node_output["messages"][-1]
            msg_type = type(last_msg).__name__
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"  [{node_name}] Calling {tc['name']}({tc['args']})")
            elif hasattr(last_msg, "name") and last_msg.name:
                print(f"  [{node_name}] Tool result: {last_msg.content[:80]}")
            else:
                content = getattr(last_msg, "content", "")
                if content:
                    print(f"  [{node_name}] {content[:120]}")


if __name__ == "__main__":
    demo_react_basic()
    demo_multi_step()
    demo_streaming_agent()
