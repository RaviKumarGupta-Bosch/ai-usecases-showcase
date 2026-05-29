"""
03 - Conditional Edges (LangGraph Basics)
==========================================
Conditional edges let the graph take different paths based on state.
They are the foundation for loops, retries, and agent reasoning cycles.

Topics covered:
  1. add_conditional_edges with a router function
  2. Router returning END to terminate early
  3. Loop with a step counter (max_iterations guard)
  4. Retry pattern — loop until condition met
  5. Multi-branch routing (more than 2 options)
  6. Nested conditional logic
  7. Agent-style think → act → observe loop
"""

import operator
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. Basic conditional edge ────────────────────────────────────────────────
class ScoreState(TypedDict):
    score: int
    result: str


def evaluate_score_node(state: ScoreState) -> dict:
    """Set result label based on current score."""
    score = state["score"]
    if score >= 90:
        result = "excellent"
    elif score >= 70:
        result = "good"
    else:
        result = "needs_improvement"
    return {"result": result}


def router_score(state: ScoreState) -> Literal["excellent", "good", "needs_improvement"]:
    """Router function: reads state, returns the name of the next node."""
    return state["result"]


def excellent_node(state: ScoreState) -> dict:
    print("  → Taking EXCELLENT path")
    return {}


def good_node(state: ScoreState) -> dict:
    print("  → Taking GOOD path")
    return {}


def improvement_node(state: ScoreState) -> dict:
    print("  → Taking NEEDS IMPROVEMENT path")
    return {}


def demo_basic_conditional():
    graph = StateGraph(ScoreState)

    graph.add_node("evaluate",    evaluate_score_node)
    graph.add_node("excellent",   excellent_node)
    graph.add_node("good",        good_node)
    graph.add_node("improvement", improvement_node)

    graph.set_entry_point("evaluate")

    # Route evaluate → one of three nodes based on result field
    graph.add_conditional_edges(
        "evaluate",
        router_score,
        {
            "excellent":        "excellent",
            "good":             "good",
            "needs_improvement": "improvement",
        },
    )

    # All paths lead to END
    graph.add_edge("excellent",   END)
    graph.add_edge("good",        END)
    graph.add_edge("improvement", END)

    app = graph.compile()

    print("=== 1. Basic Conditional Edges ===")
    for score in [95, 75, 45]:
        result = app.invoke({"score": score, "result": ""})
        print(f"  Score {score} → {result['result']}")


# ── 2. Loop with max-steps guard ──────────────────────────────────────────────
class IterState(TypedDict):
    value: int
    steps: int
    log: Annotated[list[str], operator.add]
    max_steps: int


def transform_node(state: IterState) -> dict:
    """Apply transformation: add 3 if even, subtract 1 if odd."""
    v = state["value"]
    new_v = v + 3 if v % 2 == 0 else v - 1
    return {
        "value": new_v,
        "steps": state["steps"] + 1,
        "log":   [f"step {state['steps']+1}: {v} → {new_v}"],
    }


def should_continue_loop(state: IterState) -> Literal["continue", "stop"]:
    """Loop until value reaches 20 or max_steps exceeded."""
    if state["value"] >= 20:
        return "stop"
    if state["steps"] >= state["max_steps"]:
        print(f"  [SAFETY] Max steps ({state['max_steps']}) reached — stopping")
        return "stop"
    return "continue"


def demo_loop_with_guard():
    graph = StateGraph(IterState)
    graph.add_node("transform", transform_node)

    graph.set_entry_point("transform")
    graph.add_conditional_edges(
        "transform",
        should_continue_loop,
        {"continue": "transform", "stop": END},
    )

    app = graph.compile()

    print("\n=== 2. Loop with Max-Steps Guard ===")
    result = app.invoke({"value": 1, "steps": 0, "log": [], "max_steps": 15})
    print(f"  Final value : {result['value']}")
    print(f"  Steps taken : {result['steps']}")
    print(f"  Log         : {result['log']}")


# ── 3. Retry pattern ──────────────────────────────────────────────────────────
class RetryState(TypedDict):
    question: str
    answer: str
    attempts: int
    max_attempts: int
    quality_score: float


def generate_answer_node(state: RetryState) -> dict:
    """Generate an answer with the LLM."""
    response = llm.invoke(
        f"Answer this question concisely in 1-2 sentences: {state['question']}"
    )
    return {
        "answer": response.content,
        "attempts": state["attempts"] + 1,
    }


def evaluate_quality_node(state: RetryState) -> dict:
    """Score the quality of the answer."""
    from pydantic import BaseModel

    class QualityScore(BaseModel):
        score: float
        feedback: str

    quality_llm = llm.with_structured_output(QualityScore)
    evaluation = quality_llm.invoke(
        f"Rate the quality of this answer on a scale of 0-1. "
        f"Question: {state['question']}\nAnswer: {state['answer']}\n"
        "Score 0.8+ for concise, accurate answers. Lower for vague or incorrect ones."
    )
    return {"quality_score": evaluation.score}


def retry_router(state: RetryState) -> Literal["retry", "accept"]:
    if state["quality_score"] >= 0.75 or state["attempts"] >= state["max_attempts"]:
        return "accept"
    return "retry"


def demo_retry_pattern():
    graph = StateGraph(RetryState)
    graph.add_node("generate",  generate_answer_node)
    graph.add_node("evaluate",  evaluate_quality_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        retry_router,
        {"retry": "generate", "accept": END},
    )

    app = graph.compile()

    print("\n=== 3. Retry Pattern ===")
    result = app.invoke({
        "question": "What is the capital of France?",
        "answer": "",
        "attempts": 0,
        "max_attempts": 3,
        "quality_score": 0.0,
    })
    print(f"  Attempts      : {result['attempts']}")
    print(f"  Quality score : {result['quality_score']:.2f}")
    print(f"  Final answer  : {result['answer']}")


# ── 4. Agent think → act loop ─────────────────────────────────────────────────
# A minimal manual ReAct loop (for pedagogy — use prebuilt ToolNode in production)

import json
import math

TOOLS = {
    "calculator": lambda expr: str(eval(expr, {"__builtins__": {}}, {"math": math})),
    "upper_case": lambda text: text.upper(),
    "word_count": lambda text: str(len(text.split())),
}

TOOLS_SPEC = """
Available tools (call one at a time):
- calculator(expr): evaluate a math expression, e.g. calculator("2**10")
- upper_case(text): convert text to upper case
- word_count(text): count words in text

To call a tool, respond ONLY with JSON like: {"tool": "calculator", "args": "2+2"}
When you have the final answer, respond with: {"final_answer": "your answer here"}
"""


class AgentLoopState(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    max_iterations: int
    final_answer: str


def agent_think_node(state: AgentLoopState) -> dict:
    """LLM decides: call a tool or answer directly."""
    from langchain_core.messages import SystemMessage
    history = state["messages"] or []
    if not history:
        history = [HumanMessage(content=state["question"])]

    system = SystemMessage(content=TOOLS_SPEC)
    response = llm.invoke([system] + history)
    return {
        "messages": [HumanMessage(content=state["question"])] if not state["messages"] else [],
        "messages": [response],
        "iterations": state["iterations"] + 1,
    }


def agent_act_node(state: AgentLoopState) -> dict:
    """Execute the tool the LLM chose."""
    last = state["messages"][-1]
    try:
        call = json.loads(last.content)
        if "final_answer" in call:
            return {"final_answer": call["final_answer"]}
        tool_name = call["tool"]
        tool_args = call["args"]
        if tool_name in TOOLS:
            result = TOOLS[tool_name](tool_args)
            observation = f"Tool '{tool_name}' returned: {result}"
        else:
            observation = f"Unknown tool: {tool_name}"
    except (json.JSONDecodeError, KeyError, Exception) as e:
        observation = f"Error: {e}"

    return {"messages": [HumanMessage(content=f"[Observation] {observation}")]}


def agent_router(state: AgentLoopState) -> Literal["think", "end"]:
    if state["final_answer"]:
        return "end"
    if state["iterations"] >= state["max_iterations"]:
        return "end"
    return "think"


def demo_agent_loop():
    graph = StateGraph(AgentLoopState)
    graph.add_node("think", agent_think_node)
    graph.add_node("act",   agent_act_node)

    graph.set_entry_point("think")
    graph.add_edge("think", "act")
    graph.add_conditional_edges(
        "act",
        agent_router,
        {"think": "think", "end": END},
    )

    app = graph.compile()

    print("\n=== 4. Minimal Agent Loop (think → act → observe) ===")
    result = app.invoke({
        "question": "What is 2 to the power of 10? Use the calculator tool.",
        "messages": [],
        "iterations": 0,
        "max_iterations": 6,
        "final_answer": "",
    })
    print(f"  Iterations    : {result['iterations']}")
    print(f"  Final answer  : {result['final_answer']}")
    print(f"  Message count : {len(result['messages'])}")


if __name__ == "__main__":
    demo_basic_conditional()
    demo_loop_with_guard()
    demo_retry_pattern()
    demo_agent_loop()
