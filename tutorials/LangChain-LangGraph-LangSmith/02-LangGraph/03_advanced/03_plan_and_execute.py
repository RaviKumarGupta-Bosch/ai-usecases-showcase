"""
03 - Plan-and-Execute Agent (LangGraph Advanced)
=================================================
Plan-and-Execute separates planning from execution:
  1. Planner  — breaks the goal into ordered steps
  2. Executor — executes the current step using tools
  3. Re-planner — updates the plan based on progress, or returns the final answer

This pattern handles long-horizon tasks that require adaptive planning.

Topics covered:
  1. PlanExecute state (plan, past_steps, response, input)
  2. Planner: LLM with structured output → ordered step list
  3. Executor: ReAct agent executes one step at a time
  4. Re-planner: revise plan or produce final answer
  5. Loop: plan → execute → replan → (finish | continue)
  6. MemorySaver checkpointing
"""

import operator
from typing import TypedDict, Annotated, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Tools for the executor agent ──────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    knowledge = {
        "python history": "Python was created by Guido van Rossum and first released in 1991. Python 2.0 came in 2000, Python 3.0 in 2008.",
        "python features": "Python features include dynamic typing, garbage collection, multiple paradigms (OOP, functional, procedural), and a large standard library.",
        "python popularity": "Python is ranked #1 in TIOBE index as of 2024. Used by 48% of developers (Stack Overflow 2023).",
        "python applications": "Python is used in web development (Django, Flask), data science (NumPy, Pandas), AI/ML (TensorFlow, PyTorch), automation, and scripting.",
        "python performance": "Python is slower than C/C++ and Java due to being interpreted. PyPy, Cython, and Numba can improve performance significantly.",
        "langchain": "LangChain is a framework for building LLM-powered applications, released in 2022 by Harrison Chase.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs using directed graphs.",
    }
    for key, result in knowledge.items():
        if key.lower() in query.lower():
            return result
    return f"General information about '{query}': This is a well-documented topic with many resources available online."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 2 * 3'."""
    try:
        # Safe eval for basic arithmetic only
        allowed = set("0123456789+-*/(). ")
        if all(c in allowed for c in expression):
            result = eval(expression)  # noqa: S307
            return f"{expression} = {result}"
        return "Error: Expression contains unsafe characters."
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def summarise_text(text: str, max_sentences: int = 3) -> str:
    """Summarise a block of text to the specified number of sentences."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    selected = sentences[:max_sentences]
    return ". ".join(selected) + "." if selected else text


@tool
def compare_items(item_a: str, item_b: str, criterion: str) -> str:
    """Compare two items on a given criterion and provide analysis."""
    return (
        f"Comparison of '{item_a}' vs '{item_b}' on '{criterion}':\n"
        f"  {item_a}: Generally considered strong in {criterion} based on community reports.\n"
        f"  {item_b}: Also performs well in {criterion}, with different trade-offs.\n"
        f"  Verdict: Both have merits; choice depends on specific requirements."
    )


# ── Plan-and-Execute state ────────────────────────────────────────────────────
class PlanExecute(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add]
    response: str


# ── Planner ───────────────────────────────────────────────────────────────────
class Plan(BaseModel):
    """Ordered list of steps to achieve the goal."""
    steps: list[str] = Field(
        description="Different steps to follow, in order. Each step should be a single, concrete action.",
        min_length=2,
        max_length=6,
    )


planner_prompt = SystemMessage(content="""You are an expert task planner.
Break the user's goal into 2-5 concrete, sequential steps.
Each step should be actionable and specific.
Use tools like: search_web, calculate, summarise_text, compare_items.
Avoid vague steps like "think about..." — be specific about what to do.""")

planner = (
    lambda input_dict: [planner_prompt, HumanMessage(content=f"Goal: {input_dict['input']}\nCreate a plan with 2-4 concrete steps.")]
) 

planner_llm = llm.with_structured_output(Plan)


def plan_step(state: PlanExecute) -> dict:
    """Generate the initial execution plan."""
    messages = [
        planner_prompt,
        HumanMessage(content=f"Goal: {state['input']}\n\nCreate a plan with 2-4 concrete steps."),
    ]
    plan = planner_llm.invoke(messages)
    print(f"\n  [Planner] Generated {len(plan.steps)} steps:")
    for i, step in enumerate(plan.steps, 1):
        print(f"    {i}. {step}")
    return {"plan": plan.steps}


# ── Executor ─────────────────────────────────────────────────────────────────
executor_agent = create_react_agent(
    llm,
    tools=[search_web, calculate, summarise_text, compare_items],
    state_modifier=SystemMessage(content=
        "You are a task executor. Execute exactly the given task step using the available tools. "
        "Be concise — return only the result of executing the step."
    ),
)


def execute_step(state: PlanExecute) -> dict:
    """Execute the first remaining step in the plan."""
    plan = state["plan"]
    past_steps = state["past_steps"]

    current_step = plan[0]
    print(f"\n  [Executor] Executing: '{current_step}'")

    result = executor_agent.invoke({
        "messages": [HumanMessage(content=current_step)]
    })

    output = result["messages"][-1].content
    print(f"  [Executor] Result: {output[:100]}{'...' if len(output) > 100 else ''}")

    return {"past_steps": [(current_step, output)]}


# ── Re-planner ────────────────────────────────────────────────────────────────
class Response(BaseModel):
    """Final response to the user."""
    response: str


class ReplanDecision(BaseModel):
    """Either a revised plan or a final response."""
    action: str = Field(description="'replan' to continue with updated plan, 'finish' to return final answer")
    updated_steps: list[str] = Field(default=[], description="Remaining steps if action='replan'")
    final_answer: str = Field(default="", description="Complete answer if action='finish'")


replanner_llm = llm.with_structured_output(ReplanDecision)

REPLANNER_PROMPT = """You are a replanning agent. Review the original goal, the plan, and completed steps.
Decide whether to:
1. 'finish' — if the goal has been achieved, provide the complete final answer
2. 'replan' — if more steps are needed, provide the remaining updated steps

Original goal: {input}

Original plan:
{plan}

Completed steps and results:
{past_steps}

If 'finish': Synthesise all completed results into a comprehensive answer.
If 'replan': List only the remaining steps needed."""


def replan_step(state: PlanExecute) -> dict:
    """Evaluate progress and either finish or continue with updated plan."""
    plan_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan"]))
    past_str = "\n".join(f"Step: {s}\nResult: {r}" for s, r in state["past_steps"])

    messages = [HumanMessage(content=REPLANNER_PROMPT.format(
        input=state["input"],
        plan=plan_str,
        past_steps=past_str,
    ))]

    decision = replanner_llm.invoke(messages)

    if decision.action == "finish":
        print(f"\n  [Re-planner] Goal achieved → FINISH")
        return {"response": decision.final_answer}
    else:
        print(f"\n  [Re-planner] Continuing with {len(decision.updated_steps)} remaining steps")
        return {"plan": decision.updated_steps}


# ── Routing logic ─────────────────────────────────────────────────────────────
def should_end(state: PlanExecute) -> str:
    """After replanning: if we have a response, end. Otherwise continue executing."""
    if state.get("response"):
        return "finish"
    return "continue"


# ── Build the graph ────────────────────────────────────────────────────────────
def build_plan_execute_graph(checkpointer=None):
    graph = StateGraph(PlanExecute)

    graph.add_node("planner",  plan_step)
    graph.add_node("executor", execute_step)
    graph.add_node("replanner", replan_step)

    graph.set_entry_point("planner")
    graph.add_edge("planner",  "executor")
    graph.add_edge("executor", "replanner")

    graph.add_conditional_edges(
        "replanner",
        should_end,
        {
            "continue": "executor",
            "finish":   END,
        },
    )

    return graph.compile(checkpointer=checkpointer)


def demo_plan_execute():
    app = build_plan_execute_graph()

    goal = (
        "Research Python programming language: its history, key features, and main applications. "
        "Then provide a concise summary."
    )

    print("=== Plan-and-Execute Agent ===")
    print(f"Goal: {goal}")

    result = app.invoke({"input": goal, "plan": [], "past_steps": [], "response": ""})

    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print(result["response"])
    print("\nExecution trace:")
    for i, (step, output) in enumerate(result["past_steps"], 1):
        print(f"  {i}. {step}")
        print(f"     → {output[:80]}...")


def demo_plan_execute_with_checkpoints():
    """Same workflow but with checkpointing enabled."""
    memory = MemorySaver()
    app = build_plan_execute_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "plan-execute-001"}}

    goal = "Find out what LangGraph is and compare it to LangChain."

    print("\n=== Plan-and-Execute with Checkpointing ===")
    print(f"Goal: {goal}")

    result = app.invoke(
        {"input": goal, "plan": [], "past_steps": [], "response": ""},
        config=config,
    )

    # Show the checkpoint
    snapshot = app.get_state(config)
    print(f"\nCheckpoint saved:")
    print(f"  Thread   : {snapshot.config['configurable']['thread_id']}")
    print(f"  Steps done: {len(result['past_steps'])}")
    print(f"\nFINAL ANSWER:\n{result['response']}")


def demo_visualise_graph():
    """Print the graph structure."""
    app = build_plan_execute_graph()
    print("\n=== Plan-and-Execute Graph Structure ===")
    print(app.get_graph().draw_ascii())


if __name__ == "__main__":
    demo_plan_execute()
    demo_plan_execute_with_checkpoints()
    demo_visualise_graph()
