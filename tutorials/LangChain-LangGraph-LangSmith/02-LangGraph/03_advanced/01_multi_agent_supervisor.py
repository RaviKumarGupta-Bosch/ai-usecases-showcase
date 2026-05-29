"""
01 - Multi-Agent Supervisor (LangGraph Advanced)
=================================================
A supervisor agent orchestrates multiple specialist worker agents.
The supervisor decides which worker to call next, or finishes.

Architecture:
  User → Supervisor → [Researcher | Writer | Critic] → Supervisor → User

Topics covered:
  1. Supervisor agent with structured routing output
  2. Worker agents built with create_react_agent
  3. Shared AgentState with messages + next field
  4. Route supervisor → worker → supervisor → FINISH
  5. Streaming multi-agent execution
  6. Adding a critic agent for quality review
"""

import operator
from typing import TypedDict, Annotated, Literal, Sequence
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ── Worker tools ──────────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Simulate a web search. Returns relevant facts about the query topic."""
    # Simulated search results (in production, use TavilySearchResults)
    results = {
        "langchain": (
            "LangChain is an open-source framework for building LLM applications. "
            "Released in 2022 by Harrison Chase, it provides abstractions for chains, "
            "agents, and memory. Used by thousands of companies worldwide."
        ),
        "langgraph": (
            "LangGraph is a LangChain extension for building stateful, multi-actor "
            "workflows as directed graphs. Supports cycles, persistence, and HITL. "
            "Released in 2024, it enables complex agentic workflows."
        ),
        "langsmith": (
            "LangSmith is a platform for debugging, testing, and monitoring LLM apps. "
            "Provides tracing, evaluation, and dataset management. "
            "Deeply integrated with LangChain and LangGraph."
        ),
    }
    for key, result in results.items():
        if key.lower() in query.lower():
            return result
    return (
        f"Search results for '{query}': General AI frameworks help developers build "
        "LLM-powered applications with features like tool use, memory, and workflow management."
    )


@tool
def get_word_count(text: str) -> str:
    """Count the words in the given text."""
    count = len(text.split())
    return f"Word count: {count}"


@tool
def format_as_markdown(text: str, title: str) -> str:
    """Format content as a well-structured markdown article."""
    return f"# {title}\n\n{text}\n\n---\n*Formatted by the Writer agent*"


@tool
def check_facts(claim: str) -> str:
    """Verify a factual claim (simulated)."""
    known_facts = {
        "langchain": "correct",
        "langgraph": "correct",
        "2022": "LangChain was released in late 2022 — correct",
        "2024": "LangGraph was released in 2024 — correct",
        "harrison chase": "Harrison Chase founded LangChain — correct",
    }
    for key, status in known_facts.items():
        if key.lower() in claim.lower():
            return f"Fact check: {status}"
    return f"Fact check for '{claim}': No contradicting information found"


# ── Worker agents ─────────────────────────────────────────────────────────────
researcher_agent = create_react_agent(
    llm,
    tools=[web_search],
    state_modifier=SystemMessage(content=
        "You are a Research specialist. Your job is to gather accurate, "
        "factual information using the web_search tool. "
        "Be thorough and cite your sources. Return structured findings."
    ),
)

writer_agent = create_react_agent(
    llm_creative,
    tools=[format_as_markdown, get_word_count],
    state_modifier=SystemMessage(content=
        "You are a professional Technical Writer. Given research findings, "
        "write clear, engaging, well-structured content. "
        "Use format_as_markdown for the final output. "
        "Aim for 100-150 words."
    ),
)

critic_agent = create_react_agent(
    llm,
    tools=[check_facts],
    state_modifier=SystemMessage(content=
        "You are a Quality Critic. Review written content for: "
        "accuracy (use check_facts), clarity, structure, and completeness. "
        "Provide specific, actionable feedback. "
        "Give an overall score 1-10 and say APPROVED if score >= 7."
    ),
)


# ── Supervisor state & routing ────────────────────────────────────────────────
WORKERS = ["researcher", "writer", "critic"]

class RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
        description="The next worker to call, or FINISH if the task is complete."
    )
    reasoning: str = Field(description="Brief reason for this routing decision")


supervisor_llm = llm.with_structured_output(RouteDecision)

SUPERVISOR_PROMPT = f"""You are a supervisor managing a team of workers: {', '.join(WORKERS)}.

Given the conversation, decide the next worker to call:
- researcher: gathers factual information and data
- writer: creates and formats written content  
- critic: reviews content for quality and accuracy
- FINISH: all tasks are complete and we have high-quality output

A typical flow: researcher → writer → critic → FINISH
If the critic approves (score >= 7), finish. Otherwise, send back to writer for revision."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str


def supervisor_node(state: AgentState) -> dict:
    """Supervisor reads all messages and decides which worker to activate next."""
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    decision = supervisor_llm.invoke(messages)
    return {"next": decision.next}


def make_worker_node(agent, name: str):
    """Wrap a worker agent as a graph node."""
    def worker_node(state: AgentState) -> dict:
        result = agent.invoke(state)
        # Extract the last AI message and tag it with the worker's name
        last_msg = result["messages"][-1]
        tagged = HumanMessage(content=f"[{name.upper()}]: {last_msg.content}", name=name)
        return {"messages": [tagged]}
    worker_node.__name__ = name
    return worker_node


# ── Build the supervisor graph ─────────────────────────────────────────────────
def build_supervisor_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", make_worker_node(researcher_agent, "researcher"))
    graph.add_node("writer",     make_worker_node(writer_agent,     "writer"))
    graph.add_node("critic",     make_worker_node(critic_agent,     "critic"))

    graph.set_entry_point("supervisor")

    # Supervisor routes to a worker or FINISH
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "researcher": "researcher",
            "writer":     "writer",
            "critic":     "critic",
            "FINISH":     END,
        },
    )

    # All workers return to supervisor
    for worker in ["researcher", "writer", "critic"]:
        graph.add_edge(worker, "supervisor")

    return graph.compile()


def demo_supervisor_pipeline():
    app = build_supervisor_graph()

    task = HumanMessage(content=
        "Research LangGraph, write a concise 100-word technical summary about it, "
        "and have it reviewed for quality."
    )

    print("=== Multi-Agent Supervisor Pipeline ===")
    print(f"Task: {task.content}\n")
    print("Execution trace:")

    step_count = 0
    for event in app.stream({"messages": [task], "next": ""}):
        node_name = next(iter(event.keys()))
        step_count += 1

        if node_name == "supervisor":
            next_node = event["supervisor"].get("next", "?")
            print(f"  [Step {step_count}] Supervisor → next: {next_node}")
        else:
            messages = event[node_name].get("messages", [])
            if messages:
                content = messages[-1].content
                print(f"  [Step {step_count}] {node_name.title()} output: {content[:100]}...")

    print(f"\nTotal steps: {step_count}")


def demo_streaming_supervisor():
    """Show clean streaming output."""
    app = build_supervisor_graph()

    task = HumanMessage(content=
        "Research LangChain and write a brief introduction paragraph for a tutorial."
    )

    print("\n=== Streaming Supervisor Output ===")
    final_state = app.invoke({"messages": [task], "next": ""})

    print("Final messages in state:")
    for msg in final_state["messages"]:
        if hasattr(msg, "name") and msg.name:
            print(f"\n[{msg.name.upper()}]:\n{msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"\n[USER]: {msg.content[:80]}")


if __name__ == "__main__":
    demo_supervisor_pipeline()
    demo_streaming_supervisor()
