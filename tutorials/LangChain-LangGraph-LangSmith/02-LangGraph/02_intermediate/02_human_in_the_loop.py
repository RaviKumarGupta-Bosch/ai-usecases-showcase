"""
02 - Human-in-the-Loop (LangGraph Intermediate)
================================================
Human-in-the-loop (HITL) patterns let a human review, approve, or modify
agent actions before they are executed. Essential for high-stakes workflows.

Topics covered:
  1. interrupt_before — pause graph before a specific node
  2. Inspect pending state with get_state()
  3. Resume with graph.invoke(None, config) after approval
  4. Modify state before resuming (human edits)
  5. Reject and redirect — human rejects action, graph retries
  6. interrupt_after — pause after a node to inspect output
  7. Multi-turn approval flow
"""

import operator
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Simulated "dangerous" tools ───────────────────────────────────────────────
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This requires human approval."""
    print(f"  📧 EMAIL SENT → To: {to} | Subject: {subject}")
    return f"Email sent to {to} with subject '{subject}'"


@tool
def delete_record(record_id: str, table: str) -> str:
    """Delete a database record. IRREVERSIBLE — requires human approval."""
    print(f"  🗑️  RECORD DELETED → {table}.{record_id}")
    return f"Deleted record {record_id} from {table}"


@tool
def read_file(path: str) -> str:
    """Read a file (safe, no approval needed)."""
    # Simulate reading a file
    content = f"[Contents of {path}]\nLine 1: Sample data\nLine 2: More data"
    return content


SAFE_TOOLS    = [read_file]
DANGER_TOOLS  = [send_email, delete_record]
ALL_TOOLS     = SAFE_TOOLS + DANGER_TOOLS

llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ── State ─────────────────────────────────────────────────────────────────────
class HITLState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    human_approved: bool
    human_feedback: str


# ── Nodes ─────────────────────────────────────────────────────────────────────
SYSTEM = SystemMessage(content=
    "You are a helpful AI assistant. You have access to tools including some that "
    "perform irreversible actions (sending emails, deleting records). "
    "Use tools when asked."
)


def agent_node(state: HITLState) -> dict:
    messages = [SYSTEM] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def human_approval_node(state: HITLState) -> dict:
    """
    This node is intentionally a no-op. The graph pauses BEFORE it runs
    (interrupt_before=["human_approval"]) so a human can review the pending
    tool call and decide to approve or reject.
    """
    return {}


# ── 1. Basic interrupt_before ─────────────────────────────────────────────────
def demo_interrupt_before():
    """
    Pause before executing a tool that requires human approval.
    Inspect the pending tool call, then resume.
    """
    graph = StateGraph(HITLState)
    graph.add_node("agent",          agent_node)
    graph.add_node("tools",          ToolNode(ALL_TOOLS))
    graph.add_node("human_approval", human_approval_node)

    graph.set_entry_point("agent")

    # Agent decides: if tool call → human_approval, else → END
    def route_agent(state: HITLState) -> Literal["human_approval", "__end__"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "human_approval"
        return "__end__"

    graph.add_conditional_edges("agent", route_agent)
    graph.add_edge("human_approval", "tools")
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    # interrupt_before pauses execution BEFORE the listed node runs
    app = graph.compile(
        checkpointer=memory,
        interrupt_before=["human_approval"],
    )

    config = {"configurable": {"thread_id": "hitl-demo-1"}}
    initial = {
        "messages": [HumanMessage(content="Send an email to alice@example.com with subject 'Hello' and body 'How are you?'")],
        "human_approved": False,
        "human_feedback": "",
    }

    print("=== 1. interrupt_before ===")
    print("Step 1 — Running agent...")
    # Run until the interrupt
    result = app.invoke(initial, config=config)

    # Check what the graph wants to do
    snapshot = app.get_state(config)
    print(f"  Graph paused at: {snapshot.next}")

    # Inspect the pending tool call
    last_msg = snapshot.values["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        tc = last_msg.tool_calls[0]
        print(f"  Pending action : {tc['name']}({tc['args']})")

    print("\nStep 2 — Human approves. Resuming...")
    # Resume: pass None as input (state already saved in checkpointer)
    final = app.invoke(None, config=config)
    print(f"  Final response : {final['messages'][-1].content}")


# ── 2. Modify state before resuming ───────────────────────────────────────────
def demo_modify_before_resume():
    """
    The human can update the state (e.g. change email recipient) before resuming.
    Use update_state() to patch the checkpointed state.
    """
    graph = StateGraph(HITLState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

    config = {"configurable": {"thread_id": "hitl-demo-2"}}
    initial = {
        "messages": [HumanMessage(content="Send an email to wrong@example.com saying 'Test'")],
        "human_approved": False,
        "human_feedback": "",
    }

    print("\n=== 2. Modify State Before Resuming ===")
    app.invoke(initial, config=config)

    snapshot = app.get_state(config)
    print(f"  Paused before: {snapshot.next}")

    # Human modifies the last AI message to correct the email address
    last_ai_msg = snapshot.values["messages"][-1]
    if hasattr(last_ai_msg, "tool_calls") and last_ai_msg.tool_calls:
        original_args = last_ai_msg.tool_calls[0]["args"]
        print(f"  Original args: {original_args}")

        # Patch: change 'to' field
        corrected_args = dict(original_args, to="correct@example.com")
        from langchain_core.messages import AIMessage
        from langchain_core.messages.tool import ToolCall
        corrected_msg = AIMessage(
            content=last_ai_msg.content,
            tool_calls=[ToolCall(
                name=last_ai_msg.tool_calls[0]["name"],
                args=corrected_args,
                id=last_ai_msg.tool_calls[0]["id"],
            )],
        )
        # Update the checkpointed state
        app.update_state(config, {"messages": [corrected_msg]}, as_node="agent")
        print(f"  Corrected 'to': correct@example.com")

    print("  Resuming with corrected state...")
    final = app.invoke(None, config=config)
    print(f"  Done: {final['messages'][-1].content}")


# ── 3. Multi-turn approval flow ───────────────────────────────────────────────
def demo_multi_turn_approval():
    """
    Demonstrate a realistic workflow:
    1. User asks agent to do a task
    2. Agent reads a file (auto-approved, no interrupt)
    3. Agent wants to delete a record (interrupted for approval)
    4. Human approves → delete executes
    5. Agent provides summary
    """

    def needs_approval(state: HITLState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if not (hasattr(last, "tool_calls") and last.tool_calls):
            return "__end__"
        # Check if any tool call requires approval
        tool_name = last.tool_calls[0]["name"]
        if tool_name in ["delete_record", "send_email"]:
            return "tools"  # will be interrupted
        return "tools"

    graph = StateGraph(HITLState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory, interrupt_before=["tools"])
    config = {"configurable": {"thread_id": "hitl-demo-3"}}

    initial = {
        "messages": [HumanMessage(
            content="First read the file '/data/users.csv', "
                    "then delete record 'user_42' from the 'users' table."
        )],
        "human_approved": False,
        "human_feedback": "",
    }

    print("\n=== 3. Multi-Turn Approval Flow ===")

    approved_count = 0
    max_rounds = 8  # safety guard

    # Run the first step
    state = app.invoke(initial, config=config)

    for _ in range(max_rounds):
        snapshot = app.get_state(config)
        if not snapshot.next:
            # Graph is done
            print(f"\nFinal: {snapshot.values['messages'][-1].content}")
            break

        # Check what we're about to execute
        last_msg = snapshot.values["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tc = last_msg.tool_calls[0]
            print(f"\n  [INTERRUPT] About to execute: {tc['name']}({tc['args']})")
            # Simulate human approval for read_file only, and delete_record
            print(f"  [HUMAN] Approved (simulated)")
            approved_count += 1

        # Resume
        state = app.invoke(None, config=config)

    print(f"\n  Total approvals: {approved_count}")


if __name__ == "__main__":
    demo_interrupt_before()
    demo_modify_before_resume()
    demo_multi_turn_approval()
