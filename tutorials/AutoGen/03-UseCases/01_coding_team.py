"""
AutoGen Use Case 01 — Coding Team
=====================================
A full software development team with specialised agents:
  - Product Manager : clarifies requirements, writes user stories
  - Architect       : designs the solution architecture
  - Developer       : implements the code
  - Reviewer        : performs code review
  - Tester          : writes and conceptually executes tests

The agents collaborate in a structured group chat with ordered transitions.

Run:
  python 01_coding_team.py
"""

import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

load_dotenv()

LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.2,
}


def build_coding_team():
    """Create all agents for the coding team."""

    product_manager = AssistantAgent(
        name="ProductManager",
        llm_config=LLM_CONFIG,
        system_message="""You are a Product Manager.
Your job:
1. Clarify the requirement from the user
2. Write 2-3 clear acceptance criteria
3. Hand off to the Architect with: 'Architect, please design this.'
Keep it concise — one short paragraph max.""",
    )

    architect = AssistantAgent(
        name="Architect",
        llm_config=LLM_CONFIG,
        system_message="""You are a Software Architect.
Your job:
1. Design the high-level structure (classes, functions, data flow)
2. Specify the tech stack and key design decisions
3. Hand off to Developer with: 'Developer, please implement this.'
Be specific but brief — use bullet points.""",
    )

    developer = AssistantAgent(
        name="Developer",
        llm_config=LLM_CONFIG,
        system_message="""You are a Senior Python Developer.
Your job:
1. Implement clean, working Python code based on the architecture
2. Include type hints, docstrings, and error handling
3. Hand off to Reviewer with: 'Reviewer, please review this.'
Provide the complete, runnable code.""",
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        llm_config=LLM_CONFIG,
        system_message="""You are a Code Reviewer.
Your job:
1. Review the code for bugs, style, security, and performance
2. List issues found (if any) with severity: critical/high/medium/low
3. Either approve ('Code approved. Tester, please write tests.') 
   or request changes ('Developer, please fix: <specific issues>')
Be constructive and specific.""",
    )

    tester = AssistantAgent(
        name="Tester",
        llm_config=LLM_CONFIG,
        system_message="""You are a QA Engineer.
Your job:
1. Write comprehensive pytest test cases for the implemented code
2. Cover: happy path, edge cases, error conditions
3. End with: 'Testing complete. TASK DONE.'
Provide actual runnable test code.""",
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda msg: "TASK DONE" in msg.get("content", ""),
    )

    return product_manager, architect, developer, reviewer, tester, user_proxy


def run_coding_task(task: str):
    print("=" * 65)
    print("AutoGen Coding Team")
    print("=" * 65)
    print(f"\nTask: {task}\n")
    print("-" * 65)

    pm, arch, dev, rev, tester, user = build_coding_team()

    group_chat = GroupChat(
        agents=[user, pm, arch, dev, rev, tester],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
        is_termination_msg=lambda msg: "TASK DONE" in msg.get("content", ""),
    )

    user.initiate_chat(manager, message=task)

    print("\n" + "=" * 65)
    print(f"Team collaboration complete. Total rounds: {len(group_chat.messages)}")


if __name__ == "__main__":
    tasks = [
        (
            "Build a Python class `RateLimiter` that limits function calls to N times per second. "
            "It should be usable as a decorator and support both sync and async functions."
        ),
        (
            "Create a Python `EventBus` class implementing the publish-subscribe pattern. "
            "Subscribers should be able to register handlers for named events, "
            "and publishers should be able to emit events with payloads."
        ),
    ]

    # Run one task for the demo
    run_coding_task(tasks[0])
