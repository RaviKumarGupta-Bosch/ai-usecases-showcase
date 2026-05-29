"""
AutoGen Basics 03 — GroupChat
================================
Topics covered:
  1. Basic GroupChat with round-robin speaker selection
  2. GroupChatManager — orchestrates group conversations
  3. Custom speaker selection (selector_func)
  4. Constrained allowed_or_disallowed_speaker_transitions
  5. GroupChat with roles: planner, executor, reviewer
  6. Nested chat — agent calling another agent internally

Run:
  python 03_group_chat.py
"""

import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent

load_dotenv()

LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}


# ── 1. Basic GroupChat — round-robin ─────────────────────────────────────────
def demo_basic_group_chat():
    print("\n=== 1. Basic GroupChat (Round-Robin) ===")

    scientist = AssistantAgent(
        name="Scientist",
        llm_config=LLM_CONFIG,
        system_message="You are a scientist. Provide factual, evidence-based insights.",
    )

    philosopher = AssistantAgent(
        name="Philosopher",
        llm_config=LLM_CONFIG,
        system_message="You are a philosopher. Explore the deeper implications and ethics.",
    )

    engineer = AssistantAgent(
        name="Engineer",
        llm_config=LLM_CONFIG,
        system_message="You are an engineer. Focus on practical implementation and trade-offs.",
    )

    user_proxy = UserProxyAgent(
        name="Moderator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    group_chat = GroupChat(
        agents=[user_proxy, scientist, philosopher, engineer],
        messages=[],
        max_round=6,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
    )

    user_proxy.initiate_chat(
        manager,
        message="What are the most important considerations for deploying AI in healthcare?",
    )


# ── 2. Auto speaker selection (LLM-driven) ───────────────────────────────────
def demo_auto_speaker_selection():
    print("\n=== 2. Auto Speaker Selection (LLM-Driven) ===")

    planner = AssistantAgent(
        name="Planner",
        llm_config=LLM_CONFIG,
        system_message="""You break complex tasks into clear steps.
When you have created a plan, end with 'Plan ready. Coder, please implement step 1.'""",
    )

    coder = AssistantAgent(
        name="Coder",
        llm_config=LLM_CONFIG,
        system_message="""You write clean Python code based on the plan.
After writing code, say 'Code ready. Reviewer, please check.'""",
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        llm_config=LLM_CONFIG,
        system_message="""You review code for bugs, style, and correctness.
After review, say 'Review done.' and either approve or send back to Coder.""",
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    group_chat = GroupChat(
        agents=[user_proxy, planner, coder, reviewer],
        messages=[],
        max_round=8,
        speaker_selection_method="auto",
    )

    manager = GroupChatManager(groupchat=group_chat, llm_config=LLM_CONFIG)

    user_proxy.initiate_chat(
        manager,
        message="Create a Python function that validates email addresses using regex.",
    )


# ── 3. Constrained speaker transitions ───────────────────────────────────────
def demo_constrained_transitions():
    print("\n=== 3. Constrained Speaker Transitions ===")

    writer = AssistantAgent(
        name="Writer",
        llm_config=LLM_CONFIG,
        system_message="You write the first draft of content. Be creative but concise.",
    )

    editor = AssistantAgent(
        name="Editor",
        llm_config=LLM_CONFIG,
        system_message="You edit and improve the writer's draft. Focus on clarity and flow.",
    )

    fact_checker = AssistantAgent(
        name="FactChecker",
        llm_config=LLM_CONFIG,
        system_message="You verify facts in the edited content. Flag any inaccuracies.",
    )

    user_proxy = UserProxyAgent(
        name="Publisher",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    group_chat = GroupChat(
        agents=[user_proxy, writer, editor, fact_checker],
        messages=[],
        max_round=6,
        # Enforce: Publisher → Writer → Editor → FactChecker → Publisher
        allowed_or_disallowed_speaker_transitions={
            user_proxy:   [writer],
            writer:       [editor],
            editor:       [fact_checker],
            fact_checker: [user_proxy],
        },
        speaker_transitions_type="allowed",
    )

    manager = GroupChatManager(groupchat=group_chat, llm_config=LLM_CONFIG)

    user_proxy.initiate_chat(
        manager,
        message="Write a short paragraph about the history of the internet.",
    )


# ── 4. Custom speaker selection function ─────────────────────────────────────
def demo_custom_selector():
    print("\n=== 4. Custom Speaker Selector Function ===")

    agents = {}

    for role in ["Analyst", "Developer", "Tester"]:
        agents[role] = AssistantAgent(
            name=role,
            llm_config=LLM_CONFIG,
            system_message=f"You are a {role}. Contribute from your perspective.",
        )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    pipeline = [user_proxy, agents["Analyst"], agents["Developer"], agents["Tester"]]

    def sequential_selector(last_speaker, group_chat_obj):
        """Always follow the pipeline order."""
        idx = pipeline.index(last_speaker)
        return pipeline[(idx + 1) % len(pipeline)]

    group_chat = GroupChat(
        agents=pipeline,
        messages=[],
        max_round=4,
        speaker_selection_method=sequential_selector,
    )

    manager = GroupChatManager(groupchat=group_chat, llm_config=LLM_CONFIG)

    user_proxy.initiate_chat(
        manager,
        message="We need to build a REST API for user management. Each role, give your key concern.",
    )


# ── 5. Broadcast message to all agents ───────────────────────────────────────
def demo_broadcast():
    print("\n=== 5. Group Chat Broadcast Pattern ===")

    agents_list = [
        AssistantAgent(
            name=f"Expert_{domain}",
            llm_config=LLM_CONFIG,
            system_message=f"You are a {domain} expert. Answer from your domain perspective only.",
        )
        for domain in ["Security", "Performance", "Scalability"]
    ]

    user_proxy = UserProxyAgent(
        name="Architect",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=0,
    )

    group_chat = GroupChat(
        agents=[user_proxy] + agents_list,
        messages=[],
        max_round=4,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(groupchat=group_chat, llm_config=LLM_CONFIG)

    user_proxy.initiate_chat(
        manager,
        message="We're migrating a monolith to microservices. What is your primary concern?",
    )


if __name__ == "__main__":
    demo_basic_group_chat()
    demo_auto_speaker_selection()
    demo_constrained_transitions()
    demo_custom_selector()
    demo_broadcast()
