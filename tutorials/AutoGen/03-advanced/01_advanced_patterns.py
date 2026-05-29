"""
AutoGen 03-Advanced — Advanced Patterns
=========================================
Topics covered:
  1. Custom termination conditions
  2. Nested chat patterns
  3. Group chat with custom speaker selection
  4. Sequential chats with carryover (initiate_chats)
  5. Token budget and cost controls
  6. Async agent conversations (a_initiate_chat)
  7. Practical: multi-stage research pipeline

Prerequisites:
  pip install pyautogen openai python-dotenv

Run:
  python 01_advanced_patterns.py
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    initiate_chats,
)

load_dotenv()

LLM_CONFIG = {
    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}


# ── 1. Custom termination conditions ─────────────────────────────────────────
def demo_termination():
    print("\n=== 1. Custom Termination Conditions ===")

    def is_done(msg: dict) -> bool:
        """Terminate when the assistant includes 'DONE' in its message."""
        content = msg.get("content", "") or ""
        return "DONE" in content.upper()

    assistant = AssistantAgent(
        name="TaskAgent",
        llm_config=LLM_CONFIG,
        system_message=(
            "Solve tasks step by step. "
            "When fully complete, include 'DONE' in your final message."
        ),
        is_termination_msg=is_done,
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )

    print("  Termination via is_termination_msg=<callable>")
    print("  → stops as soon as callable returns True for any message")
    print("  Other patterns:")
    print("    max_consecutive_auto_reply=N  — hard turn cap")
    print("    is_termination_msg=lambda m: 'TERMINATE' in (m.get('content') or '')")
    print(f"  Agent: {assistant.name}, User: {user.name}")


# ── 2. Nested chat ────────────────────────────────────────────────────────────
def demo_nested_chat():
    print("\n=== 2. Nested Chat Patterns ===")

    # An outer conversation can trigger an inner conversation.
    # The inner chat result is injected back as a message in the outer chat.

    inner_reviewer = AssistantAgent(
        name="Reviewer",
        llm_config=LLM_CONFIG,
        system_message="You are a code reviewer. Return concise feedback.",
    )
    inner_proxy = UserProxyAgent(
        name="InnerProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        is_termination_msg=lambda m: True,
    )

    outer_coder = AssistantAgent(
        name="Coder",
        llm_config=LLM_CONFIG,
        system_message="Write Python code, then ask for a review.",
    )
    outer_user = UserProxyAgent(
        name="OuterUser",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config=False,
    )

    # Register the nested chat: whenever outer_user gets a message from Coder,
    # it automatically triggers an inner chat with the Reviewer.
    outer_user.register_nested_chats(
        [
            {
                "recipient": inner_reviewer,
                "sender": inner_proxy,
                "message": lambda recipient, messages, sender, config: messages[-1]["content"],
                "max_turns": 1,
                "summary_method": "last_msg",
            }
        ],
        trigger=outer_coder,
    )

    print(f"  Outer: {outer_coder.name} → {outer_user.name}")
    print(f"  Inner (triggered on each Coder message): {inner_reviewer.name}")
    print("  Result flows back into the outer conversation automatically.")


# ── 3. Group chat with custom speaker selection ───────────────────────────────
def demo_group_chat():
    print("\n=== 3. Group Chat with Custom Speaker Selection ===")

    planner = AssistantAgent(
        name="Planner",
        llm_config=LLM_CONFIG,
        system_message="Break tasks into numbered sub-tasks.",
    )
    coder = AssistantAgent(
        name="Coder",
        llm_config=LLM_CONFIG,
        system_message="Implement each sub-task in Python.",
    )
    critic = AssistantAgent(
        name="Critic",
        llm_config=LLM_CONFIG,
        system_message="Review plans and code; suggest improvements.",
    )
    manager_proxy = UserProxyAgent(
        name="Manager",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    ordered_speakers = [planner, coder, critic]

    def round_robin(last_speaker, groupchat: GroupChat):
        """Cycle Planner → Coder → Critic deterministically."""
        if last_speaker in ordered_speakers:
            idx = ordered_speakers.index(last_speaker)
            return ordered_speakers[(idx + 1) % len(ordered_speakers)]
        return planner

    gc = GroupChat(
        agents=[manager_proxy, planner, coder, critic],
        messages=[],
        max_round=9,
        speaker_selection_method=round_robin,
    )
    gc_manager = GroupChatManager(groupchat=gc, llm_config=LLM_CONFIG)

    print(f"  Agents: {[a.name for a in ordered_speakers]}")
    print("  Turn order: Planner → Coder → Critic (deterministic round-robin)")
    print("  Other selection methods: 'auto' (LLM chooses), 'random', <callable>")
    print(f"  GroupChatManager: {gc_manager.name}")


# ── 4. Sequential chats with carryover ───────────────────────────────────────
def demo_sequential_chats():
    print("\n=== 4. Sequential Chats with Carryover ===")

    researcher = AssistantAgent(
        name="Researcher",
        llm_config=LLM_CONFIG,
        system_message="Research topics and write detailed summaries.",
    )
    writer = AssistantAgent(
        name="Writer",
        llm_config=LLM_CONFIG,
        system_message="Write polished articles from provided research summaries.",
    )
    editor = AssistantAgent(
        name="Editor",
        llm_config=LLM_CONFIG,
        system_message="Edit articles for clarity, grammar, and flow.",
    )
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        is_termination_msg=lambda m: True,
    )

    # initiate_chats: each chat's summary is automatically prepended to the next
    chat_sequence = [
        {
            "recipient": researcher,
            "message": "Research the top 5 benefits of async programming in Python.",
            "max_turns": 2,
            "summary_method": "last_msg",      # carry last message to next chat
        },
        {
            "recipient": writer,
            "message": "Write a 3-paragraph blog post from the research.",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": editor,
            "message": "Edit the article for publication.",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
    ]

    print(f"  Pipeline: {' → '.join(c['recipient'].name for c in chat_sequence)}")
    print("  Each stage receives the summary_method output of the previous stage.")
    print("  Execution: results = initiate_chats(user, chat_sequence)")
    print("  results[i].summary  → summary string for stage i")
    print("  results[i].chat_history  → full message list")


# ── 5. Token budget and cost controls ────────────────────────────────────────
def demo_cost_control():
    print("\n=== 5. Token Budget and Cost Controls ===")

    # Add per-request price to LLM config for cost tracking
    cost_aware_config = {
        **LLM_CONFIG,
        "price": [0.00015, 0.0006],     # [$/1k input tokens, $/1k output tokens]
        "cache_seed": 42,               # cache identical prompts (dev only)
    }

    agent = AssistantAgent(
        name="CostAwareAgent",
        llm_config=cost_aware_config,
        system_message="Be concise — max 30 words per response.",
    )
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,   # hard cap — prevents runaway loops
        code_execution_config=False,
    )

    print("  Cost config: price=[input_per_1k, output_per_1k]")
    print("  After run: agent.client.total_usage_summary")
    print("    → {'total_cost': 0.0012, 'gpt-4o-mini': {'prompt_tokens': 800, ...")
    print()
    print("  Hard caps:")
    print(f"    max_consecutive_auto_reply = {user.max_consecutive_auto_reply}")
    print("    Combine with is_termination_msg for early exit")
    print()
    print("  cache_seed=42 → identical prompts reuse cached responses in dev")


# ── 6. Async conversations ────────────────────────────────────────────────────
def demo_async():
    print("\n=== 6. Async Agent Conversations ===")

    async def single_async_chat():
        assistant = AssistantAgent(
            name="AsyncAssistant",
            llm_config=LLM_CONFIG,
            system_message="Answer in one sentence.",
        )
        user = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
            is_termination_msg=lambda m: True,
        )
        # Use a_initiate_chat for async execution
        print("  single call: await user.a_initiate_chat(assistant, message='...')")
        return "async_result_placeholder"

    async def concurrent_chats():
        """Run multiple independent chats concurrently."""
        # Conceptual example — avoids real API calls in demo
        prompts = [
            "Summarise async/await in Python.",
            "What is the GIL?",
            "Explain asyncio.gather.",
        ]
        print(f"\n  {len(prompts)} prompts run concurrently via asyncio.gather:")
        for i, p in enumerate(prompts, 1):
            print(f"    [{i}] {p!r}")
        print("  tasks = [user.a_initiate_chat(agent, message=p) for p in prompts]")
        print("  results = await asyncio.gather(*tasks)")

    asyncio.run(single_async_chat())
    asyncio.run(concurrent_chats())


# ── 7. Practical: multi-stage research pipeline ───────────────────────────────
def demo_research_pipeline():
    print("\n=== 7. Practical: Multi-Stage Research Pipeline ===")

    # 3-agent pipeline: Analyst → Researcher → Report Writer
    analyst = AssistantAgent(
        name="Analyst",
        llm_config=LLM_CONFIG,
        system_message="Identify 5 key research questions about the given topic.",
    )
    researcher = AssistantAgent(
        name="Researcher",
        llm_config=LLM_CONFIG,
        system_message="Answer each question with evidence-based facts.",
    )
    writer = AssistantAgent(
        name="ReportWriter",
        llm_config=LLM_CONFIG,
        system_message="Write a structured markdown report from the findings.",
    )
    user = UserProxyAgent(
        name="PM",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
        is_termination_msg=lambda m: True,
    )

    pipeline = [
        {
            "recipient": analyst,
            "message": "Topic: {topic}",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": researcher,
            "message": "Research the identified questions thoroughly.",
            "max_turns": 3,
            "summary_method": "last_msg",
        },
        {
            "recipient": writer,
            "message": "Write the final report using the research.",
            "max_turns": 2,
            "summary_method": "reflection_with_llm",
        },
    ]

    topic = "LLM fine-tuning strategies for domain-specific applications"
    pipeline[0]["message"] = pipeline[0]["message"].format(topic=topic)

    print(f"  Topic: {topic!r}")
    print(f"  Pipeline: {' → '.join(c['recipient'].name for c in pipeline)}")
    print("  Run: results = initiate_chats(user, pipeline)")
    print("  Final report: results[-1].summary")
    print()
    print("  Tips:")
    print("    summary_method='reflection_with_llm' → LLM-generated concise summary")
    print("    summary_method='last_msg'           → raw last message (faster)")
    print("    summary_prompt='...'                → custom summarisation instruction")


if __name__ == "__main__":
    print("AutoGen 03-Advanced — Advanced Patterns")
    print("=" * 44)
    demo_termination()
    demo_nested_chat()
    demo_group_chat()
    demo_sequential_chats()
    demo_cost_control()
    demo_async()
    demo_research_pipeline()
