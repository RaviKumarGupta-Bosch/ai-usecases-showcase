"""
AutoGen Basics 01 — Intro to Agents
======================================
Topics covered:
  1. ConversableAgent — the base agent class
  2. AssistantAgent   — LLM-powered coding/chat assistant
  3. UserProxyAgent   — human-surrogate that can execute code
  4. Simple two-message exchange
  5. Disabling code execution (human_input_mode="NEVER")
  6. LLM config setup and model selection

Run:
  python 01_intro_agents.py
"""

import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

load_dotenv()

# ── LLM configuration ─────────────────────────────────────────────────────────
LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}


# ── 1. Hello World — minimal two-agent exchange ───────────────────────────────
def demo_hello_world():
    print("\n=== 1. Hello World — Two-Agent Exchange ===")

    assistant = AssistantAgent(
        name="Assistant",
        llm_config=LLM_CONFIG,
        system_message="You are a concise AI assistant. Keep answers to 2 sentences.",
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",      # no keyboard input required
        max_consecutive_auto_reply=1,  # stop after one reply
        code_execution_config=False,   # no code execution
    )

    user_proxy.initiate_chat(
        assistant,
        message="What is AutoGen in one sentence?",
    )


# ── 2. ConversableAgent — the base class ─────────────────────────────────────
def demo_conversable_agent():
    print("\n=== 2. ConversableAgent — Base Class ===")

    agent_a = ConversableAgent(
        name="AgentA",
        llm_config=LLM_CONFIG,
        system_message="You are Agent A. Always start your reply with 'Agent A here:'",
        human_input_mode="NEVER",
    )

    agent_b = ConversableAgent(
        name="AgentB",
        llm_config=LLM_CONFIG,
        system_message="You are Agent B. Always start your reply with 'Agent B here:'",
        human_input_mode="NEVER",
    )

    # Agents can initiate a chat with each other
    result = agent_a.initiate_chat(
        agent_b,
        message="What are the three main benefits of multi-agent AI systems?",
        max_turns=2,
    )
    print(f"\nChat cost summary: {result.cost}")


# ── 3. AssistantAgent system message customisation ───────────────────────────
def demo_custom_system_messages():
    print("\n=== 3. Custom System Messages ===")

    python_expert = AssistantAgent(
        name="PythonExpert",
        llm_config=LLM_CONFIG,
        system_message="""You are a Python expert.
- Always write idiomatic, PEP8-compliant code
- Include type annotations
- Keep functions small and focused
- Always add a one-line docstring""",
    )

    reviewer = AssistantAgent(
        name="CodeReviewer",
        llm_config=LLM_CONFIG,
        system_message="""You are a strict code reviewer.
- Point out any issues with the code you receive
- Be specific about improvements
- Rate code quality 1-10""",
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Ask the Python expert for code
    user.initiate_chat(
        python_expert,
        message="Write a function that finds the longest palindrome substring in a string.",
        max_turns=1,
    )


# ── 4. Agent metadata and termination conditions ──────────────────────────────
def demo_termination():
    print("\n=== 4. Termination Conditions ===")

    agent = ConversableAgent(
        name="CountdownAgent",
        llm_config=LLM_CONFIG,
        system_message="Count down from the given number, one step per reply. End with 'DONE!'",
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "DONE!" in msg.get("content", ""),
    )

    user = ConversableAgent(
        name="Starter",
        llm_config=False,
        human_input_mode="NEVER",
        default_auto_reply="Continue.",
    )

    user.initiate_chat(agent, message="Start countdown from 3.", max_turns=10)


# ── 5. UserProxyAgent with code execution ─────────────────────────────────────
def demo_code_execution():
    print("\n=== 5. UserProxyAgent Code Execution ===")

    assistant = AssistantAgent(
        name="Coder",
        llm_config=LLM_CONFIG,
        system_message="Write Python code to solve the task. Keep it simple and self-contained.",
    )

    executor = UserProxyAgent(
        name="Executor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config={
            "work_dir":       "autogen_workspace",
            "use_docker":     False,  # set True for sandboxed execution
            "last_n_messages": 3,
        },
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    )

    executor.initiate_chat(
        assistant,
        message="Write Python code that prints the Fibonacci sequence up to the 10th number. Reply TERMINATE when done.",
    )


# ── 6. LLM config with multiple models (fallback) ────────────────────────────
def demo_llm_config_list():
    print("\n=== 6. LLM Config List (multi-model fallback) ===")

    # AutoGen can try multiple models/configs in order
    config_list = [
        {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
        # {"model": "gpt-4o",     "api_key": os.getenv("OPENAI_API_KEY")},  # fallback
    ]

    llm_config_with_list = {
        "config_list": config_list,
        "temperature": 0,
        "cache_seed": 42,  # enable caching to save API calls during dev
    }

    agent = AssistantAgent(
        name="CachingAgent",
        llm_config=llm_config_with_list,
        system_message="You are helpful. Be brief.",
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # First call — hits API
    user.initiate_chat(agent, message="What is 2+2?", max_turns=1)
    print("First call complete.")

    # Second identical call — served from cache (same cache_seed)
    user.initiate_chat(agent, message="What is 2+2?", max_turns=1)
    print("Second call complete (from cache if same seed).")


if __name__ == "__main__":
    demo_hello_world()
    demo_conversable_agent()
    demo_custom_system_messages()
    demo_termination()
    demo_code_execution()
    demo_llm_config_list()
