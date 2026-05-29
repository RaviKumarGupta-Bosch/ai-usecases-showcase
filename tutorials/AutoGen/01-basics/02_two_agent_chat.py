"""
AutoGen Basics 02 — Two-Agent Chat Patterns
=============================================
Topics covered:
  1. Synchronous initiate_chat and result inspection
  2. Customising max_turns and max_consecutive_auto_reply
  3. Termination via is_termination_msg
  4. Chat history inspection
  5. Cost and token usage tracking
  6. Async chat (initiate_chat_async)

Run:
  python 02_two_agent_chat.py
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

load_dotenv()

LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}


# ── 1. Basic chat and result inspection ──────────────────────────────────────
def demo_chat_result():
    print("\n=== 1. Chat Result Inspection ===")

    assistant = AssistantAgent(
        name="Assistant",
        llm_config=LLM_CONFIG,
        system_message="Be concise. Answer in max 3 bullet points.",
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    result = user.initiate_chat(
        assistant,
        message="What are the top 3 features of Python that make it great for AI?",
    )

    print(f"\nChat summary      : {result.summary}")
    print(f"Total tokens used : {result.cost}")
    print(f"Number of messages: {len(result.chat_history)}")
    print("\nFull chat history:")
    for msg in result.chat_history:
        role = msg.get("role", "?")
        content = msg.get("content", "")[:100]
        print(f"  [{role}] {content}...")


# ── 2. Multi-turn conversation ────────────────────────────────────────────────
def demo_multi_turn():
    print("\n=== 2. Multi-Turn Conversation ===")

    tutor = AssistantAgent(
        name="Tutor",
        llm_config=LLM_CONFIG,
        system_message="""You are a patient Python tutor. 
Teach step-by-step. Ask the student a quiz question after each explanation.""",
    )

    student = ConversableAgent(
        name="Student",
        llm_config=LLM_CONFIG,
        system_message="""You are a Python student.
Answer the tutor's question, then ask one follow-up question.
After 3 exchanges, say 'LESSON COMPLETE' to end.""",
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "LESSON COMPLETE" in msg.get("content", ""),
    )

    student.initiate_chat(
        tutor,
        message="Can you teach me about Python list comprehensions?",
        max_turns=6,
    )


# ── 3. Structured termination with custom logic ───────────────────────────────
def demo_termination_patterns():
    print("\n=== 3. Termination Patterns ===")

    def is_done(message: dict) -> bool:
        content = message.get("content", "")
        return any(kw in content.upper() for kw in ["DONE", "COMPLETE", "FINISHED", "TERMINATE"])

    analyst = AssistantAgent(
        name="Analyst",
        llm_config=LLM_CONFIG,
        system_message="""You are a data analyst.
When you have provided all requested analysis, end your message with 'ANALYSIS COMPLETE'.""",
        is_termination_msg=is_done,
    )

    user = UserProxyAgent(
        name="Manager",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config=False,
        is_termination_msg=is_done,
    )

    user.initiate_chat(
        analyst,
        message="Analyse the pros and cons of using Python vs R for data science. Keep it brief.",
    )


# ── 4. Q&A bot with conversation history context ─────────────────────────────
def demo_conversation_context():
    print("\n=== 4. Conversation Context Carried Forward ===")

    expert = AssistantAgent(
        name="DatabaseExpert",
        llm_config=LLM_CONFIG,
        system_message="You are a database expert. Remember what was discussed earlier in the conversation.",
    )

    user = UserProxyAgent(
        name="Developer",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    questions = [
        "What is the difference between SQL and NoSQL databases?",
        "For the NoSQL types you mentioned, which is best for caching?",
        "And for a social network graph, which of those options would you choose?",
    ]

    # Manual sequential chat to demonstrate context retention
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        result = user.initiate_chat(
            expert,
            message=question,
            max_turns=1,
            clear_history=(i == 1),  # only clear on first question
        )
        print(f"A{i}: {result.summary[:200]}...")


# ── 5. Async two-agent chat ───────────────────────────────────────────────────
async def demo_async_chat():
    print("\n=== 5. Async Chat ===")

    assistant = AssistantAgent(
        name="AsyncAssistant",
        llm_config=LLM_CONFIG,
        system_message="You answer questions concisely.",
    )

    user = UserProxyAgent(
        name="AsyncUser",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Run two chats concurrently
    task1 = user.a_initiate_chat(
        assistant,
        message="What is MapReduce?",
        max_turns=1,
    )
    task2 = user.a_initiate_chat(
        assistant,
        message="What is the Actor model in distributed systems?",
        max_turns=1,
        clear_history=False,
    )

    results = await asyncio.gather(task1, task2)
    print(f"Chat 1 completed: {len(results[0].chat_history)} messages")
    print(f"Chat 2 completed: {len(results[1].chat_history)} messages")


# ── 6. Sending different message types ───────────────────────────────────────
def demo_message_types():
    print("\n=== 6. Message Types (text, dict, multimodal) ===")

    agent = AssistantAgent(
        name="FlexAgent",
        llm_config=LLM_CONFIG,
        system_message="You process various message formats.",
    )

    user = UserProxyAgent(
        name="Sender",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Plain string message
    user.initiate_chat(agent, message="Hello, explain what you can receive.", max_turns=1)

    # Dict message with role
    user.initiate_chat(
        agent,
        message={"role": "user", "content": "What is the last thing I asked?"},
        max_turns=1,
        clear_history=False,
    )


if __name__ == "__main__":
    demo_chat_result()
    demo_multi_turn()
    demo_termination_patterns()
    demo_conversation_context()
    asyncio.run(demo_async_chat())
    demo_message_types()
