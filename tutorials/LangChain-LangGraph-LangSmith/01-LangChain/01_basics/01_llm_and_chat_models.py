"""
01 - LLMs and Chat Models
=========================
The entry point to LangChain: calling language models.

Topics covered:
  1. ChatOpenAI: basic invoke
  2. System / Human / AI message types
  3. Model parameters (temperature, max_tokens)
  4. Streaming token by token
  5. Batch parallel calls
  6. Async invocation
  7. Token usage tracking
"""

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()  # reads OPENAI_API_KEY (and others) from .env


# ── 1. Basic invoke ──────────────────────────────────────────────────────────
def demo_basic_invoke():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("What is LangChain in one sentence?")

    print("=== 1. Basic Invoke ===")
    print("Answer:", response.content)
    print("Model :", response.response_metadata.get("model_name"))
    print("Tokens:", response.usage_metadata)


# ── 2. Typed messages ────────────────────────────────────────────────────────
def demo_typed_messages():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content="You are a concise Python tutor. Keep answers under 3 sentences."),
        HumanMessage(content="What is a list comprehension?"),
    ]
    response = llm.invoke(messages)

    print("\n=== 2. Typed Messages ===")
    print(response.content)


# ── 3. Multi-turn conversation ───────────────────────────────────────────────
def demo_multi_turn():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    history = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="My name is Alice."),
        AIMessage(content="Nice to meet you, Alice! How can I help you today?"),
        HumanMessage(content="What did I just tell you my name was?"),
    ]
    response = llm.invoke(history)

    print("\n=== 3. Multi-Turn Conversation ===")
    print(response.content)


# ── 4. Streaming ─────────────────────────────────────────────────────────────
def demo_streaming():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("\n=== 4. Streaming ===")
    print("Response: ", end="", flush=True)
    for chunk in llm.stream("List 3 Python best practices, one per line."):
        print(chunk.content, end="", flush=True)
    print()  # newline after stream


# ── 5. Batch calls (parallel) ────────────────────────────────────────────────
def demo_batch():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    questions = [
        "Capital of France?",
        "Capital of Germany?",
        "Capital of Japan?",
    ]
    # Batch sends all requests concurrently
    responses = llm.batch(questions)

    print("\n=== 5. Batch Calls ===")
    for q, r in zip(questions, responses):
        print(f"  Q: {q:<22} → A: {r.content}")


# ── 6. Async invocation ──────────────────────────────────────────────────────
async def _async_call():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = await llm.ainvoke("Explain async/await in Python in two sentences.")
    return response.content


def demo_async():
    result = asyncio.run(_async_call())
    print("\n=== 6. Async Invocation ===")
    print(result)


# ── 7. Model parameters ──────────────────────────────────────────────────────
def demo_model_params():
    # High temperature → creative / varied
    creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.1, max_tokens=80)
    haiku = creative_llm.invoke("Write a haiku about distributed systems.")

    # temperature=0 → deterministic
    precise_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    answer = precise_llm.invoke("What is 123 * 456?")

    print("\n=== 7. Model Parameters ===")
    print(f"Creative (haiku) : {haiku.content}")
    print(f"Precise (math)   : {answer.content}")


# ── 8. OpenAI models comparison ──────────────────────────────────────────────
def demo_model_selection():
    prompt = "Explain gradient descent in one sentence."

    results = {}
    for model in ["gpt-4o-mini", "gpt-4o"]:
        llm = ChatOpenAI(model=model, temperature=0)
        r = llm.invoke(prompt)
        results[model] = {
            "answer": r.content,
            "input_tokens": r.usage_metadata["input_tokens"],
            "output_tokens": r.usage_metadata["output_tokens"],
        }

    print("\n=== 8. Model Comparison ===")
    for model, data in results.items():
        print(f"\n  Model : {model}")
        print(f"  Answer: {data['answer']}")
        print(f"  Tokens: {data['input_tokens']} in / {data['output_tokens']} out")


if __name__ == "__main__":
    demo_basic_invoke()
    demo_typed_messages()
    demo_multi_turn()
    demo_streaming()
    demo_batch()
    demo_async()
    demo_model_params()
    demo_model_selection()
