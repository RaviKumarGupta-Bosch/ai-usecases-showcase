"""
Ollama 01-Basics — Generate & Chat
=====================================
Topics covered:
  1. ollama.generate() — single-turn completion
  2. ollama.chat() — messages API (multi-turn compatible)
  3. Streaming tokens in real time
  4. Generation options: temperature, top_p, num_predict
  5. Multi-turn conversation (accumulating message history)

Prerequisites:
  - Ollama running: `ollama serve`
  - Model pulled: `ollama pull llama3.2`

Run:
  python 02_generate_and_chat.py
"""

import os
import sys
import time
from dotenv import load_dotenv
import ollama

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def check_ollama() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


# ── 1. Basic generate ─────────────────────────────────────────────────────────
def demo_generate():
    print("\n=== 1. ollama.generate() — Single-Turn Completion ===")
    prompt = "In one sentence, explain what a large language model is."

    print(f"  Prompt: {prompt}")
    response = ollama.generate(model=MODEL, prompt=prompt)
    text = response.response if hasattr(response, "response") else response.get("response", "")
    print(f"  Response: {text.strip()}")

    # Metadata
    eval_count = response.eval_count if hasattr(response, "eval_count") else response.get("eval_count", 0)
    total_ns   = response.total_duration if hasattr(response, "total_duration") else response.get("total_duration", 0)
    total_s    = total_ns / 1e9 if total_ns else 0
    print(f"  Tokens generated: {eval_count}  |  Time: {total_s:.2f}s")


# ── 2. Chat API ───────────────────────────────────────────────────────────────
def demo_chat():
    print("\n=== 2. ollama.chat() — Messages API ===")
    messages = [
        {"role": "system",    "content": "You are a concise Python expert. Answer in 2-3 sentences."},
        {"role": "user",      "content": "What is the GIL in Python?"},
    ]

    response = ollama.chat(model=MODEL, messages=messages)
    content = response.message.content if hasattr(response, "message") else response["message"]["content"]
    print(f"  Q: What is the GIL in Python?")
    print(f"  A: {content.strip()}")


# ── 3. Streaming ─────────────────────────────────────────────────────────────
def demo_streaming():
    print("\n=== 3. Streaming Tokens ===")
    prompt = "List 5 benefits of running LLMs locally with Ollama."

    print(f"  Prompt: {prompt}")
    print("  Response (streaming): ", end="", flush=True)

    full_text = ""
    for chunk in ollama.generate(model=MODEL, prompt=prompt, stream=True):
        token = chunk.response if hasattr(chunk, "response") else chunk.get("response", "")
        print(token, end="", flush=True)
        full_text += token
        done = chunk.done if hasattr(chunk, "done") else chunk.get("done", False)
        if done:
            break

    print()  # newline after streaming
    print(f"\n  Total chars streamed: {len(full_text)}")


# ── 4. Generation options ─────────────────────────────────────────────────────
def demo_options():
    print("\n=== 4. Generation Options (temperature, top_p, num_predict) ===")
    prompt = "Give me a creative name for an AI assistant."

    configs = [
        {"label": "Deterministic  (temp=0.0)", "options": {"temperature": 0.0, "num_predict": 40}},
        {"label": "Balanced       (temp=0.7)", "options": {"temperature": 0.7, "num_predict": 40}},
        {"label": "Creative       (temp=1.4)", "options": {"temperature": 1.4, "num_predict": 40}},
    ]

    print(f"  Prompt: {prompt}\n")
    for cfg in configs:
        response = ollama.generate(model=MODEL, prompt=prompt, options=cfg["options"])
        text = response.response if hasattr(response, "response") else response.get("response", "")
        print(f"  [{cfg['label']}]")
        print(f"    {text.strip()[:120]}\n")


# ── 5. Multi-turn conversation ────────────────────────────────────────────────
def demo_multi_turn():
    print("\n=== 5. Multi-Turn Conversation ===")
    history = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."}
    ]

    turns = [
        "My name is Alex and I'm learning Python.",
        "What is the best first project for a Python beginner?",
        "Do you remember my name?",
    ]

    for user_msg in turns:
        history.append({"role": "user", "content": user_msg})
        response = ollama.chat(model=MODEL, messages=history)
        assistant_msg = (
            response.message.content
            if hasattr(response, "message")
            else response["message"]["content"]
        )
        history.append({"role": "assistant", "content": assistant_msg})

        print(f"\n  User:      {user_msg}")
        print(f"  Assistant: {assistant_msg.strip()[:200]}")

    print(f"\n  Total messages in history: {len(history)}")


if __name__ == "__main__":
    print("Ollama 01-Basics — Generate & Chat")
    print("=" * 45)

    if not check_ollama():
        print("\n  ERROR: Cannot connect to Ollama. Start with: ollama serve")
        sys.exit(1)

    demo_generate()
    demo_chat()
    demo_streaming()
    demo_options()
    demo_multi_turn()
