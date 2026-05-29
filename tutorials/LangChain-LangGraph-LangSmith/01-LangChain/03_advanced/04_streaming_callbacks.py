"""
04 - Streaming & Callbacks
===========================
Streaming lets you display tokens as they're generated — critical for responsive UIs.
Callbacks let you hook into every step of chain/agent execution.

Topics covered:
  1. .stream() — synchronous token streaming
  2. .astream() — async token streaming
  3. .astream_events() — full event stream (tokens + tool calls + chain events)
  4. BaseCallbackHandler — custom sync callback
  5. AsyncCallbackHandler — custom async callback
  6. Streaming through a full chain (RAG)
  7. Token usage tracking via callbacks
"""

import asyncio
import time
from typing import Any, Union
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult, ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_det = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. Synchronous streaming ─────────────────────────────────────────────────
def demo_sync_streaming():
    print("=== 1. Synchronous Streaming (.stream()) ===")
    print("Generating: ", end="", flush=True)

    for chunk in llm.stream("Write a 3-sentence story about a robot learning to paint."):
        print(chunk.content, end="", flush=True)

    print()  # newline at end


# ── 2. Chain streaming ───────────────────────────────────────────────────────
def demo_chain_streaming():
    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about {topic}. Then explain the meaning in one sentence."
    )
    chain = prompt | llm | StrOutputParser()

    print("\n=== 2. Chain Streaming ===")
    print("Generating haiku: ", end="", flush=True)

    full_response = ""
    for chunk in chain.stream({"topic": "artificial intelligence"}):
        print(chunk, end="", flush=True)
        full_response += chunk

    print(f"\n\nTotal length: {len(full_response)} characters")


# ── 3. Async streaming ───────────────────────────────────────────────────────
async def demo_async_streaming():
    prompt = ChatPromptTemplate.from_template("List 5 creative uses for {technology} in education.")
    chain = prompt | llm | StrOutputParser()

    print("\n=== 3. Async Streaming (.astream()) ===")
    print("Generating: ", end="", flush=True)

    async for chunk in chain.astream({"technology": "augmented reality"}):
        print(chunk, end="", flush=True)

    print()


# ── 4. astream_events — full event stream ────────────────────────────────────
async def demo_astream_events():
    """
    astream_events yields structured events for every step:
    - on_chat_model_start / on_chat_model_stream / on_chat_model_end
    - on_chain_start / on_chain_end
    Useful for building rich UIs that show intermediate states.
    """
    chain = (
        ChatPromptTemplate.from_template("Translate '{text}' to French.")
        | llm
        | StrOutputParser()
    )

    print("\n=== 4. astream_events ===")
    token_count = 0

    async for event in chain.astream_events(
        {"text": "Hello, how are you today?"},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
                token_count += 1
        elif kind == "on_chain_end":
            pass  # final output captured elsewhere

    print(f"\n\nStreamed {token_count} token chunks")


# ── 5. Custom sync callback handler ─────────────────────────────────────────
class LoggingCallbackHandler(BaseCallbackHandler):
    """Logs every major event with timestamps."""

    def __init__(self):
        self.events = []
        self.start_time = None

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        self.start_time = time.time()
        print(f"\n  [CALLBACK] LLM started | prompts: {len(prompts)}")

    def on_llm_new_token(self, token: str, **kwargs):
        pass  # suppress per-token logging for readability

    def on_llm_end(self, response: LLMResult, **kwargs):
        elapsed = time.time() - self.start_time
        total_tokens = response.llm_output.get("token_usage", {}).get("total_tokens", "?")
        print(f"  [CALLBACK] LLM ended  | elapsed={elapsed:.2f}s | tokens={total_tokens}")
        self.events.append({"event": "llm_end", "elapsed": elapsed, "tokens": total_tokens})

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        chain_name = serialized.get("name", "unknown")
        print(f"  [CALLBACK] Chain '{chain_name}' started")

    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"  [CALLBACK] Chain ended")


def demo_sync_callback():
    handler = LoggingCallbackHandler()
    prompt = ChatPromptTemplate.from_template("Summarise {topic} in two sentences.")
    chain = prompt | llm_det | StrOutputParser()

    print("\n=== 5. Custom Sync Callback ===")
    result = chain.invoke({"topic": "quantum computing"}, config={"callbacks": [handler]})
    print(f"Answer: {result}")
    print(f"Events logged: {len(handler.events)}")


# ── 6. Token usage tracking callback ────────────────────────────────────────
class TokenUsageTracker(BaseCallbackHandler):
    """Accumulate token usage across multiple LLM calls."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.call_count += 1

    @property
    def total_tokens(self):
        return self.total_prompt_tokens + self.total_completion_tokens

    def report(self):
        print(f"  Calls     : {self.call_count}")
        print(f"  Prompt    : {self.total_prompt_tokens} tokens")
        print(f"  Completion: {self.total_completion_tokens} tokens")
        print(f"  Total     : {self.total_tokens} tokens")
        # Estimated cost for gpt-4o-mini (approximate)
        cost = (self.total_prompt_tokens * 0.00015 + self.total_completion_tokens * 0.0006) / 1000
        print(f"  Est. Cost : ${cost:.6f} USD")


def demo_token_tracking():
    tracker = TokenUsageTracker()
    chain = (
        ChatPromptTemplate.from_template("What is {concept}?")
        | ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[tracker])
        | StrOutputParser()
    )

    topics = ["machine learning", "neural networks", "gradient descent"]

    print("\n=== 6. Token Usage Tracking ===")
    for topic in topics:
        answer = chain.invoke({"concept": topic})
        print(f"  {topic}: {answer[:60]}...")

    print("\nUsage Summary:")
    tracker.report()


# ── 7. Async callback ────────────────────────────────────────────────────────
class AsyncProgressCallback(AsyncCallbackHandler):
    """Async callback that tracks streaming progress."""

    def __init__(self):
        self.token_buffer = []

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.token_buffer.append(token)
        if len(self.token_buffer) % 10 == 0:
            print(f"  [ASYNC] {len(self.token_buffer)} tokens received...", end="\r")

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"  [ASYNC] Done! {len(self.token_buffer)} total tokens")


async def demo_async_callback():
    callback = AsyncProgressCallback()
    llm_streaming = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True,
        callbacks=[callback],
    )

    print("\n=== 7. Async Callback ===")
    response = await llm_streaming.ainvoke(
        "Write a detailed explanation of how neural networks learn, covering forward pass, "
        "loss function, backpropagation, and gradient descent."
    )
    print(f"Response length: {len(response.content)} chars")


async def run_async_demos():
    await demo_async_streaming()
    await demo_astream_events()
    await demo_async_callback()


if __name__ == "__main__":
    demo_sync_streaming()
    demo_chain_streaming()
    asyncio.run(run_async_demos())
    demo_sync_callback()
    demo_token_tracking()
