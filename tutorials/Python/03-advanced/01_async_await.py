"""
Python 03-Advanced — async/await
==================================
Topics covered:
  1. The event loop and coroutines
  2. async def / await basics
  3. asyncio.gather — concurrent tasks
  4. asyncio.create_task — fire and don't block
  5. async generators — streaming results
  6. asyncio.Queue — producer/consumer pattern
  7. Practical: concurrent LLM API calls with rate limiting

Run:
  python 01_async_await.py
"""

from __future__ import annotations

import asyncio
import time
import random
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()


# ── 1. Coroutines and event loop basics ───────────────────────────────────────
def demo_coroutines():
    print("\n=== 1. Coroutines & Event Loop ===")

    async def greet(name: str) -> str:
        await asyncio.sleep(0.01)    # non-blocking pause
        return f"Hello, {name}!"

    # Run a single coroutine
    result = asyncio.run(greet("Alice"))
    print(f"  {result!r}")

    # A coroutine object is NOT run until awaited or scheduled
    coro = greet("Bob")
    print(f"  coroutine type: {type(coro).__name__}")   # coroutine
    # Run it explicitly
    result2 = asyncio.run(coro)
    print(f"  {result2!r}")


# ── 2. async def / await ──────────────────────────────────────────────────────
def demo_async_await():
    print("\n=== 2. async def / await ===")

    async def mock_llm_call(prompt: str, latency: float = 0.05) -> dict:
        """Simulate an async LLM API call."""
        await asyncio.sleep(latency)
        return {
            "prompt":   prompt,
            "response": f"[mock] Answer to: '{prompt}'",
            "tokens":   len(prompt.split()) * 2,
        }

    async def pipeline(prompts: list[str]) -> list[dict]:
        results = []
        for p in prompts:
            result = await mock_llm_call(p, latency=0.02)   # sequential
            results.append(result)
        return results

    t0 = time.perf_counter()
    results = asyncio.run(pipeline(["What is AI?", "What is Python?", "What is asyncio?"]))
    elapsed = time.perf_counter() - t0

    for r in results:
        print(f"  {r['prompt']!r} → tokens={r['tokens']}")
    print(f"  sequential elapsed: {elapsed*1000:.0f}ms (3 × 20ms)")


# ── 3. asyncio.gather — concurrent tasks ──────────────────────────────────────
def demo_gather():
    print("\n=== 3. asyncio.gather (concurrent) ===")

    async def mock_llm_call(prompt: str, latency: float = 0.05) -> dict:
        await asyncio.sleep(latency)
        return {"prompt": prompt, "response": f"[mock] {prompt[:20]}"}

    async def concurrent_pipeline(prompts: list[str]) -> list[dict]:
        tasks = [mock_llm_call(p, latency=0.05) for p in prompts]
        return await asyncio.gather(*tasks)   # all run concurrently

    t0 = time.perf_counter()
    prompts = ["What is AI?", "What is ML?", "What is DL?", "What is NLP?", "What is CV?"]
    results = asyncio.run(concurrent_pipeline(prompts))
    elapsed = time.perf_counter() - t0

    for r in results:
        print(f"  {r['prompt']!r}")
    print(f"  concurrent elapsed: {elapsed*1000:.0f}ms (5 tasks × 50ms, overlapping)")

    # gather with return_exceptions=True — don't fail fast
    async def maybe_fail(name: str, fail: bool) -> str:
        await asyncio.sleep(0.01)
        if fail:
            raise ValueError(f"{name} failed!")
        return f"{name} ok"

    async def resilient_gather():
        results = await asyncio.gather(
            maybe_fail("A", False),
            maybe_fail("B", True),
            maybe_fail("C", False),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                print(f"  exception: {r}")
            else:
                print(f"  success:   {r}")

    asyncio.run(resilient_gather())


# ── 4. asyncio.create_task ────────────────────────────────────────────────────
def demo_create_task():
    print("\n=== 4. asyncio.create_task ===")

    async def background_log(message: str, delay: float):
        await asyncio.sleep(delay)
        print(f"  [background] {message}")

    async def main():
        # Schedule background tasks — don't wait for them immediately
        task1 = asyncio.create_task(background_log("Indexing complete",  0.02))
        task2 = asyncio.create_task(background_log("Cache warmed up",    0.01))

        print("  [main] doing other work while tasks run...")
        await asyncio.sleep(0.005)
        print("  [main] still working...")

        # Explicitly await tasks when needed
        await task2   # task2 finishes first (shorter delay)
        await task1

    asyncio.run(main())

    # Task groups (Python 3.11+)
    async def task_group_demo():
        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(background_log("Task group A", 0.01))
            t2 = tg.create_task(background_log("Task group B", 0.02))
        # All tasks in the group are done here
        print("  [task group] all done")

    try:
        asyncio.run(task_group_demo())
    except AttributeError:
        print("  (TaskGroup requires Python 3.11+)")


# ── 5. Async generators ───────────────────────────────────────────────────────
def demo_async_generators():
    print("\n=== 5. Async Generators ===")

    async def stream_tokens(text: str, delay: float = 0.005) -> AsyncGenerator[str, None]:
        """Yield tokens one at a time with async delay (simulates SSE streaming)."""
        words = text.split()
        for word in words:
            await asyncio.sleep(delay)
            yield word + " "

    async def collect_stream(text: str) -> str:
        full = ""
        async for token in stream_tokens(text):
            full += token
        return full.strip()

    result = asyncio.run(collect_stream("The quick brown fox jumps over the lazy dog"))
    print(f"  streamed & collected: {result!r}")

    # Async generator with timeout
    async def bounded_stream(text: str, max_tokens: int = 5) -> AsyncGenerator[str, None]:
        count = 0
        async for token in stream_tokens(text, delay=0.001):
            yield token
            count += 1
            if count >= max_tokens:
                break

    async def main():
        tokens = []
        async for t in bounded_stream("the quick brown fox jumps over the lazy dog", max_tokens=4):
            tokens.append(t.strip())
        print(f"  bounded (max 4 tokens): {tokens}")

    asyncio.run(main())


# ── 6. asyncio.Queue — producer / consumer ────────────────────────────────────
def demo_queue():
    print("\n=== 6. asyncio.Queue (Producer/Consumer) ===")

    async def producer(queue: asyncio.Queue, prompts: list[str]):
        for prompt in prompts:
            await queue.put(prompt)
            print(f"  [producer] queued: {prompt!r}")
            await asyncio.sleep(0.005)
        # Sentinel to signal completion
        await queue.put(None)

    async def consumer(queue: asyncio.Queue, worker_id: int):
        results = []
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            await asyncio.sleep(0.01)   # simulate processing
            result = f"[W{worker_id}] done: {item!r}"
            results.append(result)
            print(f"  {result}")
            queue.task_done()
        return results

    async def main():
        queue = asyncio.Queue(maxsize=3)
        prompts = ["Q1: What?", "Q2: Why?", "Q3: How?", "Q4: When?"]

        prod = asyncio.create_task(producer(queue, prompts))
        cons = asyncio.create_task(consumer(queue, worker_id=1))
        await asyncio.gather(prod, cons)

    asyncio.run(main())


# ── 7. Practical: concurrent LLM calls with semaphore ─────────────────────────
def demo_concurrent_llm():
    print("\n=== 7. Practical: Concurrent LLM Calls with Rate Limit ===")

    async def mock_llm(prompt: str, model: str = "gpt-4o-mini") -> dict:
        latency = random.uniform(0.02, 0.08)
        await asyncio.sleep(latency)
        return {
            "prompt":   prompt,
            "model":    model,
            "response": f"Mock response to: {prompt[:30]}",
            "latency":  round(latency * 1000, 1),
        }

    async def batch_llm(prompts: list[str], max_concurrent: int = 3) -> list[dict]:
        """Process a list of prompts with bounded concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def throttled_call(prompt: str) -> dict:
            async with semaphore:
                return await mock_llm(prompt)

        tasks = [throttled_call(p) for p in prompts]
        return await asyncio.gather(*tasks)

    prompts = [
        "Explain Python asyncio",
        "What is a semaphore?",
        "Describe the GIL",
        "What are coroutines?",
        "How does event loop work?",
    ]

    t0 = time.perf_counter()
    results = asyncio.run(batch_llm(prompts, max_concurrent=3))
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Processed {len(results)} prompts with max_concurrent=3")
    for r in results:
        print(f"    [{r['latency']}ms] {r['prompt'][:35]!r}")
    print(f"  Total elapsed: {elapsed:.0f}ms")


if __name__ == "__main__":
    print("Python 03-Advanced — async/await")
    print("=" * 38)
    demo_coroutines()
    demo_async_await()
    demo_gather()
    demo_create_task()
    demo_async_generators()
    demo_queue()
    demo_concurrent_llm()
