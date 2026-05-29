"""
Python 02-Intermediate — Iterators, Generators & Context Managers
===================================================================
Topics covered:
  1. The iterator protocol: __iter__ / __next__
  2. Generator functions with `yield`
  3. Generator expressions (lazy evaluation)
  4. `yield from` and delegating generators
  5. Context managers with `with` / `__enter__` / `__exit__`
  6. `contextlib.contextmanager` decorator
  7. Practical patterns: streaming LLM output, token budgets, timed contexts

Run:
  python 02_iterators_and_generators.py
"""

from __future__ import annotations

import time
import contextlib
from typing import Iterator, Generator
from dotenv import load_dotenv

load_dotenv()


# ── 1. Iterator protocol ──────────────────────────────────────────────────────
def demo_iterator_protocol():
    print("\n=== 1. Iterator Protocol ===")

    class TokenRange:
        """Iterate over token IDs from start to end."""

        def __init__(self, start: int, end: int):
            self.start   = start
            self.end     = end
            self._current = start

        def __iter__(self) -> "TokenRange":
            self._current = self.start   # reset on each iter() call
            return self

        def __next__(self) -> int:
            if self._current >= self.end:
                raise StopIteration
            val = self._current
            self._current += 1
            return val

    token_ids = TokenRange(0, 5)
    print(f"  token ids: {list(token_ids)}")

    # Can iterate multiple times because __iter__ resets
    for tid in token_ids:
        print(f"    id={tid}", end="  ")
    print()

    # Built-in iter() and next()
    items = ["a", "b", "c"]
    it = iter(items)
    print(f"  next: {next(it)!r}, {next(it)!r}, {next(it)!r}")
    try:
        next(it)
    except StopIteration:
        print("  StopIteration raised — exhausted")


# ── 2. Generator functions ────────────────────────────────────────────────────
def demo_generators():
    print("\n=== 2. Generator Functions ===")

    # A generator function uses `yield` instead of `return`
    def count_tokens(texts: list[str]) -> Generator[dict, None, None]:
        """Yield token-count records one at a time — never holds all in memory."""
        for i, text in enumerate(texts):
            words = len(text.split())
            yield {"index": i, "text": text[:30], "words": words}

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python generators are memory-efficient.",
        "Yield pauses the function and returns a value.",
    ]
    gen = count_tokens(texts)
    print(f"  type: {type(gen).__name__}")   # generator, not a list
    for record in gen:
        print(f"    {record}")

    # Infinite generator — safe because it's lazy
    def token_id_stream(start: int = 0) -> Generator[int, None, None]:
        current = start
        while True:
            yield current
            current += 1

    stream = token_id_stream(100)
    first_five = [next(stream) for _ in range(5)]
    print(f"  first 5 token IDs: {first_five}")

    # Generator with send() — coroutine-lite
    def running_total() -> Generator[float, float, str]:
        total = 0.0
        while True:
            value = yield total
            if value is None:
                return "done"
            total += value

    rt = running_total()
    next(rt)          # prime the generator
    print(f"  running total: {rt.send(10.5):.1f}")
    print(f"  running total: {rt.send(5.25):.1f}")
    print(f"  running total: {rt.send(3.0):.1f}")


# ── 3. Generator expressions ─────────────────────────────────────────────────
def demo_generator_expressions():
    print("\n=== 3. Generator Expressions ===")

    # vs list comprehension — generator is lazy
    import sys
    texts = [f"document {i}" for i in range(1000)]

    list_comp = [len(t) for t in texts]
    gen_expr  = (len(t) for t in texts)

    print(f"  list comprehension size: {sys.getsizeof(list_comp):,} bytes")
    print(f"  generator expression size: {sys.getsizeof(gen_expr)} bytes  ← lazy")

    # Use in functions that accept iterables
    total = sum(len(t) for t in texts)   # no intermediate list created
    print(f"  sum of lengths: {total:,}")

    # Filter + transform pipeline
    documents = [
        {"id": i, "content": f"doc {i} " + ("relevant" if i % 3 == 0 else "irrelevant")}
        for i in range(10)
    ]
    relevant = (
        doc["content"]
        for doc in documents
        if "relevant" in doc["content"]
    )
    print(f"  relevant docs: {list(relevant)}")


# ── 4. yield from ─────────────────────────────────────────────────────────────
def demo_yield_from():
    print("\n=== 4. yield from ===")

    def stream_chunk(text: str, chunk_size: int = 3) -> Generator[str, None, None]:
        """Yield a string in fixed-size chunks (simulates LLM token streaming)."""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def stream_messages(messages: list[str]) -> Generator[str, None, None]:
        """Delegate to per-message generators using yield from."""
        for msg in messages:
            yield from stream_chunk(msg)
            yield "\n"   # separator between messages

    output = list(stream_messages(["Hello!", "How are you?"]))
    print(f"  streamed chunks: {output}")
    print(f"  reconstructed:   {''.join(output)!r}")

    # yield from with iterables — flatten a nested structure
    def flatten(nested: list) -> Generator:
        for item in nested:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item

    nested = [[1, 2, [3, 4]], [5, [6, [7]]]]
    print(f"  flattened: {list(flatten(nested))}")


# ── 5. Context managers — class-based ─────────────────────────────────────────
def demo_context_managers():
    print("\n=== 5. Context Managers ===")

    class Timer:
        """Measure execution time of a code block."""

        def __init__(self, label: str = "block"):
            self.label   = label
            self.elapsed = 0.0

        def __enter__(self) -> "Timer":
            self._start = time.perf_counter()
            return self   # value bound to `as` clause

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            self.elapsed = time.perf_counter() - self._start
            print(f"  [{self.label}] elapsed: {self.elapsed*1000:.2f} ms")
            return False   # do not suppress exceptions

    with Timer("list comprehension") as t:
        data = [i**2 for i in range(100_000)]
    print(f"  stored elapsed: {t.elapsed*1000:.2f} ms")

    # Context manager for resource cleanup
    class MockDBConnection:
        def __init__(self, dsn: str):
            self.dsn    = dsn
            self.closed = False

        def __enter__(self) -> "MockDBConnection":
            print(f"  Connecting to {self.dsn!r}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            self.closed = True
            print(f"  Connection closed (closed={self.closed})")
            return False

        def query(self, sql: str) -> list:
            return [{"row": 1}, {"row": 2}]

    with MockDBConnection("postgresql://localhost/vectors") as db:
        rows = db.query("SELECT * FROM embeddings LIMIT 2")
        print(f"  rows: {rows}")


# ── 6. contextlib.contextmanager ──────────────────────────────────────────────
def demo_contextlib():
    print("\n=== 6. contextlib.contextmanager ===")

    @contextlib.contextmanager
    def trace_call(operation: str) -> Generator[dict, None, None]:
        """Lightweight tracing context — yields a span dict."""
        span: dict = {"operation": operation, "start": time.perf_counter()}
        print(f"  ▶ START {operation}")
        try:
            yield span
        except Exception as e:
            span["error"] = str(e)
            raise
        finally:
            span["duration_ms"] = (time.perf_counter() - span["start"]) * 1000
            span["status"] = "error" if "error" in span else "ok"
            print(f"  ■ END   {operation} — {span['duration_ms']:.2f} ms [{span['status']}]")

    with trace_call("embed_documents") as span:
        time.sleep(0.001)   # simulate work
        span["docs"] = 5

    print(f"  span: {span}")

    # contextlib.suppress — ignore specific exceptions
    with contextlib.suppress(KeyError):
        d = {"a": 1}
        _ = d["missing"]   # would normally raise KeyError
    print("  KeyError suppressed — execution continues")

    # contextlib.redirect_stdout — capture print output
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print("captured output")
    print(f"  captured: {buffer.getvalue()!r}")


# ── 7. Practical: streaming LLM simulation ────────────────────────────────────
def demo_streaming_llm():
    print("\n=== 7. Practical: Streaming LLM Output ===")

    @contextlib.contextmanager
    def token_budget(max_tokens: int):
        """Context manager that tracks and limits token consumption."""
        state = {"used": 0, "limit": max_tokens}
        yield state
        pct = state["used"] / max_tokens * 100
        print(f"\n  Token budget: {state['used']}/{max_tokens} used ({pct:.1f}%)")

    def mock_llm_stream(prompt: str, max_output: int = 20) -> Generator[str, None, None]:
        """Simulate token-by-token streaming from an LLM."""
        words = f"The answer to '{prompt}' is: a detailed and helpful response.".split()
        for word in words[:max_output]:
            time.sleep(0.001)   # simulate network latency
            yield word + " "

    # Stream into a token budget context
    with token_budget(max_tokens=50) as budget:
        print("  Streaming: ", end="", flush=True)
        full_text = ""
        for token in mock_llm_stream("What is Python?"):
            print(token, end="", flush=True)
            full_text += token
            budget["used"] += 1
            if budget["used"] >= budget["limit"]:
                print("  [BUDGET EXCEEDED — truncated]", end="")
                break

    print(f"\n  Full response length: {len(full_text)} chars")


if __name__ == "__main__":
    print("Python 02-Intermediate — Iterators, Generators & Context Managers")
    print("=" * 64)
    demo_iterator_protocol()
    demo_generators()
    demo_generator_expressions()
    demo_yield_from()
    demo_context_managers()
    demo_contextlib()
    demo_streaming_llm()
