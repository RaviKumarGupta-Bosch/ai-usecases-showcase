"""
Python 02-Intermediate — Decorators
=====================================
Topics covered:
  1. What decorators are — functions as first-class objects
  2. Writing a basic decorator
  3. `functools.wraps` — preserving metadata
  4. Decorator with arguments (decorator factory)
  5. Class-based decorators
  6. Stacking multiple decorators
  7. Practical AI patterns: retry, rate-limit, cache, timed, log-calls

Run:
  python 03_decorators.py
"""

from __future__ import annotations

import time
import functools
import logging
from collections import OrderedDict
from typing import Callable, TypeVar, Any
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── 1 & 2. Basic decorator ────────────────────────────────────────────────────
def demo_basic_decorator():
    print("\n=== 1 & 2. Basic Decorator ===")

    # Functions are first-class — they can be passed and returned
    def shout(fn: Callable) -> Callable:
        """Wraps fn and uppercases its return value."""
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            return str(result).upper()
        return wrapper

    @shout                          # syntactic sugar for: greet = shout(greet)
    def greet(name: str) -> str:
        return f"hello, {name}"

    print(f"  {greet('world')!r}")

    # Without @-syntax (equivalent)
    def greet2(name: str) -> str:
        return f"hello, {name}"

    greet2 = shout(greet2)
    print(f"  {greet2('python')!r}")

    # Problem: wrapper hides the original function's metadata
    print(f"  name without functools.wraps: {greet.__name__!r}")


# ── 3. functools.wraps ────────────────────────────────────────────────────────
def demo_functools_wraps():
    print("\n=== 3. functools.wraps ===")

    def timed(fn: F) -> F:
        @functools.wraps(fn)          # copies __name__, __doc__, __annotations__
        def wrapper(*args, **kwargs):
            t0     = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [{fn.__name__}] {elapsed:.2f} ms")
            return result
        return wrapper  # type: ignore[return-value]

    @timed
    def embed_text(text: str) -> list[float]:
        """Return a mock embedding vector."""
        time.sleep(0.005)
        return [float(ord(c)) for c in text[:4]]

    vec = embed_text("hello")
    print(f"  embedding: {vec}")
    print(f"  __name__:  {embed_text.__name__!r}")    # preserved
    print(f"  __doc__:   {embed_text.__doc__!r}")     # preserved


# ── 4. Decorator factory (decorator with arguments) ───────────────────────────
def demo_decorator_factory():
    print("\n=== 4. Decorator Factory (with arguments) ===")

    def retry(max_attempts: int = 3, delay: float = 0.1, exceptions: tuple = (Exception,)):
        """Decorator factory: retry the wrapped function on failure."""
        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                last_exc: Exception | None = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return fn(*args, **kwargs)
                    except exceptions as e:
                        last_exc = e
                        print(f"  attempt {attempt}/{max_attempts} failed: {e}")
                        if attempt < max_attempts:
                            time.sleep(delay)
                raise RuntimeError(f"All {max_attempts} attempts failed") from last_exc
            return wrapper  # type: ignore[return-value]
        return decorator

    call_count = 0

    @retry(max_attempts=3, delay=0.01)
    def flaky_api_call(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Simulated network error")
        return f"Success on attempt {call_count}: response to '{prompt}'"

    result = flaky_api_call("What is AI?")
    print(f"  result: {result!r}")

    # Decorator with optional arguments (works with and without parens)
    def log(fn=None, *, level: str = "INFO"):
        def decorator(f: F) -> F:
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                print(f"  [{level}] calling {f.__name__}()")
                return f(*args, **kwargs)
            return wrapper  # type: ignore[return-value]
        return decorator(fn) if fn else decorator

    @log
    def step_a(): return "A"

    @log(level="DEBUG")
    def step_b(): return "B"

    print(f"  {step_a()}, {step_b()}")


# ── 5. Class-based decorator ──────────────────────────────────────────────────
def demo_class_decorator():
    print("\n=== 5. Class-Based Decorator ===")

    class RateLimit:
        """Allow at most `calls` per `period` seconds."""

        def __init__(self, calls: int, period: float = 1.0):
            self.calls      = calls
            self.period     = period
            self.timestamps: list[float] = []

        def __call__(self, fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                now    = time.monotonic()
                cutoff = now - self.period
                # Remove old timestamps outside the window
                self.timestamps = [t for t in self.timestamps if t > cutoff]
                if len(self.timestamps) >= self.calls:
                    wait = self.period - (now - self.timestamps[0])
                    print(f"  Rate limit hit — would wait {wait:.3f}s (skipping in demo)")
                    return None
                self.timestamps.append(now)
                return fn(*args, **kwargs)
            return wrapper  # type: ignore[return-value]

    @RateLimit(calls=2, period=1.0)
    def search(query: str) -> str:
        return f"Results for '{query}'"

    print(f"  call 1: {search('python')!r}")
    print(f"  call 2: {search('AI')!r}")
    print(f"  call 3: {search('LLM')!r}")   # rate-limited


# ── 6. Stacking decorators ────────────────────────────────────────────────────
def demo_stacking():
    print("\n=== 6. Stacking Decorators ===")

    # Applied bottom-up: @timed → @retry → function
    def timed(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            r  = fn(*args, **kwargs)
            print(f"  [{fn.__name__}] {(time.perf_counter()-t0)*1000:.1f}ms")
            return r
        return wrapper  # type: ignore[return-value]

    def validate_input(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(prompt: str, *args, **kwargs):
            if not prompt or not prompt.strip():
                raise ValueError("prompt cannot be empty")
            return fn(prompt.strip(), *args, **kwargs)
        return wrapper  # type: ignore[return-value]

    def log_result(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            print(f"  [LOG] {fn.__name__} → {str(result)[:60]!r}")
            return result
        return wrapper  # type: ignore[return-value]

    @timed
    @log_result
    @validate_input
    def call_llm(prompt: str) -> str:
        time.sleep(0.002)
        return f"Response to: {prompt}"

    try:
        result = call_llm("  What is a decorator?  ")
        call_llm("")   # should raise ValueError
    except ValueError as e:
        print(f"  Validation caught: {e}")


# ── 7. Practical: LRU cache & memoisation ────────────────────────────────────
def demo_practical_cache():
    print("\n=== 7. Practical: Cache, Memoisation & functools ===")

    # functools.lru_cache — built-in memoisation
    @functools.lru_cache(maxsize=128)
    def expensive_embed(text: str) -> tuple[float, ...]:
        """Simulate a costly embedding call (cached)."""
        time.sleep(0.01)
        return tuple(float(ord(c)) for c in text[:4])

    t0 = time.perf_counter()
    v1 = expensive_embed("hello")
    first_call = time.perf_counter() - t0

    t0 = time.perf_counter()
    v2 = expensive_embed("hello")   # cache hit
    cached_call = time.perf_counter() - t0

    print(f"  first call:  {first_call*1000:.1f}ms → {v1}")
    print(f"  cached call: {cached_call*1000:.2f}ms → {v2}")
    print(f"  cache info: {expensive_embed.cache_info()}")

    # Custom LRU cache decorator (illustrative)
    def lru(maxsize: int = 4):
        def decorator(fn: F) -> F:
            cache: OrderedDict = OrderedDict()
            @functools.wraps(fn)
            def wrapper(*args):
                if args in cache:
                    cache.move_to_end(args)
                    return cache[args]
                result = fn(*args)
                cache[args] = result
                if len(cache) > maxsize:
                    cache.popitem(last=False)
                return result
            wrapper.cache = cache   # type: ignore[attr-defined]
            return wrapper  # type: ignore[return-value]
        return decorator

    @lru(maxsize=3)
    def tokenize(text: str) -> list[str]:
        return text.split()

    tokenize("the quick brown fox")
    tokenize("hello world")
    tokenize("the quick brown fox")   # cache hit
    print(f"  custom lru cache size: {len(tokenize.cache)}")

    # functools.partial — pre-fill arguments
    import functools as ft

    def call_model(model: str, temperature: float, prompt: str) -> str:
        return f"[{model}@{temperature}] {prompt[:30]}"

    fast_chat = ft.partial(call_model, "gpt-4o-mini", 0.0)
    creative  = ft.partial(call_model, "gpt-4o",      0.9)

    print(f"  fast: {fast_chat('Summarise this.')!r}")
    print(f"  creative: {creative('Write a haiku.')!r}")


if __name__ == "__main__":
    print("Python 02-Intermediate — Decorators")
    print("=" * 42)
    demo_basic_decorator()
    demo_functools_wraps()
    demo_decorator_factory()
    demo_class_decorator()
    demo_stacking()
    demo_practical_cache()
