"""
Python 01-Basics — Functions & Scope
======================================
Topics covered:
  1. Defining functions — positional, keyword, defaults
  2. *args and **kwargs
  3. Scope: LEGB rule and `global` / `nonlocal`
  4. Closures and factory functions
  5. Lambda and anonymous functions
  6. Higher-order functions: map, filter, sorted with key
  7. Practical patterns: LLM call wrappers, config builders

Run:
  python 02_functions_and_scope.py
"""

from dotenv import load_dotenv
import time
load_dotenv()


# ── 1. Function basics ────────────────────────────────────────────────────────
def demo_function_basics():
    print("\n=== 1. Function Basics ===")

    def greet(name: str, greeting: str = "Hello") -> str:
        """Return a greeting string."""
        return f"{greeting}, {name}!"

    print(f"  {greet('Alice')}")
    print(f"  {greet('Bob', greeting='Hi')}")
    print(f"  {greet(greeting='Hey', name='Carol')}")   # kwargs can be in any order

    # Return multiple values (as tuple)
    def split_tokens(text: str) -> tuple[list[str], int]:
        words = text.split()
        return words, len(words)

    tokens, count = split_tokens("the quick brown fox")
    print(f"  tokens={tokens}, count={count}")

    # Early return / guard clause pattern
    def safe_divide(a: float, b: float) -> float | None:
        if b == 0:
            return None
        return a / b

    print(f"  10/2 = {safe_divide(10, 2)}")
    print(f"  10/0 = {safe_divide(10, 0)!r}")


# ── 2. *args and **kwargs ─────────────────────────────────────────────────────
def demo_args_kwargs():
    print("\n=== 2. *args and **kwargs ===")

    # *args — variable positional arguments
    def sum_tokens(*counts: int) -> int:
        return sum(counts)

    print(f"  sum_tokens(100, 200, 50) = {sum_tokens(100, 200, 50)}")

    # **kwargs — variable keyword arguments
    def build_llm_params(model: str, **kwargs) -> dict:
        params = {"model": model}
        params.update(kwargs)
        return params

    params = build_llm_params("gpt-4o-mini", temperature=0.7, max_tokens=512, stream=True)
    print(f"  params: {params}")

    # Combining all: positional, *args, keyword-only, **kwargs
    def log_call(fn_name: str, *args, level: str = "INFO", **kwargs):
        arg_str  = ", ".join(str(a) for a in args)
        kwarg_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
        print(f"  [{level}] {fn_name}({all_args})")

    log_call("chat", "Hello", "World", level="DEBUG", model="gpt-4o")

    # Spreading args/kwargs into another function call
    defaults = {"temperature": 0.0, "max_tokens": 256}
    overrides = {"model": "gpt-4o-mini", "temperature": 0.5}
    merged = {**defaults, **overrides}     # later keys win
    print(f"  merged config: {merged}")


# ── 3. Scope: LEGB rule ───────────────────────────────────────────────────────
def demo_scope():
    print("\n=== 3. Scope (LEGB) ===")

    # Local scope — variable only lives inside function
    def fn():
        local_var = "I'm local"
        return local_var

    print(f"  local: {fn()!r}")

    # Enclosing scope (used by closures — see next section)
    # Global scope
    _call_count = 0

    def increment():
        global _call_count
        _call_count += 1

    increment(); increment()
    print(f"  global _call_count after 2 calls: {_call_count}")

    # nonlocal — modify enclosing (non-global) variable
    def make_counter():
        count = 0
        def inc():
            nonlocal count
            count += 1
            return count
        return inc

    counter = make_counter()
    print(f"  nonlocal counter: {counter()} {counter()} {counter()}")


# ── 4. Closures and factory functions ─────────────────────────────────────────
def demo_closures():
    print("\n=== 4. Closures & Factories ===")

    # Closure captures the enclosing environment
    def make_prefix_logger(prefix: str):
        def log(msg: str) -> str:
            return f"[{prefix}] {msg}"
        return log

    info  = make_prefix_logger("INFO")
    error = make_prefix_logger("ERROR")
    print(f"  {info('Agent started')}")
    print(f"  {error('Tool call failed')}")

    # Factory: pre-configure an LLM caller
    def make_llm_caller(model: str, temperature: float = 0.0):
        """Return a function that simulates calling a specific LLM."""
        def call(prompt: str) -> str:
            # In real code: return openai.chat.completions.create(...)
            return f"[{model}@T={temperature}] Response to: '{prompt}'"
        return call

    fast_model  = make_llm_caller("gpt-4o-mini", temperature=0.0)
    creative_model = make_llm_caller("gpt-4o",  temperature=0.9)
    print(f"  {fast_model('Summarise this doc')}")
    print(f"  {creative_model('Write a poem')}")

    # Accumulator closure — useful for token counting
    def make_token_accumulator():
        total = {"prompt": 0, "completion": 0}
        def add(prompt_tokens: int, completion_tokens: int):
            total["prompt"]     += prompt_tokens
            total["completion"] += completion_tokens
            return dict(total)
        add.get_total = lambda: dict(total)   # attach helper
        return add

    track = make_token_accumulator()
    track(100, 50)
    track(200, 80)
    print(f"  accumulated tokens: {track.get_total()}")


# ── 5. Lambda and anonymous functions ─────────────────────────────────────────
def demo_lambda():
    print("\n=== 5. Lambda Functions ===")

    # Simple transformation
    to_upper = lambda s: s.upper()
    print(f"  to_upper: {to_upper('hello')!r}")

    # Lambdas as sort keys
    models = [
        {"name": "gpt-4o",      "cost": 0.005},
        {"name": "gpt-4o-mini", "cost": 0.00015},
        {"name": "o1-preview",  "cost": 0.015},
    ]
    cheapest_first = sorted(models, key=lambda m: m["cost"])
    print("  sorted by cost:")
    for m in cheapest_first:
        print(f"    {m['name']:<15} ${m['cost']}")

    # Lambda in pipeline step (common in LangChain)
    def pipeline(*fns):
        def run(data):
            for fn in fns:
                data = fn(data)
            return data
        return run

    process = pipeline(
        lambda s: s.strip(),
        lambda s: s.lower(),
        lambda s: s.replace(" ", "_"),
    )
    print(f"  pipeline result: {process('  Hello World  ')!r}")

    # When NOT to use lambda — use def for anything complex
    # Good: key=lambda x: x['score']
    # Bad:  fn = lambda x: x if x > 0 else -x   # use def abs_val(x)


# ── 6. Higher-order functions ─────────────────────────────────────────────────
def demo_higher_order():
    print("\n=== 6. Higher-Order Functions ===")

    scores = [0.45, 0.88, 0.92, 0.61, 0.79, 0.95, 0.30]

    # map — transform each element
    pct = list(map(lambda s: f"{s*100:.1f}%", scores))
    print(f"  map to %: {pct}")

    # filter — keep elements matching predicate
    passing = list(filter(lambda s: s >= 0.7, scores))
    print(f"  filter ≥0.7: {passing}")

    # sorted with complex key
    results = [
        {"model": "A", "speed": 0.9, "quality": 0.7},
        {"model": "B", "speed": 0.6, "quality": 0.95},
        {"model": "C", "speed": 0.8, "quality": 0.85},
    ]
    # rank by composite score
    ranked = sorted(results, key=lambda r: r["speed"] * 0.3 + r["quality"] * 0.7, reverse=True)
    print("  ranked by composite (0.3*speed + 0.7*quality):")
    for r in ranked:
        score = r["speed"] * 0.3 + r["quality"] * 0.7
        print(f"    {r['model']}: {score:.3f}")

    # any / all — useful guards
    responses = [True, True, True, False]
    print(f"  all passed: {all(responses)}")
    print(f"  any passed: {any(responses)}")


# ── 7. Practical patterns ─────────────────────────────────────────────────────
def demo_practical_patterns():
    print("\n=== 7. Practical Patterns ===")

    # 1. Memoisation (simple cache)
    _cache: dict = {}
    def memoised_embed(text: str) -> list[float]:
        if text in _cache:
            print(f"    cache HIT:  {text!r}")
            return _cache[text]
        print(f"    cache MISS: {text!r} — computing...")
        result = [float(ord(c)) for c in text[:4]]   # mock embedding
        _cache[text] = result
        return result

    memoised_embed("hello")
    memoised_embed("world")
    memoised_embed("hello")   # cache hit

    # 2. Timed execution decorator (preview — full decorators in 02-intermediate)
    def timed(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"    {fn.__name__} took {elapsed*1000:.2f} ms")
            return result
        return wrapper

    @timed
    def slow_process(n: int) -> int:
        return sum(range(n))

    slow_process(100_000)

    # 3. Config builder pattern
    def llm_config(
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **extras,
    ) -> dict:
        """Build a validated LLM config dict."""
        assert 0.0 <= temperature <= 2.0, "temperature out of range"
        return {"model": model, "temperature": temperature,
                "max_tokens": max_tokens, **extras}

    cfg = llm_config(model="gpt-4o", temperature=0.5, stream=True)
    print(f"\n  llm_config: {cfg}")


if __name__ == "__main__":
    print("Python 01-Basics — Functions & Scope")
    print("=" * 44)
    demo_function_basics()
    demo_args_kwargs()
    demo_scope()
    demo_closures()
    demo_lambda()
    demo_higher_order()
    demo_practical_patterns()
