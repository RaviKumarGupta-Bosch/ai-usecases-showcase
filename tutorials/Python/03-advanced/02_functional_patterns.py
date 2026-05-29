"""
Python 03-Advanced — Functional Patterns
==========================================
Topics covered:
  1. Pure functions and immutability
  2. map, filter, reduce — classic functional tools
  3. functools.partial — partial application
  4. functools.reduce and operator module
  5. Pipelines with function composition
  6. itertools — combinatorial and infinite iterators
  7. Practical: functional data transformation for AI pipelines

Run:
  python 02_functional_patterns.py
"""

from __future__ import annotations

import functools
import operator
import itertools
from typing import Callable, TypeVar, Iterable
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")
U = TypeVar("U")


# ── 1. Pure functions ─────────────────────────────────────────────────────────
def demo_pure_functions():
    print("\n=== 1. Pure Functions ===")

    # Pure: same input → same output, no side-effects
    def normalise_score(score: float, min_s: float = 0.0, max_s: float = 1.0) -> float:
        return max(min_s, min(max_s, score))

    # Impure (reads/mutates external state) — avoid where possible
    _scores: list[float] = []
    def record_score(score: float):   # side-effect: modifies _scores
        _scores.append(score)

    scores = [0.45, 1.2, -0.1, 0.88, 0.95]
    normalised = list(map(normalise_score, scores))
    print(f"  original:   {scores}")
    print(f"  normalised: {normalised}")

    # Prefer returning new objects over mutating in-place
    def add_field(record: dict, key: str, value) -> dict:
        return {**record, key: value}   # new dict, original untouched

    original = {"model": "gpt-4o-mini", "temp": 0.7}
    updated  = add_field(original, "stream", True)
    print(f"  original:  {original}")
    print(f"  updated:   {updated}")


# ── 2. map, filter ────────────────────────────────────────────────────────────
def demo_map_filter():
    print("\n=== 2. map & filter ===")

    texts = [
        "  Hello World  ",
        "",
        "  Python is great  ",
        "   ",
        "AI and ML",
    ]

    # filter out empty/whitespace strings
    non_empty = list(filter(lambda t: t.strip(), texts))
    print(f"  non_empty: {non_empty}")

    # map to clean versions
    cleaned = list(map(str.strip, non_empty))
    print(f"  cleaned:   {cleaned}")

    # Chain: filter → map
    pipeline_result = list(map(str.lower, filter(lambda t: len(t.strip()) > 3, map(str.strip, texts))))
    print(f"  pipeline:  {pipeline_result}")

    # Use comprehensions for readability when chains get deep:
    comprehension = [t.strip().lower() for t in texts if len(t.strip()) > 3]
    print(f"  comprehension (same result): {comprehension}")

    # map with multiple iterables
    prompts    = ["Summarise", "Classify", "Extract"]
    max_tokens = [256, 128, 512]
    configs    = list(map(lambda p, t: {"prompt": p, "max_tokens": t}, prompts, max_tokens))
    print(f"  zipped configs: {configs}")


# ── 3. functools.partial ──────────────────────────────────────────────────────
def demo_partial():
    print("\n=== 3. functools.partial ===")

    def call_llm(model: str, temperature: float, max_tokens: int, prompt: str) -> str:
        return f"[{model}|T={temperature}|max={max_tokens}] {prompt[:40]}"

    # Partially apply — create specialised callables
    mini_precise  = functools.partial(call_llm, "gpt-4o-mini", 0.0, 512)
    mini_creative = functools.partial(call_llm, "gpt-4o-mini", 0.9, 1024)
    big_creative  = functools.partial(call_llm, "gpt-4o",      1.0, 2048)

    prompts = ["Summarise this doc.", "Write a poem about Python.", "Explain recursion."]
    for p in prompts:
        print(f"  precise:  {mini_precise(p)!r}")
    print()

    print(f"  creative: {mini_creative('Write a haiku.')!r}")
    print(f"  big:      {big_creative('Tell a story about a robot.')!r}")

    # partial with keyword args
    def embed(text: str, *, model: str = "text-embedding-3-small", dims: int = 1536) -> str:
        return f"embed({text!r}, model={model!r}, dims={dims})"

    small_embed = functools.partial(embed, model="text-embedding-3-small", dims=512)
    large_embed = functools.partial(embed, model="text-embedding-3-large", dims=3072)
    print(f"\n  {small_embed('hello')!r}")
    print(f"  {large_embed('hello')!r}")

    # partial.__doc__ / __wrapped__
    print(f"  partial func: {small_embed.func.__name__}")
    print(f"  frozen kwargs: {small_embed.keywords}")


# ── 4. functools.reduce & operator module ─────────────────────────────────────
def demo_reduce_operator():
    print("\n=== 4. functools.reduce & operator ===")

    from functools import reduce

    # Sum token counts
    token_counts = [128, 256, 512, 64, 32]
    total = reduce(operator.add, token_counts, 0)
    print(f"  total tokens: {total}")

    # Build a merged config dict (right wins on conflict)
    configs = [
        {"temperature": 0.0, "model": "gpt-4o-mini"},
        {"max_tokens": 512},
        {"temperature": 0.7, "stream": True},
    ]
    merged = reduce(lambda a, b: {**a, **b}, configs)
    print(f"  merged config: {merged}")

    # operator functions (faster than lambdas for built-ins)
    items = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"  max via reduce: {reduce(operator.gt and max, items)}")   # compare approach
    print(f"  max operator:   {max(items)}")

    from operator import attrgetter, itemgetter

    records = [
        {"name": "gpt-4o",      "cost": 0.005, "speed": 0.7},
        {"name": "gpt-4o-mini", "cost": 0.0002,"speed": 0.95},
        {"name": "o1",          "cost": 0.015, "speed": 0.4},
    ]
    by_cost  = sorted(records, key=itemgetter("cost"))
    by_speed = sorted(records, key=itemgetter("speed"), reverse=True)
    print(f"  cheapest: {by_cost[0]['name']!r}")
    print(f"  fastest:  {by_speed[0]['name']!r}")


# ── 5. Function composition pipelines ─────────────────────────────────────────
def demo_composition():
    print("\n=== 5. Function Composition ===")

    def compose(*fns: Callable) -> Callable:
        """Right-to-left composition: compose(f, g, h)(x) == f(g(h(x)))"""
        return functools.reduce(lambda f, g: lambda *a, **kw: f(g(*a, **kw)), fns)

    def pipe(*fns: Callable) -> Callable:
        """Left-to-right composition (pipeline): pipe(h, g, f)(x) == f(g(h(x)))"""
        return functools.reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), fns)

    # Build a text-cleaning pipeline
    clean = pipe(
        str.strip,
        str.lower,
        lambda s: s.replace("\n", " "),
        lambda s: " ".join(s.split()),   # collapse multiple spaces
    )

    raw_texts = [
        "  Hello  World  \n\n",
        "PYTHON  is   GREAT",
        "   AI    and   ML  ",
    ]
    print("  cleaned texts:")
    for t in raw_texts:
        print(f"    {t!r} → {clean(t)!r}")

    # LLM request builder pipeline
    def set_model(cfg: dict) -> dict:    return {**cfg, "model": "gpt-4o-mini"}
    def set_temp(cfg: dict) -> dict:     return {**cfg, "temperature": 0.0}
    def add_system(cfg: dict) -> dict:
        cfg = dict(cfg)
        cfg.setdefault("messages", [])
        cfg["messages"] = [{"role": "system", "content": "You are helpful."}, *cfg["messages"]]
        return cfg

    build_request = pipe(set_model, set_temp, add_system)
    request = build_request({"messages": [{"role": "user", "content": "Hello!"}]})
    print(f"\n  built request: {request}")


# ── 6. itertools ──────────────────────────────────────────────────────────────
def demo_itertools():
    print("\n=== 6. itertools ===")

    # chain — flatten iterables
    batch1 = ["doc_a", "doc_b"]
    batch2 = ["doc_c"]
    batch3 = ["doc_d", "doc_e", "doc_f"]
    all_docs = list(itertools.chain(batch1, batch2, batch3))
    print(f"  chain: {all_docs}")

    # islice — take N from any iterable (works on infinite generators)
    def counter(start=0):
        n = start
        while True:
            yield n
            n += 1

    first_10 = list(itertools.islice(counter(100), 10))
    print(f"  islice: {first_10}")

    # batched (Python 3.12+) / manual chunking
    def chunked(iterable: Iterable[T], size: int) -> Iterable[list[T]]:
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, size))
            if not chunk:
                break
            yield chunk

    docs = [f"doc_{i}" for i in range(11)]
    for chunk in chunked(docs, 4):
        print(f"    batch: {chunk}")

    # combinations / permutations — model evaluation
    models = ["gpt-4o-mini", "claude-haiku", "llama3.2"]
    pairs  = list(itertools.combinations(models, 2))
    print(f"\n  model comparison pairs: {pairs}")

    # groupby — group results by category
    results = [
        {"category": "coding",   "score": 0.92},
        {"category": "coding",   "score": 0.88},
        {"category": "math",     "score": 0.75},
        {"category": "math",     "score": 0.82},
        {"category": "creative", "score": 0.95},
    ]
    results.sort(key=lambda r: r["category"])   # must sort before groupby
    for category, group in itertools.groupby(results, key=lambda r: r["category"]):
        scores = [r["score"] for r in group]
        print(f"  {category}: avg={sum(scores)/len(scores):.2f}")

    # zip_longest — align sequences of different lengths
    prompts = ["P1", "P2", "P3"]
    models2 = ["A", "B"]
    paired = list(itertools.zip_longest(prompts, models2, fillvalue="default"))
    print(f"\n  zip_longest: {paired}")


# ── 7. Practical: functional AI pipeline ─────────────────────────────────────
def demo_ai_pipeline():
    print("\n=== 7. Practical: Functional AI Data Pipeline ===")

    # Sample raw documents (simulate loaded corpus)
    raw_docs = [
        {"id": i, "text": f"  Document {i} about {'AI' if i%2==0 else 'ML'}.  ", "score": round(0.3 + i*0.07, 2)}
        for i in range(10)
    ]

    # Compose a multi-step transformation pipeline
    def clean_text(doc: dict) -> dict:
        return {**doc, "text": doc["text"].strip().lower()}

    def add_word_count(doc: dict) -> dict:
        return {**doc, "word_count": len(doc["text"].split())}

    def is_high_quality(doc: dict) -> bool:
        return doc["score"] >= 0.6 and doc["word_count"] >= 3

    def format_output(doc: dict) -> str:
        return f"[{doc['id']:02d}|score={doc['score']}] {doc['text']}"

    # Pipeline: clean → add metadata → filter → format
    processed = list(map(
        format_output,
        filter(
            is_high_quality,
            map(add_word_count,
                map(clean_text, raw_docs)
            )
        )
    ))

    print(f"  {len(processed)}/{len(raw_docs)} docs passed quality filter:")
    for doc in processed:
        print(f"    {doc}")

    # Aggregate stats using reduce
    scores = [d["score"] for d in raw_docs]
    stats = functools.reduce(
        lambda acc, s: {
            "min":   min(acc["min"], s),
            "max":   max(acc["max"], s),
            "total": acc["total"] + s,
            "count": acc["count"] + 1,
        },
        scores,
        {"min": float("inf"), "max": float("-inf"), "total": 0.0, "count": 0},
    )
    stats["avg"] = stats["total"] / stats["count"]
    print(f"\n  score stats: min={stats['min']:.2f} max={stats['max']:.2f} avg={stats['avg']:.2f}")


if __name__ == "__main__":
    print("Python 03-Advanced — Functional Patterns")
    print("=" * 46)
    demo_pure_functions()
    demo_map_filter()
    demo_partial()
    demo_reduce_operator()
    demo_composition()
    demo_itertools()
    demo_ai_pipeline()
