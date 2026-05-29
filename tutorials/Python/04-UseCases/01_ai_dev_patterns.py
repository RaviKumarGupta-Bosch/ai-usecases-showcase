"""
Python 04-UseCases — AI Developer Patterns
============================================
Topics covered:
  1. Retry with exponential backoff (for LLM API calls)
  2. Batch processing with progress tracking
  3. Configuration management (env vars + defaults + override hierarchy)
  4. Structured logging for AI applications
  5. Streaming output handler (writing chunks to console / file)
  6. Token counting and cost estimation utilities
  7. Practical: mini CLI tool combining all patterns

Prerequisites:
  pip install python-dotenv

Run:
  python 01_ai_dev_patterns.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, Generator, Iterator, TypeVar
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")


# ── 1. Retry with exponential backoff ────────────────────────────────────────
def demo_retry():
    print("\n=== 1. Retry with Exponential Backoff ===")

    def retry(
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
        jitter: bool = True,
    ):
        """Decorator: retry with exponential backoff + optional jitter."""
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                last_exc: Exception | None = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return fn(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt == max_attempts:
                            break
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                        if jitter:
                            delay *= 0.5 + random.random() * 0.5   # 50-100% of delay
                        print(f"    attempt {attempt} failed ({exc}); retrying in {delay:.2f}s...")
                        time.sleep(delay)
                raise RuntimeError(f"All {max_attempts} attempts failed") from last_exc
            return wrapper
        return decorator

    # Simulate a flaky LLM API call
    call_counter = {"n": 0}

    @retry(max_attempts=4, base_delay=0.1, exceptions=(ConnectionError, TimeoutError))
    def call_llm_api(prompt: str) -> str:
        call_counter["n"] += 1
        if call_counter["n"] < 3:                     # fail first 2 attempts
            raise ConnectionError("rate_limit_exceeded")
        return f"[mock response to: {prompt[:30]!r}]"

    result = call_llm_api("Explain Python decorators in one sentence.")
    print(f"  success after {call_counter['n']} attempt(s): {result}")

    # Retry with context manager (alternative pattern)
    class Retrier:
        def __init__(self, max_attempts: int = 3, base_delay: float = 0.2):
            self.max_attempts = max_attempts
            self.base_delay   = base_delay
            self._attempt     = 0

        def __iter__(self) -> Iterator[int]:
            for attempt in range(1, self.max_attempts + 1):
                self._attempt = attempt
                yield attempt
                if attempt < self.max_attempts:
                    time.sleep(self.base_delay * (2 ** (attempt - 1)))

        @property
        def attempt(self) -> int:
            return self._attempt

    attempts_used = 0
    for attempt in Retrier(max_attempts=3, base_delay=0.05):
        try:
            if attempt < 2:
                raise TimeoutError("timeout")
            attempts_used = attempt
            break
        except TimeoutError:
            print(f"    retrier: attempt {attempt} timed out")
    print(f"  retrier succeeded on attempt {attempts_used}")


# ── 2. Batch processing with progress tracking ────────────────────────────────
def demo_batch_processing():
    print("\n=== 2. Batch Processing with Progress ===")

    def batched(items: list[T], size: int) -> Generator[list[T], None, None]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def progress_bar(current: int, total: int, width: int = 30) -> str:
        pct   = current / total
        filled = int(width * pct)
        bar   = "█" * filled + "░" * (width - filled)
        return f"  [{bar}] {current:>4}/{total} ({pct:5.1%})"

    # Simulate embedding a corpus of documents
    documents = [f"document_{i:04d}: some text about topic {i % 10}" for i in range(47)]
    BATCH_SIZE = 10
    processed  = []
    start_time = time.time()

    for batch_num, batch in enumerate(batched(documents, BATCH_SIZE), start=1):
        # Simulate API call latency
        time.sleep(0.02)
        embeddings = [{"id": doc.split(":")[0], "dims": 1536} for doc in batch]
        processed.extend(embeddings)
        print(progress_bar(len(processed), len(documents)))

    elapsed = time.time() - start_time
    print(f"\n  processed {len(processed)} docs in {elapsed:.2f}s")
    print(f"  throughput: {len(processed)/elapsed:.0f} docs/sec")

    # Batch with error handling — collect results + errors separately
    results: list[dict] = []
    errors:  list[dict] = []

    def embed_batch(batch: list[str]) -> list[dict]:
        if random.random() < 0.15:              # simulate 15% failure rate
            raise RuntimeError("embedding service unavailable")
        return [{"text": t[:20], "vector": [0.1] * 4} for t in batch]

    for batch in batched(documents[:20], 5):
        try:
            results.extend(embed_batch(batch))
        except RuntimeError as e:
            errors.append({"batch": batch[0], "error": str(e)})

    print(f"\n  batch results: {len(results)} ok, {len(errors)} failed batches")


# ── 3. Configuration management ───────────────────────────────────────────────
def demo_config():
    print("\n=== 3. Configuration Management ===")

    @dataclass
    class LLMConfig:
        """Layered configuration: defaults → env vars → explicit overrides."""
        model:       str   = "gpt-4o-mini"
        temperature: float = 0.0
        max_tokens:  int   = 512
        timeout:     int   = 30
        api_key:     str   = field(default="", repr=False)

        def __post_init__(self):
            # Layer 2: env var overrides
            if os.getenv("LLM_MODEL"):
                self.model = os.environ["LLM_MODEL"]
            if os.getenv("LLM_TEMPERATURE"):
                self.temperature = float(os.environ["LLM_TEMPERATURE"])
            if os.getenv("LLM_MAX_TOKENS"):
                self.max_tokens = int(os.environ["LLM_MAX_TOKENS"])
            if os.getenv("OPENAI_API_KEY"):
                self.api_key = os.environ["OPENAI_API_KEY"]

        @classmethod
        def from_dict(cls, overrides: dict) -> "LLMConfig":
            """Layer 3: explicit runtime overrides have highest priority."""
            base = cls()
            for k, v in overrides.items():
                if hasattr(base, k):
                    setattr(base, k, v)
                else:
                    raise ValueError(f"Unknown config key: {k!r}")
            return base

        def validate(self) -> None:
            assert 0.0 <= self.temperature <= 2.0, "temperature must be 0-2"
            assert 1 <= self.max_tokens <= 128_000, "max_tokens out of range"
            assert self.timeout > 0, "timeout must be positive"

    # Default config
    cfg_default = LLMConfig()
    print(f"  default:  model={cfg_default.model!r}, temp={cfg_default.temperature}")

    # Override at runtime
    cfg_custom = LLMConfig.from_dict({"model": "gpt-4o", "temperature": 0.7, "max_tokens": 2048})
    cfg_custom.validate()
    print(f"  custom:   model={cfg_custom.model!r}, temp={cfg_custom.temperature}")

    # Merge multiple config sources (useful for CLI arg overrides)
    def merge_configs(*configs: dict) -> dict:
        result: dict = {}
        for cfg in configs:
            result.update(cfg)
        return result

    base = {"model": "gpt-4o-mini", "temperature": 0.0}
    env  = {"temperature": 0.3}             # from env
    cli  = {"max_tokens": 1024}             # from CLI args
    final = merge_configs(base, env, cli)
    print(f"  merged:   {final}")


# ── 4. Structured logging ─────────────────────────────────────────────────────
def demo_logging():
    print("\n=== 4. Structured Logging for AI Applications ===")

    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger              # avoid duplicate handlers on re-import
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "  %(levelname)-8s [%(name)s] %(message)s"
        ))
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    log = setup_logger("ai_pipeline", logging.DEBUG)

    # Structured log helper — log JSON-serialisable context
    def log_event(logger: logging.Logger, event: str, **kwargs) -> None:
        payload = json.dumps({"event": event, **kwargs}, default=str)
        logger.info(payload)

    # Simulate an LLM call lifecycle
    log_event(log, "llm_call_start", model="gpt-4o-mini", prompt_tokens=128)
    time.sleep(0.01)                                   # simulate latency
    log_event(log, "llm_call_end",   model="gpt-4o-mini",
              completion_tokens=64, latency_ms=312, cost_usd=0.000096)

    log.warning("  rate limit approaching: 85%% of TPM quota used")
    log.debug("  cache miss for prompt hash=abc123")

    # Log decorator — wrap any function with entry/exit logging
    def log_calls(logger: logging.Logger):
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                logger.debug(f"→ {fn.__name__}({args!r}, {kwargs!r})")
                t0 = time.perf_counter()
                result = fn(*args, **kwargs)
                ms = (time.perf_counter() - t0) * 1000
                logger.debug(f"← {fn.__name__} returned in {ms:.1f}ms")
                return result
            return wrapper
        return decorator

    @log_calls(log)
    def embed(text: str) -> list[float]:
        time.sleep(0.005)
        return [0.1, 0.2, 0.3, 0.4]

    embed("hello world")


# ── 5. Streaming output handler ───────────────────────────────────────────────
def demo_streaming():
    print("\n=== 5. Streaming Output Handler ===")

    # Generator that yields token chunks (simulates LLM streaming)
    def mock_stream(text: str, chunk_size: int = 5) -> Generator[str, None, None]:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            yield chunk + " "
            time.sleep(0.01)                      # simulate network delay

    def stream_to_console(stream: Generator[str, None, None], end: str = "\n") -> str:
        """Print each chunk as it arrives, return the full collected text."""
        print("  ", end="", flush=True)
        collected = []
        for chunk in stream:
            print(chunk, end="", flush=True)
            collected.append(chunk)
        print(end, end="")
        return "".join(collected)

    response_text = (
        "Python is a versatile, high-level programming language widely used "
        "in AI, data science, web development, and automation."
    )
    full_text = stream_to_console(mock_stream(response_text, chunk_size=4))
    print(f"  [streamed {len(full_text)} chars]")

    # Stream to file simultaneously
    output_dir = Path(__file__).parent / "_output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "stream_output.txt"

    def stream_to_file(stream: Generator[str, None, None], path: Path) -> str:
        chunks = list(stream)
        text = "".join(chunks)
        path.write_text(text, encoding="utf-8")
        return text

    saved = stream_to_file(mock_stream(response_text), output_file)
    print(f"  saved {len(saved)} chars to {output_file.name}")


# ── 6. Token counting & cost estimation ───────────────────────────────────────
def demo_token_cost():
    print("\n=== 6. Token Counting & Cost Estimation ===")

    # Lightweight token estimator (no tiktoken dependency)
    # Rule of thumb: ~1 token ≈ 4 chars (English text)
    def estimate_tokens(text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    # Model pricing table (USD per 1M tokens, as of mid-2025 approximate values)
    MODEL_PRICING: dict[str, dict[str, float]] = {
        "gpt-4o":             {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini":        {"input": 0.15,  "output": 0.60},
        "gpt-4.1":            {"input": 2.00,  "output": 8.00},
        "gpt-4.1-mini":       {"input": 0.40,  "output": 1.60},
        "o3":                 {"input": 10.00, "output": 40.00},
        "o4-mini":            {"input": 1.10,  "output": 4.40},
        "claude-3-5-haiku":   {"input": 0.80,  "output": 4.00},
        "claude-3-7-sonnet":  {"input": 3.00,  "output": 15.00},
        "llama3.2:3b":        {"input": 0.0,   "output": 0.0},  # local
    }

    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> dict[str, float]:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        input_cost  = (input_tokens  / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "input_cost":    input_cost,
            "output_cost":   output_cost,
            "total_cost":    input_cost + output_cost,
        }

    # Simulate a batch of calls
    calls = [
        {"model": "gpt-4o-mini",       "prompt": "Summarise this article: " + "x" * 800, "output": "y" * 200},
        {"model": "gpt-4o",            "prompt": "Explain quantum computing in detail: " + "z" * 1200, "output": "w" * 400},
        {"model": "claude-3-5-haiku",  "prompt": "Classify this sentiment: " + "a" * 300, "output": "b" * 50},
        {"model": "llama3.2:3b",       "prompt": "Local inference: " + "c" * 600,         "output": "d" * 150},
    ]

    total_cost  = 0.0
    total_input = 0
    total_output = 0

    print(f"  {'Model':<22} {'In tok':>7} {'Out tok':>8} {'Cost':>10}")
    print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*10}")

    for call in calls:
        in_tok  = estimate_tokens(call["prompt"])
        out_tok = estimate_tokens(call["output"])
        cost    = calculate_cost(call["model"], in_tok, out_tok)
        total_cost   += cost["total_cost"]
        total_input  += in_tok
        total_output += out_tok
        cost_str = f"${cost['total_cost']:.6f}" if cost["total_cost"] > 0 else "$0.000000 (local)"
        print(f"  {call['model']:<22} {in_tok:>7} {out_tok:>8} {cost_str:>18}")

    print(f"\n  Total: {total_input} input + {total_output} output tokens = ${total_cost:.6f}")

    # Budget guard utility
    def check_budget(estimated_cost: float, budget_usd: float) -> bool:
        if estimated_cost > budget_usd:
            print(f"  ⚠  estimated ${estimated_cost:.4f} exceeds budget ${budget_usd:.4f}")
            return False
        return True

    check_budget(total_cost, 0.005)


# ── 7. Practical: mini CLI combining all patterns ────────────────────────────
def demo_mini_cli():
    print("\n=== 7. Practical: Mini AI CLI Tool ===")

    # Lightweight config loaded once at startup
    @dataclass
    class AppConfig:
        model:      str   = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
        max_tokens: int   = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "512")))
        budget_usd: float = field(default_factory=lambda: float(os.getenv("BUDGET_USD", "0.10")))
        verbose:    bool  = False

    log = logging.getLogger("mini_cli")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.DEBUG)

    def mock_chat(prompt: str, model: str, max_tokens: int) -> Generator[str, None, None]:
        """Simulate a streaming LLM response."""
        words = f"This is a mock response from {model}. You asked: '{prompt[:40]}'. Here is a helpful answer with multiple words.".split()
        for word in words:
            yield word + " "
            time.sleep(0.003)

    def run_query(cfg: AppConfig, prompt: str) -> dict:
        in_tokens = math.ceil(len(prompt) / 4)
        # Budget pre-check (estimate conservatively)
        estimated = (in_tokens / 1_000_000) * 0.60 + (cfg.max_tokens / 1_000_000) * 2.40
        if estimated > cfg.budget_usd:
            log.warning(f"estimated cost ${estimated:.6f} exceeds budget ${cfg.budget_usd:.4f}")

        log.debug(f"query: model={cfg.model!r}, in_tokens={in_tokens}")
        t0 = time.perf_counter()

        # Stream and collect
        print(f"\n  Assistant: ", end="", flush=True)
        chunks: list[str] = []
        for chunk in mock_chat(prompt, cfg.model, cfg.max_tokens):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()
        full_response = "".join(chunks)

        out_tokens = math.ceil(len(full_response) / 4)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "response":    full_response,
            "in_tokens":   in_tokens,
            "out_tokens":  out_tokens,
            "latency_ms":  round(latency_ms, 1),
        }

    # Simulate a small REPL session
    cfg = AppConfig(verbose=True)
    queries = [
        "What is retrieval-augmented generation?",
        "List 3 Python best practices for async code.",
    ]

    total_spent = 0.0
    for q in queries:
        print(f"\n  User: {q}")
        result = run_query(cfg, q)
        cost = (result["in_tokens"] / 1_000_000 * 0.15) + (result["out_tokens"] / 1_000_000 * 0.60)
        total_spent += cost
        log.debug(f"tokens={result['in_tokens']}+{result['out_tokens']}, "
                  f"latency={result['latency_ms']}ms, cost=${cost:.6f}")

    print(f"\n  Session summary: {len(queries)} queries, total cost=${total_spent:.6f}")


if __name__ == "__main__":
    print("Python 04-UseCases — AI Developer Patterns")
    print("=" * 47)
    demo_retry()
    demo_batch_processing()
    demo_config()
    demo_logging()
    demo_streaming()
    demo_token_cost()
    demo_mini_cli()
