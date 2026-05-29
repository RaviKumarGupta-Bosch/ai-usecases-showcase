"""
Python 02-Intermediate — Type Hints & Dataclasses
===================================================
Topics covered:
  1. Basic type annotations — variables, functions, return types
  2. Generic types: list, dict, tuple, Optional, Union
  3. Type aliases and NewType
  4. dataclasses — @dataclass, field(), post_init
  5. NamedTuple — immutable typed records
  6. Protocol — structural typing (duck-typing with type safety)
  7. Practical patterns: typed configs, API response models

Run:
  python 01_type_hints_and_dataclasses.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Any, Protocol, runtime_checkable
from dotenv import load_dotenv

load_dotenv()


# ── 1. Basic type annotations ─────────────────────────────────────────────────
def demo_basic_annotations():
    print("\n=== 1. Basic Type Annotations ===")

    # Variable annotations
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = True

    # Function signature
    def truncate(text: str, max_chars: int = 100) -> str:
        return text[:max_chars] + "..." if len(text) > max_chars else text

    long_text = "Python is a versatile, high-level programming language known for its readability."
    print(f"  truncated: {truncate(long_text, 40)!r}")

    # Union type (Python 3.10+: use X | Y directly)
    def parse_tokens(value: str | int) -> int:
        if isinstance(value, str):
            return int(value.replace(",", ""))
        return value

    print(f"  parse_tokens('1,024') = {parse_tokens('1,024')}")
    print(f"  parse_tokens(512)     = {parse_tokens(512)}")

    # Optional (= Union[X, None])
    def find_model(name: str, catalog: list[str]) -> Optional[str]:
        return next((m for m in catalog if name.lower() in m.lower()), None)

    catalog = ["gpt-4o-mini", "gpt-4o", "o1-preview"]
    print(f"  find 'mini': {find_model('mini', catalog)!r}")
    print(f"  find 'gemini': {find_model('gemini', catalog)!r}")


# ── 2. Generic container types ────────────────────────────────────────────────
def demo_generic_types():
    print("\n=== 2. Generic Container Types ===")

    # list[T], dict[K, V], tuple[T, ...]
    def batch_embed(texts: list[str]) -> list[list[float]]:
        """Mock: return fake embeddings of dim=4."""
        return [[float(ord(c)) for c in t[:4]] for t in texts]

    embeddings: list[list[float]] = batch_embed(["hello", "world"])
    print(f"  embeddings: {embeddings}")

    # dict with typed keys/values
    def build_headers(api_key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    hdrs = build_headers("sk-xxx", extra={"X-Request-ID": "abc123"})
    print(f"  headers: {hdrs}")

    # tuple with fixed structure
    TokenUsage = tuple[int, int, int]   # (prompt, completion, total)

    def mock_api_call(prompt: str) -> tuple[str, TokenUsage]:
        p_tokens = len(prompt.split())
        c_tokens = p_tokens * 2
        return ("Mock response.", (p_tokens, c_tokens, p_tokens + c_tokens))

    response, usage = mock_api_call("What is Python?")
    print(f"  response={response!r}, usage={usage}")

    # Any — escape hatch (avoid when possible)
    def log_value(name: str, value: Any) -> None:
        print(f"  {name}: {value!r} ({type(value).__name__})")

    log_value("temp", 0.7)
    log_value("messages", [{"role": "user"}])


# ── 3. Type aliases & NewType ──────────────────────────────────────────────────
def demo_type_aliases():
    print("\n=== 3. Type Aliases & NewType ===")
    from typing import NewType

    # Simple alias — documentation-only, no runtime enforcement
    MessageList = list[dict[str, str]]
    Embedding = list[float]

    messages: MessageList = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": "Hello!"},
    ]
    print(f"  MessageList with {len(messages)} messages")

    # NewType — creates a distinct type at type-checker level
    ModelName = NewType("ModelName", str)
    APIKey    = NewType("APIKey", str)

    def call_model(model: ModelName, key: APIKey) -> str:
        return f"Calling {model} with key {key[:6]}..."

    result = call_model(ModelName("gpt-4o-mini"), APIKey("sk-abc123xyz"))
    print(f"  {result}")


# ── 4. dataclasses ────────────────────────────────────────────────────────────
def demo_dataclasses():
    print("\n=== 4. @dataclass ===")

    @dataclass
    class LLMConfig:
        model:       str
        temperature: float = 0.0
        max_tokens:  int   = 1024
        stream:      bool  = False
        tags:        list[str] = field(default_factory=list)   # mutable default!

        def __post_init__(self):
            if not 0.0 <= self.temperature <= 2.0:
                raise ValueError(f"temperature {self.temperature} out of [0, 2]")
            self.model = self.model.strip().lower()

        @property
        def is_streaming(self) -> bool:
            return self.stream

    cfg1 = LLMConfig("gpt-4o-mini", temperature=0.7, tags=["prod"])
    cfg2 = LLMConfig("gpt-4o-mini", temperature=0.7, tags=["prod"])
    cfg3 = LLMConfig("gpt-4o",      temperature=0.0)

    print(f"  cfg1: {cfg1}")
    print(f"  cfg1 == cfg2: {cfg1 == cfg2}")   # dataclass auto-generates __eq__
    print(f"  cfg1 == cfg3: {cfg1 == cfg3}")
    print(f"  as dict: {asdict(cfg1)}")

    # frozen=True → immutable (hashable, usable as dict key)
    @dataclass(frozen=True)
    class ModelID:
        provider: str
        name:     str
        version:  str = "latest"

        def __str__(self) -> str:
            return f"{self.provider}/{self.name}:{self.version}"

    mid = ModelID("openai", "gpt-4o", "2024-11-20")
    print(f"  ModelID: {mid}")
    lookup = {mid: 0.005}   # hashable, can be dict key
    print(f"  price lookup: ${lookup[mid]}")


# ── 5. NamedTuple ─────────────────────────────────────────────────────────────
def demo_namedtuple():
    print("\n=== 5. NamedTuple ===")

    from typing import NamedTuple

    class TokenUsage(NamedTuple):
        prompt:     int
        completion: int

        @property
        def total(self) -> int:
            return self.prompt + self.completion

        def cost(self, price_per_1k: float = 0.002) -> float:
            return (self.total / 1000) * price_per_1k

    usage = TokenUsage(prompt=512, completion=256)
    print(f"  usage:    {usage}")
    print(f"  total:    {usage.total}")
    print(f"  cost:     ${usage.cost():.4f}")

    # Unpack like a tuple
    prompt_t, completion_t = usage
    print(f"  unpack:   prompt={prompt_t}, completion={completion_t}")

    # _asdict() — dict representation
    print(f"  as dict:  {usage._asdict()}")


# ── 6. Protocol — structural typing ───────────────────────────────────────────
def demo_protocol():
    print("\n=== 6. Protocol (Structural Typing) ===")

    @runtime_checkable
    class Embedder(Protocol):
        """Any class with an embed() method satisfies this protocol."""
        def embed(self, text: str) -> list[float]: ...

    class OpenAIEmbedder:
        def embed(self, text: str) -> list[float]:
            return [float(i) for i in range(4)]   # mock

    class OllamaEmbedder:
        def embed(self, text: str) -> list[float]:
            return [float(len(text) - i) for i in range(4)]   # mock

    def embed_documents(docs: list[str], embedder: Embedder) -> list[list[float]]:
        return [embedder.embed(d) for d in docs]

    for embedder in [OpenAIEmbedder(), OllamaEmbedder()]:
        vecs = embed_documents(["hello", "world"], embedder)
        name = type(embedder).__name__
        print(f"  {name}: {vecs}")
        print(f"    isinstance check: {isinstance(embedder, Embedder)}")


# ── 7. Practical: typed API response ─────────────────────────────────────────
def demo_typed_response():
    print("\n=== 7. Practical: Typed API Response Model ===")

    from typing import NamedTuple

    @dataclass
    class Choice:
        index:         int
        message_role:  str
        message_content: str
        finish_reason: str = "stop"

    @dataclass
    class ChatResponse:
        id:      str
        model:   str
        choices: list[Choice]
        usage:   dict[str, int]

        @property
        def text(self) -> str:
            """Return the first choice's content."""
            return self.choices[0].message_content if self.choices else ""

        @property
        def total_tokens(self) -> int:
            return self.usage.get("total_tokens", 0)

        @classmethod
        def from_dict(cls, data: dict) -> "ChatResponse":
            choices = [
                Choice(
                    index=c["index"],
                    message_role=c["message"]["role"],
                    message_content=c["message"]["content"],
                    finish_reason=c.get("finish_reason", "stop"),
                )
                for c in data.get("choices", [])
            ]
            return cls(
                id=data["id"],
                model=data["model"],
                choices=choices,
                usage=data.get("usage", {}),
            )

    raw = {
        "id":    "chatcmpl-abc123",
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello! How can I help?"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }

    resp = ChatResponse.from_dict(raw)
    print(f"  id:            {resp.id}")
    print(f"  text:          {resp.text!r}")
    print(f"  total_tokens:  {resp.total_tokens}")


if __name__ == "__main__":
    print("Python 02-Intermediate — Type Hints & Dataclasses")
    print("=" * 52)
    demo_basic_annotations()
    demo_generic_types()
    demo_type_aliases()
    demo_dataclasses()
    demo_namedtuple()
    demo_protocol()
    demo_typed_response()
