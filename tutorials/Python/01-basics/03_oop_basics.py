"""
Python 01-Basics — OOP Basics
================================
Topics covered:
  1. Defining classes — `__init__`, instance attributes, methods
  2. Class attributes vs instance attributes
  3. Special (dunder) methods: __str__, __repr__, __len__, __eq__
  4. Inheritance and method overriding
  5. Properties — `@property`, `@setter`
  6. Class methods and static methods
  7. Practical pattern: base LLM client class hierarchy

Run:
  python 03_oop_basics.py
"""

from dotenv import load_dotenv
from dataclasses import dataclass
load_dotenv()


# ── 1. Defining a class ───────────────────────────────────────────────────────
def demo_basic_class():
    print("\n=== 1. Basic Class Definition ===")

    class Message:
        """Represents a single chat message."""

        def __init__(self, role: str, content: str):
            self.role    = role
            self.content = content

        def to_dict(self) -> dict:
            return {"role": self.role, "content": self.content}

        def word_count(self) -> int:
            return len(self.content.split())

    msg = Message("user", "What is the capital of France?")
    print(f"  role:       {msg.role}")
    print(f"  content:    {msg.content!r}")
    print(f"  word_count: {msg.word_count()}")
    print(f"  to_dict:    {msg.to_dict()}")


# ── 2. Class vs instance attributes ───────────────────────────────────────────
def demo_class_vs_instance():
    print("\n=== 2. Class vs Instance Attributes ===")

    class LLMClient:
        # Class attribute — shared across ALL instances
        api_version: str = "2024-11-01"
        _instance_count: int = 0

        def __init__(self, model: str, temperature: float = 0.0):
            # Instance attributes — unique per object
            self.model       = model
            self.temperature = temperature
            LLMClient._instance_count += 1

        @classmethod
        def get_count(cls) -> int:
            return cls._instance_count

    c1 = LLMClient("gpt-4o-mini")
    c2 = LLMClient("claude-3-haiku", temperature=0.5)
    print(f"  c1.model={c1.model!r}  c2.model={c2.model!r}")
    print(f"  shared api_version: {LLMClient.api_version}")
    print(f"  instances created:  {LLMClient.get_count()}")

    # Class attribute accessible via instance, but instance attr shadows it
    c1.api_version = "2025-01-01"    # sets on instance only
    print(f"  c1.api_version (instance override): {c1.api_version}")
    print(f"  c2.api_version (class):             {c2.api_version}")
    print(f"  LLMClient.api_version (class):      {LLMClient.api_version}")


# ── 3. Dunder / magic methods ──────────────────────────────────────────────────
def demo_dunder_methods():
    print("\n=== 3. Dunder Methods ===")

    class Conversation:
        def __init__(self, system_prompt: str = ""):
            self.messages: list[dict] = []
            if system_prompt:
                self.messages.append({"role": "system", "content": system_prompt})

        def add(self, role: str, content: str) -> "Conversation":
            self.messages.append({"role": role, "content": content})
            return self   # allow chaining

        def __len__(self) -> int:
            return len(self.messages)

        def __str__(self) -> str:
            lines = [f"[{m['role']}] {m['content']}" for m in self.messages]
            return "\n".join(lines)

        def __repr__(self) -> str:
            return f"Conversation(messages={len(self)})"

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Conversation):
                return NotImplemented
            return self.messages == other.messages

        def __contains__(self, text: str) -> bool:
            return any(text.lower() in m["content"].lower() for m in self.messages)

    conv = (
        Conversation("You are helpful.")
        .add("user", "What is Python?")
        .add("assistant", "Python is a high-level programming language.")
    )

    print(f"  len(conv) = {len(conv)}")
    print(f"  repr:       {conv!r}")
    print(f"  'Python' in conv: {'Python' in conv}")
    print(f"  str:\n{conv}")


# ── 4. Inheritance ─────────────────────────────────────────────────────────────
def demo_inheritance():
    print("\n=== 4. Inheritance ===")

    class BaseTool:
        """Abstract-like base for agent tools."""

        def __init__(self, name: str, description: str):
            self.name        = name
            self.description = description

        def run(self, input_text: str) -> str:
            raise NotImplementedError(f"{self.name}.run() must be implemented")

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(name={self.name!r})"

    class CalculatorTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="calculator",
                description="Evaluates simple arithmetic expressions.",
            )

        def run(self, input_text: str) -> str:
            try:
                allowed = set("0123456789+-*/(). ")
                if not all(c in allowed for c in input_text):
                    return "Error: invalid characters"
                result = eval(input_text, {"__builtins__": {}})  # noqa: S307
                return str(result)
            except Exception as e:
                return f"Error: {e}"

    class SearchTool(BaseTool):
        def __init__(self, max_results: int = 5):
            super().__init__(
                name="search",
                description="Searches for information online.",
            )
            self.max_results = max_results

        def run(self, input_text: str) -> str:
            # Mock response
            return f"[Search results for '{input_text}' — top {self.max_results} results]"

    tools = [CalculatorTool(), SearchTool(max_results=3)]
    for tool in tools:
        print(f"  {tool!r}")
        print(f"    run: {tool.run('2 + 2 * 3')}")

    # isinstance / issubclass
    print(f"  isinstance(tools[0], BaseTool): {isinstance(tools[0], BaseTool)}")
    print(f"  issubclass(SearchTool, BaseTool): {issubclass(SearchTool, BaseTool)}")


# ── 5. Properties ─────────────────────────────────────────────────────────────
def demo_properties():
    print("\n=== 5. Properties (@property / @setter) ===")

    class ModelConfig:
        def __init__(self, model: str, temperature: float = 0.0):
            self.model = model
            self._temperature = temperature   # private storage

        @property
        def temperature(self) -> float:
            return self._temperature

        @temperature.setter
        def temperature(self, value: float):
            if not 0.0 <= value <= 2.0:
                raise ValueError(f"temperature must be 0–2, got {value}")
            self._temperature = value

        @property
        def is_deterministic(self) -> bool:
            """Read-only derived property."""
            return self._temperature == 0.0

    cfg = ModelConfig("gpt-4o", temperature=0.0)
    print(f"  temperature:      {cfg.temperature}")
    print(f"  is_deterministic: {cfg.is_deterministic}")

    cfg.temperature = 0.7
    print(f"  after set 0.7:    {cfg.temperature}, deterministic={cfg.is_deterministic}")

    try:
        cfg.temperature = 3.0
    except ValueError as e:
        print(f"  validation error: {e}")


# ── 6. Class methods & static methods ─────────────────────────────────────────
def demo_class_and_static():
    print("\n=== 6. Class Methods & Static Methods ===")

    class TokenCounter:
        _total_tokens: int = 0

        def __init__(self, name: str):
            self.name          = name
            self.local_tokens  = 0

        def add(self, n: int):
            self.local_tokens        += n
            TokenCounter._total_tokens += n

        # classmethod — receives the class, not the instance
        @classmethod
        def get_total(cls) -> int:
            return cls._total_tokens

        @classmethod
        def reset_total(cls):
            cls._total_tokens = 0

        # staticmethod — no access to class or instance; pure utility
        @staticmethod
        def estimate_cost(tokens: int, price_per_1k: float = 0.002) -> float:
            return (tokens / 1000) * price_per_1k

    a = TokenCounter("agentA"); a.add(500)
    b = TokenCounter("agentB"); b.add(300)

    print(f"  agentA local: {a.local_tokens}")
    print(f"  agentB local: {b.local_tokens}")
    print(f"  class total:  {TokenCounter.get_total()}")
    print(f"  cost for 800: ${TokenCounter.estimate_cost(800):.4f}")


# ── 7. Practical: LLM client hierarchy ────────────────────────────────────────
def demo_llm_hierarchy():
    print("\n=== 7. LLM Client Hierarchy (practical pattern) ===")

    class BaseChat:
        """Minimal interface any LLM client must implement."""

        def __init__(self, model: str, temperature: float = 0.0):
            self.model       = model
            self.temperature = temperature
            self.call_count  = 0

        def complete(self, prompt: str) -> str:
            raise NotImplementedError

        def _record_call(self):
            self.call_count += 1

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(model={self.model!r}, calls={self.call_count})"

    class MockOpenAIChat(BaseChat):
        def complete(self, prompt: str) -> str:
            self._record_call()
            return f"[OpenAI/{self.model}] Reply to: '{prompt[:40]}...'"

    class MockAnthropicChat(BaseChat):
        MAX_TOKENS = 4096

        def complete(self, prompt: str) -> str:
            self._record_call()
            return f"[Anthropic/{self.model}] Reply to: '{prompt[:40]}...'"

    clients: list[BaseChat] = [
        MockOpenAIChat("gpt-4o-mini"),
        MockAnthropicChat("claude-3-haiku"),
    ]

    prompt = "Explain Python decorators in one sentence."
    for client in clients:
        resp = client.complete(prompt)
        print(f"  {client}")
        print(f"    → {resp}")


if __name__ == "__main__":
    print("Python 01-Basics — OOP Basics")
    print("=" * 38)
    demo_basic_class()
    demo_class_vs_instance()
    demo_dunder_methods()
    demo_inheritance()
    demo_properties()
    demo_class_and_static()
    demo_llm_hierarchy()
