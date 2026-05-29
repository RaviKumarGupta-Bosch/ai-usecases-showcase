"""
Python 03-Advanced — Modules & Packaging
==========================================
Topics covered:
  1. Import system — absolute vs relative imports
  2. __name__ == "__main__" guard
  3. sys.path and module discovery
  4. __all__ — controlling public API
  5. Package structure and __init__.py
  6. importlib — dynamic imports
  7. Practical: plugin registry pattern for AI tools

Note: This file is self-contained and demonstrates concepts inline.
      For real package structure, see the annotated layout below.

Run:
  python 03_modules_and_packaging.py
"""

from __future__ import annotations

import sys
import importlib
import importlib.util
import os
import types
from dotenv import load_dotenv

load_dotenv()


# ── 1. Import system ──────────────────────────────────────────────────────────
def demo_imports():
    print("\n=== 1. Import System ===")

    # Standard absolute import
    import math
    import json
    from pathlib import Path
    from collections import defaultdict

    print(f"  math.pi = {math.pi:.4f}")
    print(f"  json.dumps({{'k': 1}}) = {json.dumps({'k': 1})!r}")
    print(f"  Path.cwd(): {Path.cwd().name}")

    # Import with alias — common in AI/data work
    import os.path as osp
    print(f"  os.path.sep: {osp.sep!r}")

    # from X import * (avoid in production — pollutes namespace)
    # from math import *  ← bad
    from math import sqrt, log, exp   # explicit is better
    print(f"  sqrt(2) = {sqrt(2):.4f}")

    # Re-checking what's loaded
    math_location = sys.modules["math"].__spec__.origin if hasattr(sys.modules["math"], "__spec__") else "built-in"
    print(f"  math location: {math_location}")


# ── 2. __name__ guard ─────────────────────────────────────────────────────────
def demo_name_guard():
    print("\n=== 2. __name__ == '__main__' ===")

    # When a file is run directly, __name__ == "__main__"
    # When imported, __name__ == the module's dotted name

    print(f"  __name__:    {__name__!r}")
    print(f"  __file__:    {os.path.basename(__file__)}")

    # The guard prevents test/demo code from running on import
    # Pattern used throughout this tutorial:
    #
    #   if __name__ == "__main__":
    #       demo_something()
    #
    # This means:
    # - `python mymodule.py` → demos run
    # - `import mymodule`    → demos do NOT run, only definitions load

    print("  (if you import this file, the demos at the bottom will NOT run)")


# ── 3. sys.path ───────────────────────────────────────────────────────────────
def demo_sys_path():
    print("\n=== 3. sys.path & Module Discovery ===")

    print(f"  sys.path has {len(sys.path)} entries")
    for p in sys.path[:5]:
        print(f"    {p!r}")
    print("    ...")

    # Add a directory to sys.path at runtime
    custom_dir = os.path.join(os.path.dirname(__file__), "_helpers")
    if custom_dir not in sys.path:
        sys.path.insert(0, custom_dir)
    print(f"\n  added to sys.path: {custom_dir!r}")
    print(f"  (In practice, use proper packaging instead)")

    # Show where a module comes from
    import json
    print(f"  json module file: {json.__file__}")


# ── 4. __all__ ────────────────────────────────────────────────────────────────
def demo_all():
    print("\n=== 4. __all__ — Controlling Public API ===")

    # Simulate what a module with __all__ looks like
    # In a real module file you'd have:
    #
    #   __all__ = ["PublicClass", "public_function"]
    #
    #   class PublicClass: ...
    #   def public_function(): ...
    #   def _private_helper(): ...   # excluded
    #
    # When someone does `from mymodule import *`, only __all__ items are imported.

    # Create a fake module in-memory to demonstrate
    fake_module = types.ModuleType("fake_tools")
    fake_module.__all__ = ["search", "calculate"]     # only these are "public"

    def search(query: str) -> str: return f"results for {query}"
    def calculate(expr: str) -> str: return f"result of {expr}"
    def _internal() -> str: return "hidden"

    fake_module.search    = search
    fake_module.calculate = calculate
    fake_module._internal = _internal

    public_api = [name for name in fake_module.__all__]
    print(f"  public API (__all__): {public_api}")
    print(f"  _internal accessible: {fake_module._internal() !r}  (but not in __all__)")


# ── 5. Package structure (annotated) ──────────────────────────────────────────
def demo_package_structure():
    print("\n=== 5. Package Structure ===")

    layout = """
  Typical AI project package layout:

  my_ai_project/
  ├── pyproject.toml          # PEP 517 build config (replaces setup.py)
  ├── README.md
  ├── .env
  ├── src/
  │   └── my_ai_project/
  │       ├── __init__.py     # marks directory as package; exports public API
  │       ├── agents/
  │       │   ├── __init__.py
  │       │   ├── base.py
  │       │   └── research.py
  │       ├── tools/
  │       │   ├── __init__.py
  │       │   ├── search.py
  │       │   └── calculator.py
  │       └── utils/
  │           ├── __init__.py
  │           └── tokens.py
  └── tests/
      ├── test_agents.py
      └── test_tools.py

  pyproject.toml example:
  [project]
  name = "my-ai-project"
  version = "0.1.0"
  dependencies = ["openai>=1.0.0", "python-dotenv>=1.0.0"]

  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"
"""
    print(layout)

    # __init__.py pattern: re-export the public surface
    init_example = '''
  # src/my_ai_project/__init__.py
  from .agents.research import ResearchAgent
  from .tools.search import SearchTool
  from .utils.tokens import count_tokens

  __all__ = ["ResearchAgent", "SearchTool", "count_tokens"]
  __version__ = "0.1.0"
'''
    print("  __init__.py example:", init_example)


# ── 6. importlib — dynamic imports ───────────────────────────────────────────
def demo_importlib():
    print("\n=== 6. importlib — Dynamic Imports ===")

    # Import a module by name at runtime (useful for plugin systems)
    module_name = "json"
    mod = importlib.import_module(module_name)
    result = mod.dumps({"dynamic": True})
    print(f"  dynamic import of {module_name!r}: {result!r}")

    # Check if a module is available before using it
    def is_available(package: str) -> bool:
        spec = importlib.util.find_spec(package)
        return spec is not None

    packages_to_check = ["json", "pathlib", "openai", "langchain", "nonexistent_pkg"]
    for pkg in packages_to_check:
        status = "✓" if is_available(pkg) else "✗"
        print(f"  {status} {pkg}")

    # Reload a module (useful during development)
    import json as _json
    importlib.reload(_json)
    print("  json module reloaded")


# ── 7. Practical: plugin registry ────────────────────────────────────────────
def demo_plugin_registry():
    print("\n=== 7. Practical: Plugin Registry for AI Tools ===")

    # A registry maps string names to tool classes at runtime
    # This is the pattern used by LangChain, AutoGen, CrewAI, etc.

    _TOOL_REGISTRY: dict[str, type] = {}

    def register_tool(name: str):
        """Decorator that registers a tool class under a given name."""
        def decorator(cls):
            _TOOL_REGISTRY[name] = cls
            return cls
        return decorator

    def get_tool(name: str, **kwargs):
        if name not in _TOOL_REGISTRY:
            raise KeyError(f"Tool {name!r} not found. Available: {list(_TOOL_REGISTRY)}")
        return _TOOL_REGISTRY[name](**kwargs)

    # Register tools using the decorator
    @register_tool("search")
    class SearchTool:
        def __init__(self, max_results: int = 5):
            self.max_results = max_results
        def run(self, query: str) -> str:
            return f"[Search] Top {self.max_results} results for: '{query}'"

    @register_tool("calculator")
    class CalculatorTool:
        def __init__(self):
            pass
        def run(self, expr: str) -> str:
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expr):
                return "Error: invalid expression"
            return str(eval(expr, {"__builtins__": {}}))   # noqa: S307

    @register_tool("summariser")
    class SummariserTool:
        def __init__(self, max_length: int = 100):
            self.max_length = max_length
        def run(self, text: str) -> str:
            return f"[Summary of {len(text)} chars] {text[:self.max_length]}..."

    # Use the registry
    print(f"  registered tools: {list(_TOOL_REGISTRY)}")

    tool_calls = [
        ("search",     {"max_results": 3}, "Python asyncio tutorial"),
        ("calculator", {},                  "2 ** 10 + 42"),
        ("summariser", {"max_length": 30},  "Python is a high-level, general-purpose programming language."),
    ]

    for tool_name, kwargs, input_text in tool_calls:
        tool = get_tool(tool_name, **kwargs)
        result = tool.run(input_text)
        print(f"  [{tool_name}] {result!r}")

    # Dynamic tool loading by name list (e.g., from config)
    requested_tools = ["search", "calculator"]
    agent_tools = [get_tool(name) for name in requested_tools]
    print(f"\n  agent loaded {len(agent_tools)} tools: {[t.__class__.__name__ for t in agent_tools]}")


if __name__ == "__main__":
    print("Python 03-Advanced — Modules & Packaging")
    print("=" * 46)
    demo_imports()
    demo_name_guard()
    demo_sys_path()
    demo_all()
    demo_package_structure()
    demo_importlib()
    demo_plugin_registry()
