# 🐍 Python — Essentials for AI Development

> A practical Python refresher covering the language features you'll use every day when building AI applications.

---

## 📋 Curriculum

| # | Folder | File | What You'll Learn |
|---|--------|------|-------------------|
| 1 | `01-basics/` | `01_data_types_and_structures.py` | Lists, dicts, sets, tuples, comprehensions |
| 2 | `01-basics/` | `02_functions_and_scope.py` | Functions, *args/**kwargs, closures, lambda |
| 3 | `01-basics/` | `03_oop_basics.py` | Classes, inheritance, dunder methods |
| 4 | `02-intermediate/` | `01_type_hints_and_dataclasses.py` | Type annotations, dataclasses, NamedTuple |
| 5 | `02-intermediate/` | `02_iterators_and_generators.py` | Iterators, generators, `yield`, context managers |
| 6 | `02-intermediate/` | `03_decorators.py` | Decorators, `functools.wraps`, stacking |
| 7 | `03-advanced/` | `01_async_await.py` | asyncio, `async/await`, concurrent tasks |
| 8 | `03-advanced/` | `02_functional_patterns.py` | `map/filter/reduce`, `functools`, `partial` |
| 9 | `03-advanced/` | `03_modules_and_packaging.py` | Modules, packages, `__init__`, imports |
| 10 | `04-UseCases/` | `01_ai_dev_patterns.py` | Retry logic, batching, config, logging for AI apps |

---

## ⚡ Quick Start

```bash
cd tutorials/Python
pip install -r requirements.txt
python 01-basics/01_data_types_and_structures.py
```

No API keys required for most files. Some advanced examples use `python-dotenv`.

---

## 🔑 Key Concepts

| Concept | Why It Matters for AI |
|---------|----------------------|
| List comprehensions | Concise data transformation pipelines |
| Dataclasses | Lightweight structured data (model configs, API responses) |
| Generators | Memory-efficient streaming of large datasets |
| Decorators | Retry, caching, timing wrappers for LLM calls |
| async/await | Non-blocking I/O for concurrent API calls |
| Type hints | Self-documenting code, IDE support, Pydantic integration |
| `functools.partial` | Pre-filling LLM parameters without lambda soup |

---

## 📦 Dependencies

```
python-dotenv>=1.0.0
```

Most files use only the Python standard library.
