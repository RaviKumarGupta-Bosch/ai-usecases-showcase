"""
LlamaIndex 02-intermediate — Router & Sub-Question Query Engines
================================================================
Topics covered:
  1. RouterQueryEngine — route queries to the right index
  2. SubQuestionQueryEngine — decompose complex questions
  3. Combining multiple knowledge sources
  4. Custom selectors and routing logic

Run:
  python 01_router_query_engine.py
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SummaryIndex, Document, Settings
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# Two domain corpora
PYTHON_DOCS = [
    Document(text="Python list comprehensions: [expr for item in iterable if condition]. Faster than loops for building lists. Support nesting for matrices.", metadata={"domain": "python"}),
    Document(text="Python dictionaries: key-value pairs, O(1) lookup. Dict comprehensions: {k: v for k, v in items}. update(), get(), setdefault(), pop().", metadata={"domain": "python"}),
    Document(text="Python context managers (with statement) ensure cleanup. __enter__ and __exit__ dunder methods. contextlib.contextmanager decorator for generator-based managers.", metadata={"domain": "python"}),
    Document(text="Python virtual environments: python -m venv env. Activate with source env/bin/activate. pip install -r requirements.txt installs dependencies.", metadata={"domain": "python"}),
    Document(text="Python type hints: List[str], Dict[str, int], Optional[X], Union[X, Y], Tuple[X, ...]. mypy for static type checking. typing module provides generic types.", metadata={"domain": "python"}),
]

ML_DOCS = [
    Document(text="Gradient descent optimises model parameters by moving in the direction of the negative gradient. Learning rate controls step size. Adam, SGD, RMSprop are popular optimisers.", metadata={"domain": "ml"}),
    Document(text="Cross-validation splits data into k folds for reliable evaluation. Stratified k-fold preserves class balance. Leave-one-out CV for small datasets.", metadata={"domain": "ml"}),
    Document(text="Feature engineering: normalisation (0-1), standardisation (z-score), one-hot encoding for categoricals, polynomial features for non-linear relationships.", metadata={"domain": "ml"}),
    Document(text="Ensemble methods: bagging (Random Forest), boosting (XGBoost, LightGBM), stacking. Reduce variance or bias. Combine weak learners into a strong one.", metadata={"domain": "ml"}),
    Document(text="Hyperparameter tuning: grid search, random search, Bayesian optimisation. Early stopping prevents overfitting. Learning rate schedulers adjust LR during training.", metadata={"domain": "ml"}),
]


def build_tools():
    python_index = VectorStoreIndex.from_documents(PYTHON_DOCS)
    ml_index = VectorStoreIndex.from_documents(ML_DOCS)

    python_tool = QueryEngineTool(
        query_engine=python_index.as_query_engine(similarity_top_k=3),
        metadata=ToolMetadata(
            name="python_knowledge",
            description="Answers questions about Python programming: syntax, data structures, modules, best practices.",
        ),
    )

    ml_tool = QueryEngineTool(
        query_engine=ml_index.as_query_engine(similarity_top_k=3),
        metadata=ToolMetadata(
            name="ml_knowledge",
            description="Answers questions about machine learning: algorithms, optimisation, evaluation, feature engineering.",
        ),
    )

    return python_tool, ml_tool


# ── 1. RouterQueryEngine ──────────────────────────────────────────────────────
def demo_router(python_tool, ml_tool):
    print("\n=== 1. RouterQueryEngine ===")

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[python_tool, ml_tool],
        verbose=True,
    )

    queries = [
        "How do I use type hints in Python?",
        "What is gradient descent and how does learning rate affect it?",
        "Explain Python virtual environments",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        response = router.query(q)
        print(f"Answer: {response}")


# ── 2. SubQuestionQueryEngine ─────────────────────────────────────────────────
def demo_sub_question(python_tool, ml_tool):
    print("\n=== 2. SubQuestionQueryEngine ===")

    sub_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[python_tool, ml_tool],
        verbose=True,
    )

    # Complex question spanning both domains
    query = "What Python data structures are useful when building machine learning pipelines, and how do ensemble methods work?"
    print(f"\nComplex Query: {query}\n")
    response = sub_engine.query(query)
    print(f"\nFinal Synthesised Answer:\n{response}")


# ── 3. Multi-selector routing ─────────────────────────────────────────────────
def demo_multi_selector(python_tool, ml_tool):
    print("\n=== 3. LLMMultiSelector (can choose multiple engines) ===")

    router = RouterQueryEngine(
        selector=LLMMultiSelector.from_defaults(),
        query_engine_tools=[python_tool, ml_tool],
        verbose=True,
    )

    query = "Explain feature engineering using Python dict comprehensions for encoding categorical data"
    print(f"\nQuery: {query}")
    response = router.query(query)
    print(f"Answer: {response}")


if __name__ == "__main__":
    print("Building domain indexes and tools...")
    py_tool, ml_tool = build_tools()

    demo_router(py_tool, ml_tool)
    demo_sub_question(py_tool, ml_tool)
    demo_multi_selector(py_tool, ml_tool)
