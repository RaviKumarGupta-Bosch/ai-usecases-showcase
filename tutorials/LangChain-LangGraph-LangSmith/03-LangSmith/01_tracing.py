"""
01 - Tracing with LangSmith
============================
LangSmith automatically captures every LLM call, chain step, and tool
invocation when LANGCHAIN_TRACING_V2=true is set.

Topics covered:
  1. Environment setup for automatic tracing
  2. Auto-tracing LangChain chains (zero extra code)
  3. @traceable decorator for custom Python functions
  4. Nested @traceable calls (tree view in LangSmith UI)
  5. traceable with custom name, tags, and metadata
  6. Manual run feedback via langsmith.Client
  7. Tracing LangGraph workflows
  8. Disabling tracing selectively
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# LangSmith is activated by these environment variables (set in .env):
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=<your-langsmith-api-key>
#   LANGCHAIN_PROJECT=langchain-langgraph-tutorial  (optional, defaults to "default")

from langsmith import traceable, Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ── 1. Auto-tracing — LangChain chains traced automatically ───────────────────
def demo_auto_tracing():
    """
    All LangChain components (LLMs, prompts, chains) are traced automatically
    when LANGCHAIN_TRACING_V2=true. No extra code needed.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer in one sentence."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    print("=== 1. Auto-Tracing (LangChain Chain) ===")
    result = chain.invoke({"question": "What is LangSmith used for?"})
    print(f"  Answer: {result}")
    print("  [Check LangSmith UI → your project → latest trace]")


# ── 2. @traceable — wrap any Python function ──────────────────────────────────
@traceable(name="fetch_product_info", tags=["retrieval", "mock"])
def fetch_product_info(product_id: str) -> dict:
    """Simulate a database lookup. Traced as a custom span in LangSmith."""
    products = {
        "P001": {"name": "LangChain Pro",    "price": 99.0,  "category": "software"},
        "P002": {"name": "LangGraph Studio", "price": 149.0, "category": "software"},
        "P003": {"name": "LangSmith Teams",  "price": 299.0, "category": "platform"},
    }
    return products.get(product_id, {"name": "Unknown", "price": 0.0, "category": "unknown"})


@traceable(name="generate_product_description", tags=["llm", "marketing"])
def generate_description(product: dict) -> str:
    """Generate a product description. Traced as a child span."""
    prompt = ChatPromptTemplate.from_template(
        "Write a 2-sentence marketing description for: {name} (${price})"
    )
    chain = prompt | llm_creative | StrOutputParser()
    return chain.invoke({"name": product["name"], "price": product["price"]})


@traceable(name="product_pipeline", tags=["e2e"], metadata={"version": "1.0"})
def product_pipeline(product_id: str) -> dict:
    """
    Top-level traceable function. In LangSmith, you'll see:
    product_pipeline
      ├── fetch_product_info
      └── generate_product_description
           └── ChatOpenAI
    """
    product = fetch_product_info(product_id)
    if product["category"] == "unknown":
        return {"error": f"Product {product_id} not found"}
    description = generate_description(product)
    return {
        "product_id": product_id,
        "product":    product,
        "description": description,
    }


def demo_traceable_decorator():
    print("\n=== 2. @traceable Decorator ===")
    result = product_pipeline("P001")
    print(f"  Product     : {result['product']['name']}")
    print(f"  Description : {result['description']}")
    print("  [Check LangSmith → 'product_pipeline' trace with nested spans]")


# ── 3. Nested @traceable — deep call trees ────────────────────────────────────
@traceable(name="step_preprocess")
def preprocess(text: str) -> str:
    return text.strip().lower()


@traceable(name="step_classify")
def classify(text: str) -> str:
    chain = (
        ChatPromptTemplate.from_template("Classify this text into one category (tech/business/other): {text}")
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"text": text}).strip()


@traceable(name="step_respond")
def generate_response(text: str, category: str) -> str:
    chain = (
        ChatPromptTemplate.from_template(
            "You are a {category} expert. Answer this in one sentence: {text}"
        )
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"text": text, "category": category})


@traceable(name="full_query_pipeline")
def full_pipeline(user_query: str) -> dict:
    """
    LangSmith tree:
    full_query_pipeline
      ├── step_preprocess
      ├── step_classify
      │    └── ChatOpenAI
      └── step_respond
           └── ChatOpenAI
    """
    clean   = preprocess(user_query)
    category = classify(clean)
    response = generate_response(clean, category)
    return {"query": user_query, "category": category, "response": response}


def demo_nested_tracing():
    print("\n=== 3. Nested @traceable (Deep Call Tree) ===")
    queries = [
        "How do transformers work in deep learning?",
        "What is the ROI of AI investments?",
    ]
    for q in queries:
        result = full_pipeline(q)
        print(f"  Query    : {result['query']}")
        print(f"  Category : {result['category']}")
        print(f"  Response : {result['response']}")
        print()


# ── 4. Adding metadata and tags programmatically ──────────────────────────────
@traceable(
    name="sentiment_analysis",
    tags=["sentiment", "classification"],
    metadata={"model": "gpt-4o-mini", "use_case": "customer_feedback"},
)
def analyse_sentiment(text: str, customer_id: str) -> dict:
    """Trace includes metadata visible in LangSmith for filtering."""
    from pydantic import BaseModel, Field
    class SentimentResult(BaseModel):
        sentiment: str = Field(description="positive/negative/neutral")
        confidence: float = Field(ge=0, le=1)
        reason: str

    result = llm.with_structured_output(SentimentResult).invoke(
        f"Analyse the sentiment of this customer feedback: '{text}'"
    )
    return {
        "customer_id": customer_id,
        "text":        text,
        "sentiment":   result.sentiment,
        "confidence":  result.confidence,
        "reason":      result.reason,
    }


def demo_metadata_tags():
    print("\n=== 4. Metadata & Tags on Traces ===")
    feedbacks = [
        ("C001", "The product is amazing! Saved me hours every day."),
        ("C002", "Not impressed. Crashes frequently and support is slow."),
        ("C003", "It works okay, nothing special but gets the job done."),
    ]
    for customer_id, text in feedbacks:
        result = analyse_sentiment(text, customer_id)
        print(f"  [{result['customer_id']}] {result['sentiment']:10s} ({result['confidence']:.0%}) — {result['reason'][:60]}")


# ── 5. LangSmith Client — run feedback ────────────────────────────────────────
def demo_langsmith_client():
    """
    Use the LangSmith Client to:
    - List recent runs
    - Add thumbs up/down feedback
    """
    print("\n=== 5. LangSmith Client ===")

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("  [Skip] LANGCHAIN_API_KEY not set — LangSmith client unavailable")
        return

    client = Client()

    # Check connectivity
    try:
        projects = list(client.list_projects())
        project_names = [p.name for p in projects[:5]]
        print(f"  Connected to LangSmith.")
        print(f"  Projects: {project_names}")
    except Exception as e:
        print(f"  [Error] Could not connect to LangSmith: {e}")
        return

    # Run a traced chain and capture the run_id for feedback
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    prompt  = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
    chain   = prompt | llm | StrOutputParser()

    with client.trace(name="demo_client_run", project_name=project) as run:
        result = chain.invoke({"topic": "vector embeddings"})
        run_id = run.id

    print(f"  Run ID    : {run_id}")
    print(f"  Output    : {result}")

    # Add thumbs-up feedback
    try:
        client.create_feedback(
            run_id=run_id,
            key="user_feedback",
            score=1,
            comment="Great explanation!",
        )
        print("  Feedback submitted: thumbs up ✓")
    except Exception as e:
        print(f"  [Feedback error] {e}")


# ── 6. Tracing a LangGraph workflow ───────────────────────────────────────────
def demo_langgraph_tracing():
    """LangGraph workflows are automatically traced — each node appears as a span."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage
    import operator

    class State(TypedDict):
        messages: list[str]
        results:  list[str]

    @traceable(name="node_extract")
    def extract_node(state: State) -> dict:
        input_text = state["messages"][0] if state["messages"] else ""
        chain = (
            ChatPromptTemplate.from_template("Extract 3 key facts from: {text}")
            | llm
            | StrOutputParser()
        )
        return {"results": [chain.invoke({"text": input_text})]}

    @traceable(name="node_format")
    def format_node(state: State) -> dict:
        raw = state["results"][0] if state["results"] else ""
        chain = (
            ChatPromptTemplate.from_template("Format these facts as bullet points:\n{facts}")
            | llm
            | StrOutputParser()
        )
        return {"results": [chain.invoke({"facts": raw})]}

    graph = StateGraph(State)
    graph.add_node("extract", extract_node)
    graph.add_node("format",  format_node)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "format")
    graph.add_edge("format",  END)
    app = graph.compile()

    print("\n=== 6. LangGraph Workflow Tracing ===")
    result = app.invoke({
        "messages": ["LangSmith provides tracing, evaluation, and dataset management for LLM apps."],
        "results": [],
    })
    print(f"  Formatted output:\n{result['results'][-1]}")
    print("  [Each node appears as a separate span in LangSmith UI]")


if __name__ == "__main__":
    demo_auto_tracing()
    demo_traceable_decorator()
    demo_nested_tracing()
    demo_metadata_tags()
    demo_langsmith_client()
    demo_langgraph_tracing()
