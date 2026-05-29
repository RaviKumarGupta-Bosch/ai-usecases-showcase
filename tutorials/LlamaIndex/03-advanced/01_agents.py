"""
LlamaIndex 03-advanced — ReAct Agent with Tools
================================================
Topics covered:
  1. LlamaIndex ReAct agent basics
  2. Function tools (Python functions as tools)
  3. QueryEngineTools (using indexes as tools)
  4. Multi-step reasoning with tool calls
  5. Streaming agent responses

Run:
  python 01_agents.py
"""

import os
import math
import requests
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# Knowledge base for one of the query engine tools
ML_DOCS = [
    Document(text="Supervised learning uses labelled training data. Common algorithms: linear regression, decision trees, SVM, neural networks. Evaluation metrics: accuracy, F1, AUC-ROC."),
    Document(text="Unsupervised learning finds patterns without labels. K-means clusters data points. PCA reduces dimensionality. Autoencoders learn compressed representations."),
    Document(text="Reinforcement learning trains agents via rewards. Q-learning, policy gradients, and PPO are common algorithms. Used in games, robotics, and recommendation systems."),
    Document(text="Neural networks have layers of connected neurons. Activation functions (ReLU, sigmoid) introduce non-linearity. Backpropagation computes gradients for weight updates."),
    Document(text="Transformers use self-attention to process sequences. BERT is pretrained for NLP classification. GPT uses decoder-only architecture for text generation."),
    Document(text="Overfitting occurs when the model memorises training data. Remedies: dropout, L1/L2 regularisation, early stopping, more training data, data augmentation."),
]


# ── 1. Function tools ─────────────────────────────────────────────────────────
def demo_function_tools():
    print("\n=== 1. Function Tools ===")

    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression. Supports sqrt, sin, cos, log, pi, e, etc."""
        try:
            safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            return str(eval(expression, {"__builtins__": {}}, safe_ns))
        except Exception as e:
            return f"Error: {e}"

    def search_wikipedia(topic: str) -> str:
        """Search Wikipedia and return a brief summary about any topic."""
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.json().get("extract", "No summary found.")[:500]
            return f"No Wikipedia article found for '{topic}'"
        except Exception as e:
            return f"Error: {e}"

    tools = [
        FunctionTool.from_defaults(fn=calculate,         name="calculator"),
        FunctionTool.from_defaults(fn=search_wikipedia,  name="wikipedia"),
    ]

    agent = ReActAgent.from_tools(tools, verbose=True, max_iterations=10)

    queries = [
        "What is the square root of 2 multiplied by pi?",
        "Who invented Python programming language and when?",
    ]

    for q in queries:
        print(f"\n--- Query: {q} ---")
        response = agent.chat(q)
        print(f"Final Answer: {response}")


# ── 2. QueryEngineTool (using an index as a tool) ─────────────────────────────
def demo_query_engine_tool():
    print("\n=== 2. QueryEngineTool (Index as Agent Tool) ===")

    # Build a vector index over ML docs
    ml_index = VectorStoreIndex.from_documents(ML_DOCS)
    ml_query_engine = ml_index.as_query_engine(similarity_top_k=3)

    ml_tool = QueryEngineTool(
        query_engine=ml_query_engine,
        metadata=ToolMetadata(
            name="ml_knowledge_base",
            description="Answers questions about machine learning algorithms, neural networks, transformers, and AI concepts.",
        ),
    )

    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            return str(eval(expression, {"__builtins__": {}}, safe_ns))
        except Exception as e:
            return f"Error: {e}"

    tools = [FunctionTool.from_defaults(fn=calculate), ml_tool]
    agent = ReActAgent.from_tools(tools, verbose=True, max_iterations=15)

    query = "Explain how neural networks avoid overfitting, and calculate log base 2 of 256"
    print(f"\nQuery: {query}")
    response = agent.chat(query)
    print(f"\nFinal Answer: {response}")


# ── 3. Multi-turn agent conversation ─────────────────────────────────────────
def demo_multi_turn():
    print("\n=== 3. Multi-Turn Agent Conversation ===")

    def count_words(text: str) -> str:
        """Count the number of words in a text."""
        return str(len(text.split()))

    def to_uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

    def reverse_text(text: str) -> str:
        """Reverse a string character by character."""
        return text[::-1]

    tools = [
        FunctionTool.from_defaults(fn=count_words),
        FunctionTool.from_defaults(fn=to_uppercase),
        FunctionTool.from_defaults(fn=reverse_text),
    ]

    agent = ReActAgent.from_tools(tools, verbose=False, max_iterations=10)

    turns = [
        "How many words are in 'The quick brown fox jumps over the lazy dog'?",
        "Now convert that sentence to uppercase",
        "Reverse the word 'LlamaIndex'",
    ]

    for turn in turns:
        print(f"\nUser: {turn}")
        response = agent.chat(turn)
        print(f"Agent: {response}")


if __name__ == "__main__":
    demo_function_tools()
    demo_query_engine_tool()
    demo_multi_turn()
