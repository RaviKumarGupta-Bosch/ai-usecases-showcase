"""
02 - Text Splitters
===================
Large documents must be split into chunks before embedding and retrieval.
The right split strategy dramatically impacts RAG quality.

Topics covered:
  1. RecursiveCharacterTextSplitter  — best default choice
  2. CharacterTextSplitter           — single separator
  3. TokenTextSplitter               — split by token count
  4. MarkdownHeaderTextSplitter      — structure-aware for Markdown
  5. Language-aware code splitter    — keeps functions intact
  6. chunk_size / chunk_overlap tuning
  7. Preserving metadata across splits
"""

from pathlib import Path
import tempfile
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

load_dotenv()

# Long sample text for demonstrations
LONG_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to 
the natural intelligence displayed by animals including humans. AI research has been 
defined as the field of study of intelligent agents, which refers to any system that 
perceives its environment and takes actions that maximize its chance of achieving its goals.

Machine learning (ML) is a type of AI that allows software applications to become more 
accurate at predicting outcomes without being explicitly programmed to do so. ML algorithms 
use historical data as input to predict new output values. Supervised learning, unsupervised 
learning, and reinforcement learning are the three main categories.

Deep learning is a subset of machine learning that uses artificial neural networks with 
multiple layers—hence the term "deep"—to progressively extract higher-level features from 
raw input data. For example, in image processing, lower layers may identify edges, while 
higher layers may identify concepts relevant to humans, such as digits, letters, or faces.

Natural language processing (NLP) is a subfield of linguistics, computer science, and AI 
concerned with the interactions between computers and human language. It is used to apply 
algorithms to identify and extract natural language rules so that unstructured language data 
is converted into a form that computers can understand. Key tasks include sentiment analysis, 
named entity recognition, machine translation, and question answering.

Reinforcement learning (RL) is an area of machine learning concerned with how intelligent 
agents ought to take actions in an environment in order to maximize the notion of cumulative 
reward. RL is one of three basic machine learning paradigms, alongside supervised learning 
and unsupervised learning. Its application areas include robotics, game playing, and 
autonomous vehicle navigation.
""".strip()

MARKDOWN_TEXT = """
# Introduction to LangChain

LangChain is a framework for developing applications powered by language models.

## Core Concepts

### Chains
Chains are sequences of calls to LLMs or other utilities. The LCEL (LangChain Expression 
Language) uses the pipe `|` operator to compose chains declaratively.

### Retrievers
Retrievers return relevant documents given a query. They power RAG applications by 
fetching context before the LLM generates an answer.

## Installation

```bash
pip install langchain langchain-openai
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello, world!")
```

## Advanced Topics

### Agents
Agents use LLMs to decide which actions to take. They combine reasoning with tool use.

### Memory
Memory components store and retrieve information across conversations.
"""

PYTHON_CODE = '''
def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers."""
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


class Calculator:
    """A simple stateful calculator."""

    def __init__(self):
        self.result = 0
        self.history: list[str] = []

    def add(self, value: float) -> "Calculator":
        self.history.append(f"+ {value}")
        self.result += value
        return self

    def subtract(self, value: float) -> "Calculator":
        self.history.append(f"- {value}")
        self.result -= value
        return self

    def reset(self) -> "Calculator":
        self.history.clear()
        self.result = 0
        return self

    def get_result(self) -> float:
        return self.result
'''


# ── 1. RecursiveCharacterTextSplitter ────────────────────────────────────────
def demo_recursive_splitter():
    """
    Best default. Tries to split on paragraphs, then sentences, then words.
    Respects natural language structure as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
        add_start_index=True,   # metadata: char offset of each chunk
    )

    chunks = splitter.create_documents(
        texts=[LONG_TEXT],
        metadatas=[{"source": "ai_overview.txt", "domain": "AI"}],
    )

    print("=== 1. RecursiveCharacterTextSplitter ===")
    print(f"Input: {len(LONG_TEXT)} chars → {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\nChunk {i} (chars {chunk.metadata['start_index']}–"
              f"{chunk.metadata['start_index'] + len(chunk.page_content)}):")
        print(f"  {chunk.page_content[:100]}...")


# ── 2. CharacterTextSplitter — split on a single separator ───────────────────
def demo_character_splitter():
    splitter = CharacterTextSplitter(
        separator="\n\n",       # split on double newlines (paragraphs)
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(LONG_TEXT)

    print("\n=== 2. CharacterTextSplitter (paragraph splits) ===")
    print(f"{len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2], 1):
        print(f"\nChunk {i}: {chunk[:80]}...")


# ── 3. TokenTextSplitter — split by token count ──────────────────────────────
def demo_token_splitter():
    """
    Ensures each chunk doesn't exceed a token budget.
    Requires `tiktoken` (included in requirements.txt).
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=100,     # max tokens per chunk
        chunk_overlap=20,
    )
    chunks = splitter.split_text(LONG_TEXT)

    print("\n=== 3. Token-Based Splitter ===")
    print(f"{len(chunks)} chunks (each ≤ 100 tokens)")
    for i, chunk in enumerate(chunks[:2], 1):
        print(f"  Chunk {i}: {chunk[:80]}...")


# ── 4. MarkdownHeaderTextSplitter — structure-aware ──────────────────────────
def demo_markdown_splitter():
    """Splits on markdown headers and injects header hierarchy into metadata."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",  "h1"),
            ("##", "h2"),
            ("###","h3"),
        ],
        strip_headers=False,
    )
    chunks = splitter.split_text(MARKDOWN_TEXT)

    print("\n=== 4. MarkdownHeaderTextSplitter ===")
    print(f"{len(chunks)} section(s)")
    for chunk in chunks:
        print(f"\n  Headers : {chunk.metadata}")
        print(f"  Content : {chunk.page_content[:100]}...")


# ── 5. Code splitter — language-aware ────────────────────────────────────────
def demo_code_splitter():
    """
    Keeps Python functions/classes intact instead of cutting through them.
    Supported: Python, JS, TS, Java, C, C++, Go, Ruby, Rust, …
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=300,
        chunk_overlap=30,
    )
    chunks = splitter.create_documents([PYTHON_CODE])

    print("\n=== 5. Code Splitter (Python) ===")
    print(f"{len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:\n{chunk.page_content}")


# ── 6. Chunk size / overlap tuning comparison ────────────────────────────────
def demo_tuning_comparison():
    """
    Show the effect of different chunk_size and chunk_overlap settings.
    Small chunks → more precise retrieval, more noise.
    Large chunks → more context, less precise matching.
    """
    configs = [
        {"chunk_size": 200, "chunk_overlap": 0},
        {"chunk_size": 200, "chunk_overlap": 50},
        {"chunk_size": 500, "chunk_overlap": 100},
    ]

    print("\n=== 6. Chunk Size / Overlap Comparison ===")
    for cfg in configs:
        splitter = RecursiveCharacterTextSplitter(**cfg)
        chunks = splitter.split_text(LONG_TEXT)
        print(f"  size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']} → {len(chunks)} chunks")


# ── 7. Split and preserve metadata ───────────────────────────────────────────
def demo_metadata_preservation():
    """
    Metadata from the source Document propagates to all child chunks.
    """
    source_docs = [
        Document(
            page_content=LONG_TEXT,
            metadata={
                "source": "ai_intro.txt",
                "author": "Wikipedia",
                "version": "2024-01",
                "language": "en",
            },
        )
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(source_docs)

    print("\n=== 7. Metadata Preservation ===")
    print(f"Source: 1 doc → {len(chunks)} chunks, all with inherited metadata")
    for chunk in chunks[:2]:
        print(f"  Metadata : {chunk.metadata}")
        print(f"  Content  : {chunk.page_content[:60]}...\n")


if __name__ == "__main__":
    demo_recursive_splitter()
    demo_character_splitter()
    demo_token_splitter()
    demo_markdown_splitter()
    demo_code_splitter()
    demo_tuning_comparison()
    demo_metadata_preservation()
