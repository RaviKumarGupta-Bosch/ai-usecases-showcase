"""
01 - Document Loaders
=====================
Document loaders bring external data (files, web pages, databases)
into LangChain as a list of Document objects.

Topics covered:
  1. TextLoader — load plain text files
  2. WebBaseLoader — scrape web pages
  3. DirectoryLoader — load all files in a folder
  4. Creating Document objects manually
  5. Metadata: adding custom fields to documents
  6. Lazy loading (streaming large corpora)
  7. CSV and JSON loaders
"""

import os
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    CSVLoader,
    JSONLoader,
)

load_dotenv()


# ── Helper: create sample files in a temp directory ──────────────────────────
def create_sample_files(base_dir: Path):
    (base_dir / "docs").mkdir(exist_ok=True)

    (base_dir / "docs" / "ml_intro.txt").write_text(
        "Machine learning (ML) is a subset of artificial intelligence. "
        "ML algorithms learn patterns from data to make predictions or decisions "
        "without being explicitly programmed for each task."
    )
    (base_dir / "docs" / "deep_learning.txt").write_text(
        "Deep learning uses neural networks with many layers (deep architectures). "
        "It excels at tasks like image recognition, natural language processing, "
        "and speech synthesis. GPUs accelerate training significantly."
    )
    (base_dir / "data.csv").write_text(
        "product,category,price\n"
        "Laptop Pro,Electronics,1299.99\n"
        "Running Shoes,Sports,89.99\n"
        "Coffee Maker,Appliances,49.99\n"
    )
    (base_dir / "employees.jsonl").write_text(
        '{"name": "Alice", "role": "Engineer", "department": "AI Research"}\n'
        '{"name": "Bob", "role": "Manager", "department": "Product"}\n'
        '{"name": "Carol", "role": "Designer", "department": "UX"}\n'
    )

    return base_dir


# ── 1. TextLoader ─────────────────────────────────────────────────────────────
def demo_text_loader(base_dir: Path):
    loader = TextLoader(str(base_dir / "docs" / "ml_intro.txt"), encoding="utf-8")
    docs = loader.load()

    print("=== 1. TextLoader ===")
    print(f"Loaded {len(docs)} document(s)")
    for doc in docs:
        print(f"  Source  : {doc.metadata.get('source')}")
        print(f"  Content : {doc.page_content[:80]}...")


# ── 2. WebBaseLoader — scrape a live web page ────────────────────────────────
def demo_web_loader():
    # Loads and cleans the HTML text from any URL
    urls = ["https://en.wikipedia.org/wiki/LangChain"]
    loader = WebBaseLoader(urls)
    docs = loader.load()

    print("\n=== 2. WebBaseLoader ===")
    print(f"Loaded {len(docs)} document(s)")
    for doc in docs:
        print(f"  Source  : {doc.metadata.get('source')}")
        print(f"  Title   : {doc.metadata.get('title', 'N/A')}")
        print(f"  Content : {doc.page_content[:120].strip()}...")


# ── 3. DirectoryLoader — load all files in a folder ─────────────────────────
def demo_directory_loader(base_dir: Path):
    loader = DirectoryLoader(
        str(base_dir / "docs"),
        glob="*.txt",           # only .txt files
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()

    print("\n=== 3. DirectoryLoader ===")
    print(f"Loaded {len(docs)} document(s) from directory")
    for doc in docs:
        print(f"  File: {Path(doc.metadata['source']).name}")
        print(f"  Text: {doc.page_content[:60]}...")


# ── 4. CSVLoader ─────────────────────────────────────────────────────────────
def demo_csv_loader(base_dir: Path):
    loader = CSVLoader(
        file_path=str(base_dir / "data.csv"),
        csv_args={"delimiter": ","},
    )
    docs = loader.load()

    print("\n=== 4. CSVLoader ===")
    print(f"Loaded {len(docs)} row(s)")
    for doc in docs:
        print(f"  Row: {doc.page_content}")


# ── 5. JSONLoader ─────────────────────────────────────────────────────────────
def demo_json_loader(base_dir: Path):
    loader = JSONLoader(
        file_path=str(base_dir / "employees.jsonl"),
        jq_schema=".",          # load each object as-is
        json_lines=True,
    )
    docs = loader.load()

    print("\n=== 5. JSONLoader ===")
    print(f"Loaded {len(docs)} document(s)")
    for doc in docs:
        print(f"  Content : {doc.page_content}")


# ── 6. Manual Document creation ──────────────────────────────────────────────
def demo_manual_documents():
    docs = [
        Document(
            page_content=(
                "The Transformer architecture, introduced in 'Attention is All You Need' (2017), "
                "uses self-attention mechanisms to process sequences in parallel."
            ),
            metadata={
                "source": "research_paper",
                "title": "Attention is All You Need",
                "year": 2017,
                "authors": ["Vaswani et al."],
                "domain": "deep_learning",
            },
        ),
        Document(
            page_content="BERT (Bidirectional Encoder Representations from Transformers) "
                         "pre-trains deep bidirectional representations from unlabelled text.",
            metadata={
                "source": "research_paper",
                "title": "BERT",
                "year": 2018,
                "domain": "nlp",
            },
        ),
    ]

    print("\n=== 6. Manual Document Creation ===")
    for doc in docs:
        print(f"  [{doc.metadata['year']}] {doc.metadata['title']}")
        print(f"  Content: {doc.page_content[:80]}...")


# ── 7. Lazy loading (memory-efficient for large corpora) ─────────────────────
def demo_lazy_loading(base_dir: Path):
    loader = DirectoryLoader(
        str(base_dir / "docs"),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    print("\n=== 7. Lazy Loading ===")
    total_chars = 0
    for doc in loader.lazy_load():          # yields one at a time, no memory spike
        total_chars += len(doc.page_content)
        print(f"  Streamed: {Path(doc.metadata['source']).name} ({len(doc.page_content)} chars)")
    print(f"  Total characters processed: {total_chars}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        create_sample_files(base)

        demo_text_loader(base)
        demo_web_loader()
        demo_directory_loader(base)
        demo_csv_loader(base)
        demo_json_loader(base)
        demo_manual_documents()
        demo_lazy_loading(base)
