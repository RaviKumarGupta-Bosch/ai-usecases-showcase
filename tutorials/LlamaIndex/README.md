# LlamaIndex Tutorial

LlamaIndex (formerly GPT Index) is a data framework for connecting LLMs to external data sources.
It specialises in indexing, retrieval, and query workflows over documents and knowledge bases.

## Curriculum

```
01-basics/
  01_simple_index.py          — VectorStoreIndex from in-memory documents
  02_query_engine.py          — QueryEngine, RetrieverQueryEngine, response modes

02-intermediate/
  01_router_query_engine.py   — Route queries to different indexes by type
  02_sub_question_engine.py   — Break complex questions into sub-questions

03-advanced/
  01_agents.py                — LlamaIndex ReAct agent with tools
  02_chat_engine.py           — Conversational chat engine with memory

04-UseCases/
  01_multi_document_qa.py     — Q&A over multiple document sources
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python 01-basics/01_simple_index.py
```

## Key LlamaIndex Concepts

| Concept | Description |
|---------|-------------|
| **Node** | A chunk of text with metadata |
| **Index** | A structure for organising nodes for retrieval |
| **QueryEngine** | End-to-end query pipeline (retrieve + synthesise) |
| **RetrieverQueryEngine** | Query engine with a custom retriever |
| **Router** | Routes queries to the appropriate index/engine |
| **Agent** | LLM that can use tools (including query engines) |

## Prerequisites
- Python 3.10+
- `OPENAI_API_KEY` in `.env`
