# 🤖 AI Use-Cases Showcase: Manufacturing & Finance

> Practical, hands-on AI solutions for real-world industries — built to learn, explore, and extend.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![.NET 8](https://img.shields.io/badge/.NET-8.0-purple.svg)](https://dotnet.microsoft.com/download/dotnet/8.0)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![Ollama/LLaMA](https://img.shields.io/badge/LLaMA-Ollama-orange.svg)](https://ollama.ai/)

---

## 📖 What Is This?

This repository is a **beginner-friendly showcase** of practical AI applications across two major industries:

- 🏭 **Manufacturing** — predictive maintenance, quality control, demand forecasting, anomaly detection
- 💰 **Finance** — fraud detection, sentiment analysis, loan risk assessment, portfolio advising

Every use-case is implemented in **both Python and C# (.NET 8)**, each supporting two AI backends:

| Backend | Python | .NET |
|---------|--------|------|
| OpenAI GPT-4o-mini (cloud) | `OPENAI_API_KEY` | `AZURE_OPENAI_KEY` + `AZURE_OPENAI_ENDPOINT` |
| LLaMA 3 via Ollama (local) | `AI_PROVIDER=ollama` | `AI_PROVIDER=ollama` |

---

## 📁 Repository Structure

```
ai-usecases-showcase/
│
├── 📁 python/                         # Python implementations
│   ├── Shared/                      ← ai_service.py, data_loader.py, json_helper.py
│   ├── Manufacturing/               ← 4 use-case folders
│   └── Finance/                     ← 4 use-case folders
│
├── 📁 dotnet/                         # C# / .NET 8 implementations
│   ├── AIUsecasesShowcase.sln
│   ├── Shared/Core/                 ← IAIService, OpenAIService, OllamaService
│   ├── Shared/Data/                 ← CsvLoader, JsonLoader
│   ├── Manufacturing/               ← 4 use-case projects
│   └── Finance/                     ← 4 use-case projects
│
├── 📁 tutorials/                      # Python deep-dive tutorials
│   ├── LangChain-LangGraph-LangSmith/  ← chains, graphs, observability
│   ├── AutoGen/                        ← multi-agent conversations
│   ├── CrewAI/                         ← role-based agent crews
│   ├── RAG/                            ← retrieval-augmented generation
│   ├── MCP/                            ← Model Context Protocol
│   ├── VectorDatabases/                ← FAISS, Chroma, embeddings
│   ├── LlamaIndex/                     ← data framework & agents
│   ├── PromptEngineering/              ← prompting techniques
│   ├── Ollama/                         ← local LLMs, offline RAG, vision
│   └── Python/                         ← Python fundamentals for AI developers
│
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## 🚀 Quick Start — Python

```bash
git clone https://github.com/RaviKumarGupta-Bosch/ai-usecases-showcase.git
cd ai-usecases-showcase
pip install -r requirements.txt

# OpenAI
set OPENAI_API_KEY=your-key    # Windows
export OPENAI_API_KEY=your-key # macOS/Linux

# OR local LLaMA
ollama pull llama3
set AI_PROVIDER=ollama

cd python/Manufacturing/01-PredictiveMaintenance
python main.py
```

---

## 🚀 Quick Start — .NET 8

```bash
cd dotnet
dotnet restore AIUsecasesShowcase.sln

# Azure OpenAI
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_KEY=your-key

# OR local LLaMA
set AI_PROVIDER=ollama

cd Manufacturing/01-PredictiveMaintenance
dotnet run
```

---

## 🏭 Manufacturing Use-Cases

| # | Python | .NET | Description |
|---|--------|------|-------------|
| 01 | [01-PredictiveMaintenance](./python/Manufacturing/01-PredictiveMaintenance/) | [PredictiveMaintenance](./dotnet/Manufacturing/01-PredictiveMaintenance/) | Analyse sensor data to predict machine failures |
| 02 | [02-QualityControl](./python/Manufacturing/02-QualityControl/) | [QualityControl](./dotnet/Manufacturing/02-QualityControl/) | Classify production batches as PASS / FAIL |
| 03 | [03-DemandForecasting](./python/Manufacturing/03-DemandForecasting/) | [DemandForecasting](./dotnet/Manufacturing/03-DemandForecasting/) | Forecast demand from 12-month sales history |
| 04 | [04-AnomalyDetection](./python/Manufacturing/04-AnomalyDetection/) | [AnomalyDetection](./dotnet/Manufacturing/04-AnomalyDetection/) | Detect anomalies on production lines |

---

## 💰 Finance Use-Cases

| # | Python | .NET | Description |
|---|--------|------|-------------|
| 01 | [01-FraudDetection](./python/Finance/01-FraudDetection/) | [FraudDetection](./dotnet/Finance/01-FraudDetection/) | Flag suspicious financial transactions |
| 02 | [02-SentimentAnalysis](./python/Finance/02-SentimentAnalysis/) | [SentimentAnalysis](./dotnet/Finance/02-SentimentAnalysis/) | Score financial news as BULLISH / BEARISH / NEUTRAL |
| 03 | [03-LoanRiskAssessment](./python/Finance/03-LoanRiskAssessment/) | [LoanRiskAssessment](./dotnet/Finance/03-LoanRiskAssessment/) | Rate credit applications with AI reasoning |
| 04 | [04-PortfolioAdvisor](./python/Finance/04-PortfolioAdvisor/) | [PortfolioAdvisor](./dotnet/Finance/04-PortfolioAdvisor/) | AI-powered investment rebalancing recommendations |

---

## 🔧 How to Add a New Use-Case

See [CONTRIBUTING.md](./CONTRIBUTING.md) for step-by-step instructions.

---

## 📚 Python Tutorials

Step-by-step, runnable Python tutorials covering the AI ecosystem — each folder has its own `README.md`, `requirements.txt`, and `.env.example`.

Every tutorial follows a consistent four-tier progression:

```
<Tutorial>/
├── README.md              # curriculum & quick-start
├── requirements.txt       # pinned dependencies
├── .env.example           # required environment variables
├── 01-basics/             # core concepts, first working examples
├── 02-intermediate/       # real-world patterns, integrations
├── 03-advanced/           # production patterns, performance, architecture
└── 04-UseCases/           # end-to-end domain applications
```

### Tutorial Overview

| Tutorial | Topics Covered |
|----------|----------------|
| [LangChain-LangGraph-LangSmith](./tutorials/LangChain-LangGraph-LangSmith/) | LCEL chains, memory, tools, LangGraph stateful agents, streaming, LangSmith observability & evaluation |
| [AutoGen](./tutorials/AutoGen/) | Two-agent chat, group chat, tool use, termination conditions, nested chats, async conversations, multi-stage pipelines |
| [CrewAI](./tutorials/CrewAI/) | Agents & tasks, crew processes, custom tools, task context & Pydantic outputs, Flows with `@start`/`@listen`/`@router`, state management |
| [RAG](./tutorials/RAG/) | Naive RAG, advanced retrieval, hybrid search, HyDE, CRAG, conversational RAG |
| [MCP](./tutorials/MCP/) | Protocol concepts, FastMCP server, client connections, resources & URI templates, prompt templates, multi-server architecture, tool namespacing |
| [VectorDatabases](./tutorials/VectorDatabases/) | Embeddings primer, FAISS, Chroma, metadata filtering, MMR, re-ranking, hybrid BM25+dense search |
| [LlamaIndex](./tutorials/LlamaIndex/) | Document indexing, router query engine, ReAct agents, multi-document Q&A |
| [PromptEngineering](./tutorials/PromptEngineering/) | Zero/few-shot, chain-of-thought, role prompting, structured output, prompt injection defence, prompt library |
| [Ollama](./tutorials/Ollama/) | Model management, generate & chat APIs, LangChain integration, local RAG, structured output, vision models, private AI assistant |
| [Python](./tutorials/Python/) | Data types & structures, functions & scope, OOP, type hints & dataclasses, generators, decorators, async/await, functional patterns, packaging, AI dev patterns |

---

## 📄 License

MIT License — free to use, modify, and share.

---

## 👤 Author

**Ravi Gupta**
AI Developer & Enthusiast | Full Stack .NET & Angular Developer | Tech Lead

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ravi%20Gupta-0077B5?logo=linkedin)](https://www.linkedin.com/in/ravi-gupta-28b40b36)

---

## ⭐ Show Your Support

If this repository helps you learn or inspires a project, please give it a ⭐!

Built with ❤️ to showcase practical, hands-on AI skills across Python and .NET.
