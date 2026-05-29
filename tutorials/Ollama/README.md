# 🦙 Ollama — Run LLMs Locally

> Run powerful open-source LLMs entirely on your own machine — no API keys, no cloud, no cost after download.

---

## 📋 Curriculum

| # | Folder | File | What You'll Learn |
|---|--------|------|-------------------|
| 1 | `01-basics/` | `01_model_management.py` | Pull, list, inspect, and delete models |
| 2 | `01-basics/` | `02_generate_and_chat.py` | `generate` vs `chat`, streaming, options |
| 3 | `01-basics/` | `03_langchain_integration.py` | `OllamaLLM`, `OllamaEmbeddings`, LCEL chains |
| 4 | `02-intermediate/` | `01_local_rag.py` | Fully offline RAG — FAISS + Ollama |
| 5 | `02-intermediate/` | `02_structured_output.py` | JSON mode, Pydantic schema enforcement |
| 6 | `03-advanced/` | `01_vision_models.py` | `llava` multimodal — image + text |
| 7 | `04-UseCases/` | `01_private_assistant.py` | Offline personal assistant with memory |

---

## ⚡ Quick Start

### 1. Install Ollama

```bash
# Windows / macOS: download from https://ollama.com/download
# Linux:
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull models

```bash
ollama pull llama3.2            # default chat model (~2 GB)
ollama pull nomic-embed-text    # embedding model (~270 MB)
ollama pull llava               # vision model (~4.7 GB, optional)
```

### 3. Start the server (if not already running)

```bash
ollama serve
```

### 4. Install Python dependencies

```bash
cd tutorials/Ollama
pip install -r requirements.txt
```

### 5. Run any file

```bash
python 01-basics/01_model_management.py
```

> **No `.env` file needed for basics.** Ollama runs on `http://localhost:11434` by default.  
> Copy `.env.example` → `.env` only if you need to customise the host.

---

## 🦙 Model Reference

| Model | Best For | Size | Pull Command |
|-------|----------|------|--------------|
| `llama3.2` | General chat, reasoning | ~2 GB | `ollama pull llama3.2` |
| `llama3.2:1b` | Fast & lightweight | ~1.3 GB | `ollama pull llama3.2:1b` |
| `mistral` | Instruction following | ~4.1 GB | `ollama pull mistral` |
| `codellama` | Code generation | ~3.8 GB | `ollama pull codellama` |
| `llava` | Vision (image + text) | ~4.7 GB | `ollama pull llava` |
| `nomic-embed-text` | Embeddings | ~270 MB | `ollama pull nomic-embed-text` |

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| `ollama.generate()` | Single-turn completion (prompt → response) |
| `ollama.chat()` | Multi-turn messages API (OpenAI-compatible) |
| `stream=True` | Stream tokens as they are generated |
| `format="json"` | Force model to return valid JSON |
| `OllamaLLM` | LangChain wrapper for Ollama |
| `OllamaEmbeddings` | LangChain embeddings using Ollama |
| Local RAG | Full pipeline: embed → FAISS → retrieve → generate, zero cloud |

---

## 📦 Dependencies

```
ollama>=0.3.0
langchain-ollama>=0.2.0
langchain-community>=0.3.0
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
```
