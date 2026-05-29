# 🤖 AI Use-Cases Showcase: Manufacturing & Finance

> Practical, hands-on AI solutions for real-world industries — built to learn, explore, and extend.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![Ollama/LLaMA](https://img.shields.io/badge/LLaMA-Ollama-orange.svg)](https://ollama.ai/)

---

## 📖 What Is This?

This repository is a **beginner-friendly showcase** of practical AI applications across two major industries:

- 🏭 **Manufacturing** — predictive maintenance, quality control, demand forecasting, anomaly detection
- 💰 **Finance** — fraud detection, sentiment analysis, loan risk assessment, portfolio advising

Each use-case is **self-contained** and runs with either:
- **OpenAI GPT-4o-mini** (cloud, requires API key)
- **LLaMA 3 via [Ollama](https://ollama.ai/)** (free, runs locally on your machine)

No complicated infrastructure needed. Just Python, a few packages, and curiosity.

---

## 📁 Repository Structure

```
ai-usecases-showcase/
│
├── 📁 manufacturing/               # AI use-cases for manufacturing
│   ├── README.md
│   ├── 01-predictive-maintenance/  # Predict machine failures from sensor data
│   ├── 02-quality-control/         # Detect product defects with AI
│   ├── 03-demand-forecasting/      # Forecast production demand
│   └── 04-anomaly-detection/       # Spot unusual production patterns
│
├── 📁 finance/                     # AI use-cases for finance
│   ├── README.md
│   ├── 01-fraud-detection/         # Flag suspicious transactions
│   ├── 02-sentiment-analysis/      # Analyze financial news sentiment
│   ├── 03-loan-risk-assessment/    # Score credit applications
│   └── 04-portfolio-advisor/       # AI-powered investment recommendations
│
├── 📁 dotnet-future/               # Future .NET / C# implementations (roadmap)
│
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/RaviKumarGupta-Bosch/ai-usecases-showcase.git
cd ai-usecases-showcase
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Choose your AI provider

**Option A: OpenAI (cloud)**
```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# macOS / Linux
export OPENAI_API_KEY=your-api-key-here
```
Get a free API key at [platform.openai.com](https://platform.openai.com).

**Option B: LLaMA via Ollama (local, free)**
```bash
# 1. Install Ollama from https://ollama.ai/
# 2. Pull the LLaMA 3 model
ollama pull llama3

# 3. Tell the scripts to use Ollama
set AI_PROVIDER=ollama   # Windows
export AI_PROVIDER=ollama  # macOS/Linux
```

### 4. Run any use-case

```bash
cd manufacturing/01-predictive-maintenance
python predictive_maintenance.py
```

---

## 🏭 Manufacturing Use-Cases

| # | Use-Case | Description | Key Concepts |
|---|----------|-------------|--------------|
| 01 | [Predictive Maintenance](./manufacturing/01-predictive-maintenance/) | Analyze sensor data to predict machine failures | AI + IoT data |
| 02 | [Quality Control](./manufacturing/02-quality-control/) | Classify products as pass/fail using AI judgment | AI inspection |
| 03 | [Demand Forecasting](./manufacturing/03-demand-forecasting/) | Forecast future production demand with AI insights | Time-series + AI |
| 04 | [Anomaly Detection](./manufacturing/04-anomaly-detection/) | Detect unusual patterns in production line data | Statistical + AI |

---

## 💰 Finance Use-Cases

| # | Use-Case | Description | Key Concepts |
|---|----------|-------------|--------------|
| 01 | [Fraud Detection](./finance/01-fraud-detection/) | Flag potentially fraudulent financial transactions | Classification + AI |
| 02 | [Sentiment Analysis](./finance/02-sentiment-analysis/) | Score financial news as bullish, bearish, or neutral | NLP + AI |
| 03 | [Loan Risk Assessment](./finance/03-loan-risk-assessment/) | Rate credit applications with AI reasoning | Risk scoring + AI |
| 04 | [Portfolio Advisor](./finance/04-portfolio-advisor/) | Get AI-powered investment rebalancing advice | Portfolio + AI |

---

## 🔧 How to Add a New Use-Case

See [CONTRIBUTING.md](./CONTRIBUTING.md) for step-by-step instructions on:
- Structuring your use-case folder
- Writing beginner-friendly Python scripts
- Supporting both OpenAI and Ollama backends
- Adding sample data and a README

---

## 🔮 Future: .NET / C# Version

A C# implementation of these use-cases is planned using:
- **Microsoft Semantic Kernel** (AI orchestration)
- **Azure OpenAI Service**
- **ML.NET** (for local model inference)

See [dotnet-future/README.md](./dotnet-future/README.md) for the full roadmap.

---

## 📄 License

MIT License — free to use, modify, and share.

---

## ⭐ Show Your Support

If this repository helps you learn or inspires a project, please give it a ⭐ — it really helps!

Built with ❤️ to showcase practical, hands-on AI skills in real-world industry domains.
