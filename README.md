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
├── 📁 manufacturing/               # Python AI use-cases for manufacturing
├── 📁 finance/                     # Python AI use-cases for finance
│
├── 📁 dotnet/                      # C# / .NET 8 implementations
│   ├── AIUsecasesShowcase.sln
│   ├── Shared/Core/                # IAIService, OpenAIService, OllamaService
│   ├── Shared/Data/                # CsvLoader, JsonLoader
│   ├── Manufacturing/              # 4 use-case projects
│   └── Finance/                   # 4 use-case projects
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

cd manufacturing/01-predictive-maintenance
python predictive_maintenance.py
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
| 01 | [predictive-maintenance](./manufacturing/01-predictive-maintenance/) | [PredictiveMaintenance](./dotnet/Manufacturing/01-PredictiveMaintenance/) | Analyse sensor data to predict machine failures |
| 02 | [quality-control](./manufacturing/02-quality-control/) | [QualityControl](./dotnet/Manufacturing/02-QualityControl/) | Classify production batches as PASS / FAIL |
| 03 | [demand-forecasting](./manufacturing/03-demand-forecasting/) | [DemandForecasting](./dotnet/Manufacturing/03-DemandForecasting/) | Forecast demand from 12-month sales history |
| 04 | [anomaly-detection](./manufacturing/04-anomaly-detection/) | [AnomalyDetection](./dotnet/Manufacturing/04-AnomalyDetection/) | Detect anomalies on production lines |

---

## 💰 Finance Use-Cases

| # | Python | .NET | Description |
|---|--------|------|-------------|
| 01 | [fraud-detection](./finance/01-fraud-detection/) | [FraudDetection](./dotnet/Finance/01-FraudDetection/) | Flag suspicious financial transactions |
| 02 | [sentiment-analysis](./finance/02-sentiment-analysis/) | [SentimentAnalysis](./dotnet/Finance/02-SentimentAnalysis/) | Score financial news as BULLISH / BEARISH / NEUTRAL |
| 03 | [loan-risk-assessment](./finance/03-loan-risk-assessment/) | [LoanRiskAssessment](./dotnet/Finance/03-LoanRiskAssessment/) | Rate credit applications with AI reasoning |
| 04 | [portfolio-advisor](./finance/04-portfolio-advisor/) | [PortfolioAdvisor](./dotnet/Finance/04-PortfolioAdvisor/) | AI-powered investment rebalancing recommendations |

---

## 🔧 How to Add a New Use-Case

See [CONTRIBUTING.md](./CONTRIBUTING.md) for step-by-step instructions.

---

## 📄 License

MIT License — free to use, modify, and share.

---

## ⭐ Show Your Support

If this repository helps you learn or inspires a project, please give it a ⭐!

Built with ❤️ to showcase practical, hands-on AI skills across Python and .NET.
