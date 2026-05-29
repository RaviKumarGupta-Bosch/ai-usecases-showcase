# 🔷 .NET / C# Implementations

Full C# 12 / .NET 8 implementations of all AI use-cases in this repository.
Every project mirrors its Python counterpart and supports the same dual-backend pattern:

| Backend | How to select |
|---------|---------------|
| Azure OpenAI (GPT-4o-mini) | set `AI_PROVIDER=openai` (default) |
| LLaMA 3 via Ollama | set `AI_PROVIDER=ollama` |

---

## Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- **OpenAI path:** Azure OpenAI resource — set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_KEY`
- **Ollama path:** [Ollama](https://ollama.ai) running locally — `ollama pull llama3`

---

## Solution Structure

```
dotnet/
├── AIUsecasesShowcase.sln
├── Shared/
│   ├── Core/          # IAIService, OpenAIService, OllamaService, JsonHelper
│   └── Data/          # CsvLoader, JsonLoader
├── Manufacturing/
│   ├── 01-PredictiveMaintenance/
│   ├── 02-QualityControl/
│   ├── 03-DemandForecasting/
│   └── 04-AnomalyDetection/
└── Finance/
    ├── 01-FraudDetection/
    ├── 02-SentimentAnalysis/
    ├── 03-LoanRiskAssessment/
    └── 04-PortfolioAdvisor/
```

---

## Quick Start

```bash
cd dotnet

# restore all projects
dotnet restore AIUsecasesShowcase.sln

# set env vars (Windows)
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_KEY=your-key

# OR for Ollama
set AI_PROVIDER=ollama

# run any use-case
cd Manufacturing/01-PredictiveMaintenance
dotnet run
```

---

## Use-Cases

### Manufacturing
| # | Project | What It Does |
|---|---------|---------------|
| 01 | [Predictive Maintenance](Manufacturing/01-PredictiveMaintenance/) | Analyse sensor data and predict failure risk |
| 02 | [Quality Control](Manufacturing/02-QualityControl/) | Inspect batch measurements against specs |
| 03 | [Demand Forecasting](Manufacturing/03-DemandForecasting/) | Forecast demand from historical sales |
| 04 | [Anomaly Detection](Manufacturing/04-AnomalyDetection/) | Detect production-line anomalies statistically |

### Finance
| # | Project | What It Does |
|---|---------|---------------|
| 01 | [Fraud Detection](Finance/01-FraudDetection/) | Score transactions for fraud likelihood |
| 02 | [Sentiment Analysis](Finance/02-SentimentAnalysis/) | Classify financial news sentiment |
| 03 | [Loan Risk Assessment](Finance/03-LoanRiskAssessment/) | Evaluate loan applications and suggest rates |
| 04 | [Portfolio Advisor](Finance/04-PortfolioAdvisor/) | Recommend portfolio rebalancing |
