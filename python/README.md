# 🐍 Python — AI Use-Cases Showcase

> All 8 use-cases implemented in Python, supporting OpenAI and local Ollama (LLaMA 3) backends.

## Structure

```
python/
├── Shared/                      ← Reusable helpers (AI backends, data loaders, JSON parser)
│   ├── ai_service.py            ← call_ai(prompt) — dispatches to OpenAI or Ollama
│   ├── data_loader.py           ← load_csv(path) and load_json(path)
│   └── json_helper.py           ← parse_json_response(raw)
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

## Quick Start

```bash
# Install dependencies
pip install openai requests

# --- OpenAI (cloud) ---
set OPENAI_API_KEY=your-key          # Windows
export OPENAI_API_KEY=your-key       # macOS/Linux

# --- OR local Ollama ---
ollama pull llama3
set AI_PROVIDER=ollama

# Run any use-case
cd Manufacturing/01-PredictiveMaintenance
python main.py
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_PROVIDER` | `openai` | Set to `ollama` for local inference |
| `OPENAI_API_KEY` | *(required for OpenAI)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |

---

## 🏭 Manufacturing Use-Cases

| # | Folder | Description |
|---|--------|-------------|
| 01 | [01-PredictiveMaintenance](./Manufacturing/01-PredictiveMaintenance/) | Predict machine failures from sensor data |
| 02 | [02-QualityControl](./Manufacturing/02-QualityControl/) | Classify production batches as PASS / FAIL |
| 03 | [03-DemandForecasting](./Manufacturing/03-DemandForecasting/) | Forecast demand from 12-month sales history |
| 04 | [04-AnomalyDetection](./Manufacturing/04-AnomalyDetection/) | Detect anomalies on production lines |

## 💰 Finance Use-Cases

| # | Folder | Description |
|---|--------|-------------|
| 01 | [01-FraudDetection](./Finance/01-FraudDetection/) | Flag suspicious financial transactions |
| 02 | [02-SentimentAnalysis](./Finance/02-SentimentAnalysis/) | Score financial news as BULLISH / BEARISH / NEUTRAL |
| 03 | [03-LoanRiskAssessment](./Finance/03-LoanRiskAssessment/) | Rate credit applications with AI reasoning |
| 04 | [04-PortfolioAdvisor](./Finance/04-PortfolioAdvisor/) | AI-powered investment rebalancing recommendations |

---

## Shared Helpers

| File | Purpose |
|------|---------|
| `Shared/ai_service.py` | `call_ai(prompt)` dispatches to OpenAI or Ollama. Also exports `AI_PROVIDER`. |
| `Shared/data_loader.py` | `load_csv(path)` → list of dicts; `load_json(path)` → parsed object |
| `Shared/json_helper.py` | `parse_json_response(raw)` strips markdown fences and deserialises JSON |
