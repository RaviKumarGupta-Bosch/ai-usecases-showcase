# 💰 Finance AI Use-Cases

This folder contains beginner-friendly AI use-cases for the finance industry.
Each sub-folder is a standalone module — pick whichever interests you and run it independently.

---

## Use-Cases

| # | Folder | What It Does |
|---|--------|--------------|
| 01 | [fraud-detection](./01-fraud-detection/) | Score financial transactions for fraud likelihood |
| 02 | [sentiment-analysis](./02-sentiment-analysis/) | Classify financial news as bullish, bearish, or neutral |
| 03 | [loan-risk-assessment](./03-loan-risk-assessment/) | Evaluate credit applications and assign a risk rating |
| 04 | [portfolio-advisor](./04-portfolio-advisor/) | Get AI-powered portfolio rebalancing recommendations |

---

## Prerequisites

All use-cases share the same setup. From the repository root:

```bash
pip install -r requirements.txt

# OpenAI
set OPENAI_API_KEY=your-key   # Windows
export OPENAI_API_KEY=your-key  # macOS/Linux

# OR local LLaMA
ollama pull llama3
set AI_PROVIDER=ollama
```

---

See the root [CONTRIBUTING.md](../CONTRIBUTING.md) to add your own finance use-case.
