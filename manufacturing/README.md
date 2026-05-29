# 🏭 Manufacturing AI Use-Cases

This folder contains beginner-friendly AI use-cases for the manufacturing industry.
Each sub-folder is a standalone module — pick whichever interests you and run it independently.

---

## Use-Cases

| # | Folder | What It Does |
|---|--------|--------------|
| 01 | [predictive-maintenance](./01-predictive-maintenance/) | Analyze machine sensor data to predict failures before they occur |
| 02 | [quality-control](./02-quality-control/) | Evaluate product inspection reports and flag defects with AI |
| 03 | [demand-forecasting](./03-demand-forecasting/) | Forecast future production demand using AI analysis of historical data |
| 04 | [anomaly-detection](./04-anomaly-detection/) | Detect unusual patterns in real-time production line metrics |

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

## Running a Use-Case

```bash
# Navigate to any use-case folder
cd manufacturing/01-predictive-maintenance

# Run the script
python predictive_maintenance.py
```

---

See the root [CONTRIBUTING.md](../CONTRIBUTING.md) to add your own manufacturing use-case.
