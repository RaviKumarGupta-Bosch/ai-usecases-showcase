# 01 — Predictive Maintenance

> Use AI to analyze machine sensor readings and predict whether a machine is at risk of failing.

---

## What This Does

In a factory, machines generate continuous streams of sensor data — temperature, vibration, pressure.
Predictive maintenance means spotting warning signs **before** a breakdown happens, saving both
cost and downtime.

This script feeds each machine's sensor row to an AI model. The AI responds with:
- A **risk level** (LOW / MEDIUM / HIGH / CRITICAL)
- An estimated **days-until-failure** (if applicable)
- A **recommended action** ("schedule inspection", "replace bearing", etc.)
- A brief **reasoning** explanation

---

## How to Run

```bash
cd manufacturing/01-predictive-maintenance
python predictive_maintenance.py
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PROVIDER` | `openai` | Use `openai` or `ollama` |
| `OPENAI_API_KEY` | *(required for openai)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |

---

## Expected Input

File: `data/sample_sensor_data.csv`

| Column | Type | Description |
|--------|------|-------------|
| `machine_id` | string | Unique machine identifier |
| `temperature` | float | Operating temperature in °C |
| `vibration` | float | Vibration level in mm/s |
| `pressure` | float | Operating pressure in bar |
| `operating_hours` | int | Total hours the machine has run |

---

## Sample Output

```
🏭 Predictive Maintenance Analysis
   AI Provider: OPENAI
============================================================

📊 Analyzing Machine: MACHINE-001
   Risk Level:  🟢 LOW
   Action:      Continue normal operations. Schedule routine check in 30 days.
   Reason:      All sensor readings are within safe operating ranges.

📊 Analyzing Machine: MACHINE-004
   Risk Level:  🔴 CRITICAL
   Failure in:  ~7 days
   Action:      Immediate inspection required. Possible bearing failure.
   Reason:      Temperature 102°C exceeds safe threshold. Vibration at 6.3 mm/s
                is 3x normal. High operating hours compound the risk.

✅ Analysis complete.
```

---

## How It Works

1. **Load data** — reads `sample_sensor_data.csv` row by row
2. **Build prompt** — formats each row into a structured AI prompt
3. **Call AI** — sends the prompt to OpenAI or Ollama
4. **Parse response** — extracts the JSON result from the AI reply
5. **Display results** — prints a colour-coded summary

---

## Switching AI Providers

```bash
# Use LLaMA locally (no API key needed)
set AI_PROVIDER=ollama    # Windows
export AI_PROVIDER=ollama   # macOS/Linux
python predictive_maintenance.py
```

---

## Extending This Use-Case

- Add more sensor columns (humidity, RPM, power consumption)
- Connect to a live MQTT/IoT stream instead of a CSV file
- Save results to a database or send email alerts for CRITICAL machines
