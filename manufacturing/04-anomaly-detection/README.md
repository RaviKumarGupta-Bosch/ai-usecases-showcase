# 04 — Anomaly Detection

> Use AI to spot unusual patterns in real-time production line metrics.

---

## What This Does

Production lines generate many metrics every hour: cycle time, reject rate, machine utilisation, energy use.
When something goes wrong — a tool wear issue, a supply shortage, a calibration drift — the numbers shift.

This script compares each hour's metrics against the historical baseline and asks the AI to:
- Flag **anomalous readings**
- Estimate the **likely root cause**
- Suggest the **next action** for the operator

---

## How to Run

```bash
cd manufacturing/04-anomaly-detection
python anomaly_detection.py
```

---

## Expected Input

File: `data/sample_production_data.csv`

| Column | Description |
|--------|-------------|
| `timestamp` | Date and time of the reading |
| `line_id` | Production line identifier |
| `cycle_time_sec` | Average seconds per unit produced |
| `reject_rate_pct` | Percentage of units rejected |
| `utilisation_pct` | Machine utilisation rate |
| `energy_kwh` | Energy consumed that hour |

---

## Sample Output

```
⚠️  Anomaly Detection — Production Line Monitor
   AI Provider: OPENAI
============================================================

🕐 2024-12-15 08:00  |  LINE-02
   Status      : 🔴 ANOMALY DETECTED
   Anomalies   : reject_rate_pct = 12.4% (baseline: 2.1%). cycle_time_sec = 58 (baseline: 41).
   Likely Cause: Tool wear or fixture misalignment causing rework.
   Action      : Pause line for tool inspection. Notify maintenance.

🕐 2024-12-15 09:00  |  LINE-01
   Status      : 🟢 NORMAL

✅ Monitoring complete. 1 anomaly found in 6 readings.
```

---

## How It Works

1. Calculate a simple **statistical baseline** from the first half of the data
2. For each reading, compute the **deviation** from baseline
3. Send deviations to the AI to interpret
4. Display colour-coded alerts

---

## Extending This Use-Case

- Feed live data from a SCADA or MES system
- Use Z-score / IQR for tighter statistical baselines
- Trigger automated alerts (email, Slack, PagerDuty)
