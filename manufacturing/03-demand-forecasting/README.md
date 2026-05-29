# 03 — Demand Forecasting

> Use AI to analyse historical sales data and predict future production demand.

---

## What This Does

Production planners need to know how much to produce next month. Too much = waste; too little = lost sales.

This script reads historical monthly sales data and asks the AI to:
- Identify **trends** and **seasonal patterns**
- Predict demand for the **next 3 months**
- Highlight any **risks or uncertainties**
- Suggest a **production plan**

---

## How to Run

```bash
cd manufacturing/03-demand-forecasting
python demand_forecasting.py
```

---

## Expected Input

File: `data/sample_demand_data.csv`

| Column | Description |
|--------|-------------|
| `month` | Month in YYYY-MM format |
| `product` | Product name |
| `units_sold` | Units sold that month |
| `avg_price` | Average selling price |
| `is_promo_month` | 1 if a promotional campaign ran, else 0 |

---

## Sample Output

```
📈 Demand Forecasting Assistant
   AI Provider: OPENAI
============================================================

Product: Widget-A
  Trend       : Upward (+12% YoY). Strong peak in Nov-Dec.
  Forecast    :
    2025-01 : 1,350 units
    2025-02 : 1,180 units
    2025-03 : 1,220 units
  Risk        : Promo effect in Jan 2024 may inflate baseline.
  Plan        : Build safety stock of ~200 units heading into Q1.

✅ Forecast complete.
```

---

## How It Works

1. Group historical data by product
2. Summarise the last 12 months into a compact table
3. Ask the AI to reason about trends and produce a forecast
4. Display the forecast with risk notes

---

## Extending This Use-Case

- Add external signals (weather, economic indicators)
- Use statsmodels / Prophet for a statistical baseline, then ask AI to adjust it
- Output forecast to Excel for the planning team
