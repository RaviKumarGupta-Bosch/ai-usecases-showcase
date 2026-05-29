# 04 — Portfolio Advisor

> Use AI to review your investment portfolio and get personalised rebalancing recommendations.

---

## What This Does

A well-balanced portfolio aligns with your risk tolerance and investment horizon.
This script takes a snapshot of your current holdings and asks the AI to:

- Assess the **overall risk profile**
- Identify **over-concentrated** positions
- Suggest a **rebalancing plan** (what to reduce, what to increase)
- Explain the **reasoning** behind each change

---

## How to Run

```bash
cd finance/04-portfolio-advisor
python portfolio_advisor.py
```

---

## Expected Input

File: `data/sample_portfolio.json`

Top-level fields:
- `investor_profile` — `risk_tolerance` (conservative/moderate/aggressive), `horizon_years`
- `holdings` — list of positions, each with `ticker`, `name`, `asset_class`, `value_usd`, `allocation_pct`

---

## Sample Output

```
💼 Portfolio Advisor
   AI Provider: OPENAI
   Investor Profile: MODERATE risk | 10-year horizon
   Portfolio Total: $52,500
============================================================

Overall Risk Profile : Moderately aggressive (equity-heavy)

Rebalancing Recommendations:
  1. REDUCE  TSLA   (12% -> 5%) — High single-stock volatility. Over-concentrated in tech.
  2. INCREASE BND    (5% -> 15%) — Insufficient fixed income for a moderate risk profile.
  3. ADD     VNQ    (0% -> 5%)  — Add real estate for diversification and inflation hedge.
  4. HOLD    AAPL   (15%)       — Well-positioned. No change needed.

Summary: Shift ~10% from individual tech stocks into bonds and real estate
         to better match your moderate risk profile and 10-year horizon.

✅ Advice generated.
```

---

## Extending This Use-Case

- Pull live prices from Yahoo Finance or Alpha Vantage
- Add current market context (interest rates, inflation) to the prompt
- Generate a PDF report with charts using `reportlab` or `matplotlib`
