# 01 — Fraud Detection

> Use AI to evaluate financial transactions and flag those that look suspicious.

---

## What This Does

Fraud detection is one of the most important applications of AI in finance.
Instead of relying purely on rigid rules, this script uses an AI model to reason
about each transaction — considering amount, location, merchant, time, and user history —
and assigns a fraud risk score with an explanation.

The AI returns:
- A **fraud probability** (0–100 %)
- A **risk label** (LOW / MEDIUM / HIGH)
- **Red flags** that triggered the suspicion
- A **recommended action** (approve, flag for review, block)

---

## How to Run

```bash
cd finance/01-fraud-detection
python fraud_detection.py
```

---

## Expected Input

File: `data/sample_transactions.csv`

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique transaction ID |
| `user_id` | Customer identifier |
| `amount_usd` | Transaction amount |
| `merchant_category` | Category of the merchant |
| `location_country` | Country where the transaction occurred |
| `user_home_country` | Customer's registered country |
| `hour_of_day` | Hour the transaction was made (0–23) |
| `is_online` | 1 = online, 0 = in-person |
| `prev_30d_avg_usd` | User's average spend in the last 30 days |

---

## Sample Output

```
🕵️ Fraud Detection System
   AI Provider: OPENAI
============================================================

Transaction: TXN-0003  |  User: USR-102  |  Amount: $4,200.00
   Risk      : 🔴 HIGH  (fraud probability: 87%)
   Red Flags : Foreign country (NG vs US). Amount 14x monthly average. Unusual hour (03:00).
   Action    : BLOCK transaction. Notify customer immediately.

Transaction: TXN-0001  |  User: USR-101  |  Amount: $48.50
   Risk      : 🟢 LOW  (fraud probability: 3%)
   Red Flags : None
   Action    : APPROVE

✅ Screening complete. 1/5 transactions flagged HIGH risk.
```

---

## How It Works

1. Load transactions from CSV
2. Build a detailed prompt with transaction context
3. Ask the AI to reason about fraud signals
4. Parse structured JSON response
5. Display colour-coded risk summary

---

## Extending This Use-Case

- Integrate with a live payment stream (Kafka, webhooks)
- Add user behavioural history (velocity, device fingerprint)
- Send blocked transactions to a review queue
