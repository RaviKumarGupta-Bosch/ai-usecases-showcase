# 03 — Loan Risk Assessment

> Use AI to evaluate loan applications and assign a credit risk rating with clear reasoning.

---

## What This Does

Lenders need to decide quickly whether to approve a loan and at what interest rate.
Traditional credit scoring models are rigid. AI can reason across multiple factors holistically.

This script reads loan applications and asks the AI to:
- Assign a **risk rating** (LOW / MEDIUM / HIGH / VERY HIGH)
- Recommend **approval or rejection**
- Suggest an appropriate **interest rate**
- Explain the key factors driving the decision

---

## How to Run

```bash
cd finance/03-loan-risk-assessment
python loan_risk.py
```

---

## Expected Input

File: `data/sample_loan_applications.csv`

| Column | Description |
|--------|-------------|
| `application_id` | Unique application identifier |
| `age` | Applicant age |
| `annual_income_usd` | Annual income |
| `loan_amount_usd` | Requested loan amount |
| `loan_purpose` | Purpose of the loan |
| `credit_score` | Credit score (300–850) |
| `existing_debt_usd` | Current outstanding debt |
| `employment_years` | Years in current employment |
| `has_collateral` | 1 = yes, 0 = no |

---

## Sample Output

```
🏦 Loan Risk Assessment System
   AI Provider: OPENAI
============================================================

Application: APP-0002  |  Loan: $85,000  |  Purpose: Real Estate
  Risk Rating : 🟡 MEDIUM
  Decision    : APPROVE with conditions
  Rate        : 7.2%
  Key Factors : Solid income but high debt-to-income ratio (68%). Good credit (720).

Application: APP-0004  |  Loan: $25,000  |  Purpose: Personal
  Risk Rating : 🔴 HIGH
  Decision    : REJECT
  Rate        : N/A
  Key Factors : Low credit score (510). No collateral. Short employment history.

✅ Assessment complete. Approved: 3 | Rejected: 2
```

---

## Extending This Use-Case

- Add more features: payment history, number of credit inquiries
- Compare AI risk rating with a traditional scoring model
- Build an API endpoint with FastAPI to serve assessments in real time
