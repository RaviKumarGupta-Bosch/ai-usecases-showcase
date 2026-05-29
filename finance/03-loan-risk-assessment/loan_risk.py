"""
🏦 Loan Risk Assessment System
==============================
Evaluate loan applications with AI to assign risk ratings and approval decisions.

Supports:
  - OpenAI GPT-4o-mini  (set OPENAI_API_KEY)
  - LLaMA 3 via Ollama  (set AI_PROVIDER=ollama)

Usage:
  python loan_risk.py
"""

import os
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_applications(filepath: str) -> list:
    """Load loan applications from a CSV file."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(app: dict) -> str:
    """Build a credit risk assessment prompt for a single loan application."""
    dti = (float(app['existing_debt_usd']) / float(app['annual_income_usd'])) * 100
    collateral = "Yes" if app['has_collateral'] == '1' else "No"

    return f"""You are a credit risk AI for a financial institution.
Evaluate the following loan application and provide a risk assessment.

Application ID    : {app['application_id']}
Applicant Age     : {app['age']}
Annual Income     : ${float(app['annual_income_usd']):,.0f}
Loan Amount       : ${float(app['loan_amount_usd']):,.0f}
Loan Purpose      : {app['loan_purpose']}
Credit Score      : {app['credit_score']} / 850
Existing Debt     : ${float(app['existing_debt_usd']):,.0f}  (DTI: {dti:.0f}%)
Employment        : {app['employment_years']} years
Has Collateral    : {collateral}

Respond ONLY with a JSON object:
{{
  "risk_rating": "LOW | MEDIUM | HIGH | VERY_HIGH",
  "decision": "APPROVE | APPROVE_WITH_CONDITIONS | REJECT",
  "suggested_interest_rate_pct": <float or null>,
  "key_factors": "<two or three key positive or negative factors>",
  "reasoning": "<one or two sentences>"
}}"""


# ---------------------------------------------------------------------------
# AI provider calls
# ---------------------------------------------------------------------------
def call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Run: pip install openai") from exc

    if not OPENAI_API_KEY:
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


def call_ollama(prompt: str) -> str:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("Run: pip install requests") from exc

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def call_ai(prompt: str) -> str:
    if AI_PROVIDER == "ollama":
        return call_ollama(prompt)
    return call_openai(prompt)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_json_response(raw: str) -> dict:
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"raw_response": raw.strip()}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "VERY_HIGH": "🔴"}


def display_result(app: dict, result: dict) -> str:
    """Print a formatted assessment result and return the decision."""
    loan = float(app['loan_amount_usd'])
    print(f"\nApplication: {app['application_id']}  |  Loan: ${loan:,.0f}  |  Purpose: {app['loan_purpose']}")

    if "risk_rating" in result:
        emoji = RISK_EMOJI.get(result["risk_rating"], "⚪")
        rate = result.get('suggested_interest_rate_pct')
        rate_str = f"{rate}%" if rate else "N/A"
        print(f"  Risk Rating : {emoji} {result['risk_rating']}")
        print(f"  Decision    : {result.get('decision', 'N/A')}")
        print(f"  Rate        : {rate_str}")
        print(f"  Key Factors : {result.get('key_factors', 'N/A')}")
        print(f"  Reasoning   : {result.get('reasoning', 'N/A')}")
        return result.get("decision", "UNKNOWN")
    else:
        print(f"  Response    : {result.get('raw_response', 'No response')}")
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_file = Path(__file__).parent / "data" / "sample_loan_applications.csv"

    print("🏦 Loan Risk Assessment System")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    applications = load_applications(data_file)
    print(f"   Loaded {len(applications)} applications.")

    approved = 0
    rejected = 0
    for app in applications:
        result = parse_json_response(call_ai(build_prompt(app)))
        decision = display_result(app, result)
        if decision == "REJECT":
            rejected += 1
        elif decision in ("APPROVE", "APPROVE_WITH_CONDITIONS"):
            approved += 1

    print("\n" + "=" * 60)
    print(f"✅ Assessment complete. Approved: {approved} | Rejected: {rejected}")


if __name__ == "__main__":
    main()
