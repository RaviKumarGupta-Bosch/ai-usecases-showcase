"""
🕵️ Fraud Detection System
========================
Evaluate financial transactions with AI to identify fraud risk.

Supports:
  - OpenAI GPT-4o-mini  (set OPENAI_API_KEY)
  - LLaMA 3 via Ollama  (set AI_PROVIDER=ollama)

Usage:
  python fraud_detection.py
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
def load_transactions(filepath: str) -> list:
    """Load transaction records from a CSV file."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(txn: dict) -> str:
    """Create a fraud analysis prompt for a single transaction."""
    return f"""You are a fraud detection AI for a financial institution.
Analyse the following transaction and assess the likelihood of fraud.

Transaction ID       : {txn['transaction_id']}
User ID              : {txn['user_id']}
Amount               : ${float(txn['amount_usd']):,.2f}
Merchant Category    : {txn['merchant_category']}
Transaction Country  : {txn['location_country']}
User Home Country    : {txn['user_home_country']}
Hour of Day          : {txn['hour_of_day']}:00
Channel              : {'Online' if txn['is_online'] == '1' else 'In-Person'}
User 30-day Avg Spend: ${float(txn['prev_30d_avg_usd']):,.2f}

Consider: unusual location, abnormal amount, suspicious timing, channel mismatch.

Respond ONLY with a JSON object:
{{
  "fraud_probability_pct": <0-100>,
  "risk_level": "LOW | MEDIUM | HIGH",
  "red_flags": ["<flag 1>", "<flag 2>"],
  "recommended_action": "APPROVE | FLAG_FOR_REVIEW | BLOCK",
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
RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}


def display_result(txn: dict, result: dict) -> None:
    """Print a formatted fraud risk summary for one transaction."""
    amount = float(txn['amount_usd'])
    print(f"\nTransaction: {txn['transaction_id']}  |  User: {txn['user_id']}  |  Amount: ${amount:,.2f}")

    if "risk_level" in result:
        emoji = RISK_EMOJI.get(result["risk_level"], "⚪")
        prob = result.get("fraud_probability_pct", "?")
        print(f"   Risk      : {emoji} {result['risk_level']}  (fraud probability: {prob}%)")

        flags = result.get("red_flags", [])
        print(f"   Red Flags : {'; '.join(flags) if flags else 'None'}")
        print(f"   Action    : {result.get('recommended_action', 'N/A')}")
        print(f"   Reason    : {result.get('reasoning', 'N/A')}")
    else:
        print(f"   Response  : {result.get('raw_response', 'No response')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_file = Path(__file__).parent / "data" / "sample_transactions.csv"

    print("🕵️ Fraud Detection System")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    transactions = load_transactions(data_file)
    print(f"   Loaded {len(transactions)} transactions.")

    high_risk_count = 0
    for txn in transactions:
        result = parse_json_response(call_ai(build_prompt(txn)))
        display_result(txn, result)
        if result.get("risk_level") == "HIGH":
            high_risk_count += 1

    print("\n" + "=" * 60)
    print(f"✅ Screening complete. {high_risk_count}/{len(transactions)} transactions flagged HIGH risk.")


if __name__ == "__main__":
    main()
