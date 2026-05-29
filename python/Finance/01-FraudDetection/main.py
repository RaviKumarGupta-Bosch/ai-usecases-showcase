"""
🕵️ Fraud Detection System
==========================
Evaluate financial transactions with AI to identify fraud risk.

Run:
  python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared"))

from ai_service import call_ai, AI_PROVIDER
from data_loader import load_csv
from json_helper import parse_json_response

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}


def build_prompt(txn: dict) -> str:
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

Respond ONLY with a JSON object:
{{
  "fraud_probability_pct": <0-100>,
  "risk_level": "LOW | MEDIUM | HIGH",
  "red_flags": ["<flag 1>", "<flag 2>"],
  "recommended_action": "APPROVE | FLAG_FOR_REVIEW | BLOCK",
  "reasoning": "<one or two sentences>"
}}"""


def display_result(txn: dict, result: dict) -> None:
    amount = float(txn["amount_usd"])
    print(f"\nTransaction: {txn['transaction_id']}  |  User: {txn['user_id']}  |  Amount: ${amount:,.2f}")
    if "risk_level" in result:
        emoji = RISK_EMOJI.get(result["risk_level"], "⚪")
        print(f"   Risk      : {emoji} {result['risk_level']}  (fraud probability: {result.get('fraud_probability_pct', '?')}%)")
        flags = result.get("red_flags", [])
        print(f"   Red Flags : {'; '.join(flags) if flags else 'None'}")
        print(f"   Action    : {result.get('recommended_action', 'N/A')}")
        print(f"   Reason    : {result.get('reasoning', 'N/A')}")
    else:
        print(f"   Response  : {result.get('raw_response', 'No response')}")


def main():
    data_file = Path(__file__).parent / "data" / "sample_transactions.csv"

    print("🕵️ Fraud Detection System")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    transactions = load_csv(data_file)
    print(f"   Loaded {len(transactions)} transactions.")

    high_risk_count = 0
    for txn in transactions:
        result = parse_json_response(call_ai(build_prompt(txn)))
        display_result(txn, result)
        if result.get("risk_level") == "HIGH":
            high_risk_count += 1

    print("\n" + "=" * 60)
    print(f"✅ Screening complete. {high_risk_count}/{len(transactions)} flagged HIGH risk.")


if __name__ == "__main__":
    main()
