"""
💼 Portfolio Advisor
====================
AI-powered investment portfolio analysis and rebalancing recommendations.

Run:
  python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared"))

from ai_service import call_ai, AI_PROVIDER
from data_loader import load_json
from json_helper import parse_json_response

ACTION_EMOJI = {
    "REDUCE": "🔻",
    "INCREASE": "🔺",
    "ADD": "➕",
    "REMOVE": "➖",
    "HOLD": "⏸️",
}


def build_prompt(portfolio: dict) -> str:
    profile = portfolio["investor_profile"]
    holdings = portfolio["holdings"]
    total = sum(h["value_usd"] for h in holdings)

    table_lines = [
        "Ticker | Name                     | Class        | Value (USD) | Alloc%",
        "-" * 72,
    ]
    for h in holdings:
        table_lines.append(
            f"{h['ticker']:<6} | {h['name']:<24} | {h['asset_class']:<12} | "
            f"${h['value_usd']:>10,.0f} | {h['allocation_pct']:>5.1f}%"
        )
    table_text = "\n".join(table_lines)

    return f"""You are a professional portfolio advisor AI.
Review the following investment portfolio and provide rebalancing recommendations.

Investor Profile:
  Risk Tolerance    : {profile['risk_tolerance']}
  Investment Horizon: {profile['horizon_years']} years
  Total Value       : ${total:,.0f}

Current Holdings:
{table_text}

Respond ONLY with a JSON object:
{{
  "overall_risk_profile": "<brief assessment of current portfolio risk>",
  "recommendations": [
    {{
      "action": "REDUCE | INCREASE | ADD | REMOVE | HOLD",
      "ticker": "<symbol>",
      "current_allocation_pct": <number>,
      "suggested_allocation_pct": <number>,
      "rationale": "<one sentence reason>"
    }}
  ],
  "summary": "<two to three sentence rebalancing strategy>"
}}"""


def display_advice(portfolio: dict, result: dict) -> None:
    profile = portfolio["investor_profile"]
    total = sum(h["value_usd"] for h in portfolio["holdings"])
    print(f"   Investor Profile : {profile['risk_tolerance'].upper()} risk | {profile['horizon_years']}-year horizon")
    print(f"   Portfolio Total  : ${total:,.0f}")
    print("=" * 60)
    if "overall_risk_profile" in result:
        print(f"\nOverall Risk Profile : {result['overall_risk_profile']}")
        print("\nRebalancing Recommendations:")
        for i, rec in enumerate(result.get("recommendations", []), 1):
            emoji = ACTION_EMOJI.get(rec["action"], "▶️")
            curr = rec.get("current_allocation_pct", 0)
            sugg = rec.get("suggested_allocation_pct", 0)
            print(
                f"  {i}. {emoji} {rec['action']:<8} {rec['ticker']:<6} "
                f"({curr:.0f}% -> {sugg:.0f}%) — {rec['rationale']}"
            )
        print(f"\nSummary: {result.get('summary', 'N/A')}")
    else:
        print(f"Response: {result.get('raw_response', 'No response')}")


def main():
    data_file = Path(__file__).parent / "data" / "sample_portfolio.json"

    print("💼 Portfolio Advisor")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")

    portfolio = load_json(data_file)
    result = parse_json_response(call_ai(build_prompt(portfolio)))
    display_advice(portfolio, result)

    print("\n" + "=" * 60)
    print("✅ Advice generated.")


if __name__ == "__main__":
    main()
