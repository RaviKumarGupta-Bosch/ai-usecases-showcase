"""
🏦 Loan Risk Assessment System
================================
Evaluate loan applications with AI to assign risk ratings and approval decisions.

Run:
  python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared"))

from ai_service import call_ai, AI_PROVIDER
from data_loader import load_csv
from json_helper import parse_json_response

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "VERY_HIGH": "🔴"}


def build_prompt(app: dict) -> str:
    dti = (float(app["existing_debt_usd"]) / float(app["annual_income_usd"])) * 100
    collateral = "Yes" if app["has_collateral"] == "1" else "No"
    return f"""You are a credit risk AI for a financial institution.
Evaluate the following loan application and provide a risk assessment.

Application ID   : {app['application_id']}
Applicant Age    : {app['age']}
Annual Income    : ${float(app['annual_income_usd']):,.0f}
Loan Amount      : ${float(app['loan_amount_usd']):,.0f}
Loan Purpose     : {app['loan_purpose']}
Credit Score     : {app['credit_score']} / 850
Existing Debt    : ${float(app['existing_debt_usd']):,.0f}  (DTI: {dti:.0f}%)
Employment       : {app['employment_years']} years
Has Collateral   : {collateral}

Respond ONLY with a JSON object:
{{
  "risk_rating": "LOW | MEDIUM | HIGH | VERY_HIGH",
  "decision": "APPROVE | APPROVE_WITH_CONDITIONS | REJECT",
  "suggested_interest_rate_pct": <float or null>,
  "key_factors": "<two or three key factors>",
  "reasoning": "<one or two sentences>"
}}"""


def display_result(app: dict, result: dict) -> str:
    loan = float(app["loan_amount_usd"])
    print(f"\nApplication: {app['application_id']}  |  Loan: ${loan:,.0f}  |  Purpose: {app['loan_purpose']}")
    if "risk_rating" in result:
        emoji = RISK_EMOJI.get(result["risk_rating"], "⚪")
        rate = result.get("suggested_interest_rate_pct")
        print(f"   Risk Rating : {emoji} {result['risk_rating']}")
        print(f"   Decision    : {result.get('decision', 'N/A')}")
        print(f"   Rate        : {f'{rate}%' if rate else 'N/A'}")
        print(f"   Key Factors : {result.get('key_factors', 'N/A')}")
        print(f"   Reasoning   : {result.get('reasoning', 'N/A')}")
        return result.get("decision", "UNKNOWN")
    print(f"   Response    : {result.get('raw_response', 'No response')}")
    return "UNKNOWN"


def main():
    data_file = Path(__file__).parent / "data" / "sample_loan_applications.csv"

    print("🏦 Loan Risk Assessment System")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    applications = load_csv(data_file)
    print(f"   Loaded {len(applications)} applications.")

    approved, rejected = 0, 0
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
