"""
💼 Portfolio Advisor
====================
Get AI-powered investment portfolio analysis and rebalancing recommendations.

Supports:
  - OpenAI GPT-4o-mini  (set OPENAI_API_KEY)
  - LLaMA 3 via Ollama  (set AI_PROVIDER=ollama)

Usage:
  python portfolio_advisor.py
"""

import os
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
def load_portfolio(filepath: str) -> dict:
    """Load portfolio data from a JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(portfolio: dict) -> str:
    """
    Summarise the portfolio into a prompt and request rebalancing recommendations.
    """
    profile = portfolio["investor_profile"]
    holdings = portfolio["holdings"]
    total = sum(h["value_usd"] for h in holdings)

    # Build a compact holdings table
    table_lines = ["Ticker | Name                     | Class        | Value (USD) | Alloc%"]
    table_lines.append("-" * 72)
    for h in holdings:
        table_lines.append(
            f"{h['ticker']:<6} | {h['name']:<24} | {h['asset_class']:<12} | "
            f"${h['value_usd']:>10,.0f} | {h['allocation_pct']:>5.1f}%"
        )
    table_text = "\n".join(table_lines)

    return f"""You are a professional portfolio advisor AI.
Review the following investment portfolio and provide rebalancing recommendations.

Investor Profile:
  Risk Tolerance : {profile['risk_tolerance']}
  Investment Horizon : {profile['horizon_years']} years
  Total Portfolio Value : ${total:,.0f}

Current Holdings:
{table_text}

Respond ONLY with a JSON object:
{{
  "overall_risk_profile": "<brief assessment of the current portfolio risk>",
  "recommendations": [
    {{
      "action": "REDUCE | INCREASE | ADD | REMOVE | HOLD",
      "ticker": "<ticker symbol>",
      "current_allocation_pct": <current % or 0 if not held>,
      "suggested_allocation_pct": <suggested %>,
      "rationale": "<one sentence reason>"
    }}
  ],
  "summary": "<two to three sentence overall rebalancing strategy>"
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
        temperature=0.3,
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
ACTION_EMOJI = {
    "REDUCE": "🔻",
    "INCREASE": "🔺",
    "ADD": "➕",
    "REMOVE": "➖",
    "HOLD": "⏸️",
}


def display_advice(portfolio: dict, result: dict) -> None:
    """Print the portfolio advice in a readable format."""
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
            print(f"  {i}. {emoji} {rec['action']:<8} {rec['ticker']:<6} "
                  f"({curr:.0f}% -> {sugg:.0f}%) — {rec['rationale']}")
        print(f"\nSummary: {result.get('summary', 'N/A')}")
    else:
        print(f"Response: {result.get('raw_response', 'No response')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_file = Path(__file__).parent / "data" / "sample_portfolio.json"

    print("💼 Portfolio Advisor")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")

    portfolio = load_portfolio(data_file)
    result = parse_json_response(call_ai(build_prompt(portfolio)))
    display_advice(portfolio, result)

    print("\n" + "=" * 60)
    print("✅ Advice generated.")


if __name__ == "__main__":
    main()
