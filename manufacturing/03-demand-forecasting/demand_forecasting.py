"""
📈 Demand Forecasting Assistant
================================
Analyse historical production/sales data with AI to forecast future demand.

Supports:
  - OpenAI GPT-4o-mini  (set OPENAI_API_KEY)
  - LLaMA 3 via Ollama  (set AI_PROVIDER=ollama)

Usage:
  python demand_forecasting.py
"""

import os
import csv
import json
from pathlib import Path
from collections import defaultdict

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
def load_demand_data(filepath: str) -> dict:
    """
    Load CSV and group rows by product.
    Returns: {product_name: [list of row dicts]}
    """
    products = defaultdict(list)
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products[row["product"]].append(row)
    return dict(products)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(product_name: str, history: list) -> str:
    """
    Summarise the product's sales history into a prompt and ask for a forecast.
    """
    # Build a compact table of the history
    table_lines = ["Month      | Units Sold | Avg Price | Promo"]
    table_lines.append("-" * 45)
    for row in history:
        promo = "Yes" if row.get("is_promo_month") == "1" else "No"
        table_lines.append(
            f"{row['month']}  |  {row['units_sold']:>9} | {row['avg_price']:>9} | {promo}"
        )
    table_text = "\n".join(table_lines)

    return f"""You are a demand forecasting AI for a manufacturing company.
Analyse the monthly sales history below for product '{product_name}'
and generate a forecast for the next 3 months.

Sales History:
{table_text}

Respond ONLY with a JSON object:
{{
  "trend_summary": "<brief description of the trend>",
  "forecast": [
    {{"month": "YYYY-MM", "predicted_units": <integer>}},
    {{"month": "YYYY-MM", "predicted_units": <integer>}},
    {{"month": "YYYY-MM", "predicted_units": <integer>}}
  ],
  "key_risks": "<any risks or caveats to the forecast>",
  "production_recommendation": "<short actionable advice for the planning team>"
}}"""


# ---------------------------------------------------------------------------
# AI provider calls
# ---------------------------------------------------------------------------
def call_openai(prompt: str) -> str:
    """Call OpenAI and return raw response text."""
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
    """Call local Ollama and return raw response text."""
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
    """Dispatch to the configured AI provider."""
    if AI_PROVIDER == "ollama":
        return call_ollama(prompt)
    return call_openai(prompt)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_json_response(raw: str) -> dict:
    """Extract JSON from the AI response, handling markdown fences."""
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
def display_forecast(product_name: str, result: dict) -> None:
    """Print the demand forecast for one product."""
    print(f"\nProduct: {product_name}")

    if "trend_summary" in result:
        print(f"  Trend       : {result['trend_summary']}")
        print("  Forecast    :")
        for entry in result.get("forecast", []):
            print(f"    {entry['month']} : {entry['predicted_units']:,} units")
        print(f"  Risk        : {result.get('key_risks', 'N/A')}")
        print(f"  Plan        : {result.get('production_recommendation', 'N/A')}")
    else:
        print(f"  Response    : {result.get('raw_response', 'No response')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_file = Path(__file__).parent / "data" / "sample_demand_data.csv"

    print("📈 Demand Forecasting Assistant")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    product_data = load_demand_data(data_file)
    print(f"   Loaded data for {len(product_data)} product(s).")

    for product_name, history in product_data.items():
        result = parse_json_response(call_ai(build_prompt(product_name, history)))
        display_forecast(product_name, result)

    print("\n" + "=" * 60)
    print("✅ Forecast complete.")


if __name__ == "__main__":
    main()
