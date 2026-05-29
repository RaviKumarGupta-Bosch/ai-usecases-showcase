"""
📈 Demand Forecasting Assistant
================================
Analyse historical sales data with AI to forecast future demand.

Run:
  python main.py
"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared"))

from ai_service import call_ai, AI_PROVIDER
from data_loader import load_csv
from json_helper import parse_json_response


def build_prompt(product_name: str, history: list) -> str:
    table_lines = ["Month      | Units Sold | Avg Price | Promo", "-" * 45]
    for row in history:
        promo = "Yes" if row.get("is_promo_month") == "1" else "No"
        table_lines.append(
            f"{row['month']}  |  {int(row['units_sold']):>9,} | {float(row['avg_price']):>9.2f} | {promo}"
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


def display_forecast(product_name: str, result: dict) -> None:
    print(f"\nProduct: {product_name}")
    if "trend_summary" in result:
        print(f"  Trend       : {result['trend_summary']}")
        print("  Forecast    :")
        for entry in result.get("forecast", []):
            print(f"    {entry['month']} : {entry['predicted_units']:,} units")
        print(f"  Key Risks   : {result.get('key_risks', 'N/A')}")
        print(f"  Plan        : {result.get('production_recommendation', 'N/A')}")
    else:
        print(f"  Response    : {result.get('raw_response', 'No response')}")


def main():
    data_file = Path(__file__).parent / "data" / "sample_demand_data.csv"

    print("📈 Demand Forecasting Assistant")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    rows = load_csv(data_file)
    products = defaultdict(list)
    for row in rows:
        products[row["product"]].append(row)

    print(f"   Loaded data for {len(products)} product(s).")

    for product_name, history in products.items():
        result = parse_json_response(call_ai(build_prompt(product_name, history)))
        display_forecast(product_name, result)

    print("\n" + "=" * 60)
    print("✅ Forecast complete.")


if __name__ == "__main__":
    main()
