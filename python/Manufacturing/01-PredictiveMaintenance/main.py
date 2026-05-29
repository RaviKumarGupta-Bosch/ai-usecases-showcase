"""
🏭 Predictive Maintenance
=========================
Analyse machine sensor data with AI to predict potential failures.

Run:
  python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared"))

from ai_service import call_ai, AI_PROVIDER
from data_loader import load_csv
from json_helper import parse_json_response

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}


def build_prompt(sensor: dict) -> str:
    return f"""You are an industrial AI assistant specialised in predictive maintenance.
Analyse the following machine sensor reading and assess the risk of failure.

Machine ID       : {sensor['machine_id']}
Temperature (C)  : {sensor['temperature']}
Vibration (mm/s) : {sensor['vibration']}
Pressure (bar)   : {sensor['pressure']}
Operating Hours  : {sensor['operating_hours']}

Respond ONLY with a JSON object:
{{
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "predicted_failure_days": <integer or null>,
  "recommendation": "<short action for the maintenance team>",
  "reasoning": "<one or two sentences explaining your assessment>"
}}"""


def display_result(sensor: dict, result: dict) -> None:
    print(f"\n📊 Machine: {sensor['machine_id']}")
    if "risk_level" in result:
        emoji = RISK_EMOJI.get(result["risk_level"], "⚪")
        print(f"   Risk Level : {emoji} {result['risk_level']}")
        days = result.get("predicted_failure_days")
        if days:
            print(f"   Failure in : ~{days} days")
        print(f"   Action     : {result.get('recommendation', 'N/A')}")
        print(f"   Reason     : {result.get('reasoning', 'N/A')}")
    else:
        print(f"   Response   : {result.get('raw_response', 'No response received')}")


def main():
    data_file = Path(__file__).parent / "data" / "sample_sensor_data.csv"

    print("🏭 Predictive Maintenance Analysis")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    sensors = load_csv(data_file)
    print(f"   Loaded {len(sensors)} machine records.")

    for sensor in sensors:
        result = parse_json_response(call_ai(build_prompt(sensor)))
        display_result(sensor, result)

    print("\n" + "=" * 60)
    print("✅ Analysis complete.")


if __name__ == "__main__":
    main()
