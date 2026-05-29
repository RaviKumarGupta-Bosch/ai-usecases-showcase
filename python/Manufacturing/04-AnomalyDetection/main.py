"""
⚠️  Anomaly Detection — Production Line Monitor
================================================
Detect anomalies in production metrics and explain them with AI.

Run:
  python main.py
"""
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared"))

from ai_service import call_ai, AI_PROVIDER
from data_loader import load_csv
from json_helper import parse_json_response

METRIC_COLUMNS = ["cycle_time_sec", "reject_rate_pct", "utilisation_pct", "energy_kwh"]
ANOMALY_THRESHOLD_STD = 2.0
URGENCY_EMOJI = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}


def compute_baseline(rows: list) -> dict:
    half = max(len(rows) // 2, 1)
    baseline_rows = rows[:half]
    baseline = {}
    for col in METRIC_COLUMNS:
        values = [r[col] for r in baseline_rows]
        baseline[col] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        }
    return baseline


def find_anomalies(row: dict, baseline: dict) -> list:
    anomalies = []
    for col in METRIC_COLUMNS:
        b = baseline[col]
        actual, mu, std = row[col], b["mean"], b["std"]
        if std > 0 and abs(actual - mu) > ANOMALY_THRESHOLD_STD * std:
            anomalies.append(f"{col} = {actual} (baseline mean: {mu:.1f}, std: {std:.1f})")
    return anomalies


def build_prompt(row: dict, anomaly_list: list) -> str:
    bullet_lines = "\n".join(f"- {a}" for a in anomaly_list)
    return f"""You are an AI assistant for a manufacturing production line.
The following anomalies were detected at {row['timestamp']} on line {row['line_id']}:
{bullet_lines}

Full metrics for this reading:
  cycle_time_sec  : {row['cycle_time_sec']}
  reject_rate_pct : {row['reject_rate_pct']}
  utilisation_pct : {row['utilisation_pct']}
  energy_kwh      : {row['energy_kwh']}

Respond ONLY with a JSON object:
{{
  "likely_cause": "<brief root-cause hypothesis>",
  "urgency": "LOW | MEDIUM | HIGH",
  "recommended_action": "<short instruction for the operator>"
}}"""


def display_reading(row: dict, anomalies: list, ai_result: dict) -> None:
    print(f"\n🕐 {row['timestamp']}  |  {row['line_id']}")
    if not anomalies:
        print("   Status      : 🟢 NORMAL")
        return
    print("   Status      : 🔴 ANOMALY DETECTED")
    print(f"   Anomalies   : {'; '.join(anomalies)}")
    if "likely_cause" in ai_result:
        urgency = ai_result.get("urgency", "")
        emoji = URGENCY_EMOJI.get(urgency, "⚪")
        print(f"   Urgency     : {emoji} {urgency}")
        print(f"   Likely Cause: {ai_result['likely_cause']}")
        print(f"   Action      : {ai_result['recommended_action']}")
    else:
        print(f"   Response    : {ai_result.get('raw_response', 'No response')}")


def main():
    data_file = Path(__file__).parent / "data" / "sample_production_data.csv"

    print("⚠️  Anomaly Detection — Production Line Monitor")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    rows = load_csv(data_file)
    for row in rows:
        for col in METRIC_COLUMNS:
            row[col] = float(row[col])

    baseline = compute_baseline(rows)
    print(f"   Loaded {len(rows)} readings. Baseline from first {len(rows) // 2} rows.")

    anomaly_count = 0
    for row in rows:
        anomalies = find_anomalies(row, baseline)
        ai_result = parse_json_response(call_ai(build_prompt(row, anomalies))) if anomalies else {}
        if anomalies:
            anomaly_count += 1
        display_reading(row, anomalies, ai_result)

    print("\n" + "=" * 60)
    print(f"✅ Monitoring complete. {anomaly_count} anomaly/anomalies found in {len(rows)} readings.")


if __name__ == "__main__":
    main()
