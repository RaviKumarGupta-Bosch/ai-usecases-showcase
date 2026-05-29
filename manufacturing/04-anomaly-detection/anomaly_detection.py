"""
⚠️  Anomaly Detection — Production Line Monitor
================================================
Detect unusual patterns in production metrics using statistical baselines
and AI-powered root-cause interpretation.

Supports:
  - OpenAI GPT-4o-mini  (set OPENAI_API_KEY)
  - LLaMA 3 via Ollama  (set AI_PROVIDER=ollama)

Usage:
  python anomaly_detection.py
"""

import os
import csv
import json
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# How many standard deviations above baseline counts as an anomaly
ANOMALY_THRESHOLD_STD = float(os.getenv("ANOMALY_THRESHOLD_STD", "2.0"))

# Columns we want to monitor for anomalies
METRIC_COLUMNS = ["cycle_time_sec", "reject_rate_pct", "utilisation_pct", "energy_kwh"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_production_data(filepath: str) -> list:
    """Load production metrics from CSV and convert numeric fields."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in METRIC_COLUMNS:
                row[col] = float(row[col])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Baseline calculation
# ---------------------------------------------------------------------------
def compute_baseline(rows: list) -> dict:
    """
    Compute mean and standard deviation for each metric column
    using the first half of the data as the 'normal' reference window.
    """
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


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
def find_anomalies(row: dict, baseline: dict) -> list:
    """
    Return a list of anomaly descriptions for metrics that deviate
    more than ANOMALY_THRESHOLD_STD standard deviations from the baseline.
    """
    anomalies = []
    for col in METRIC_COLUMNS:
        b = baseline[col]
        actual = row[col]
        std = b["std"]
        mu = b["mean"]
        # Avoid division by zero for zero-variance columns
        if std > 0 and abs(actual - mu) > ANOMALY_THRESHOLD_STD * std:
            anomalies.append(
                f"{col} = {actual} (baseline mean: {mu:.1f}, std: {std:.1f})"
            )
    return anomalies


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(row: dict, anomaly_list: list) -> str:
    """Build an AI prompt asking for root-cause analysis of the detected anomalies."""
    return f"""You are an AI assistant for a manufacturing production line.
The following anomalies were detected at {row['timestamp']} on line {row['line_id']}:

{chr(10).join('- ' + a for a in anomaly_list)}

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
        temperature=0.2,
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
URGENCY_EMOJI = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}


def display_reading(row: dict, anomalies: list, ai_result: dict) -> None:
    """Print the status for one production reading."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_file = Path(__file__).parent / "data" / "sample_production_data.csv"

    print("⚠️  Anomaly Detection — Production Line Monitor")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    rows = load_production_data(data_file)
    baseline = compute_baseline(rows)
    print(f"   Loaded {len(rows)} readings. Baseline computed from first {len(rows)//2} rows.")

    anomaly_count = 0
    for row in rows:
        anomalies = find_anomalies(row, baseline)
        if anomalies:
            ai_result = parse_json_response(call_ai(build_prompt(row, anomalies)))
            anomaly_count += 1
        else:
            ai_result = {}
        display_reading(row, anomalies, ai_result)

    print("\n" + "=" * 60)
    print(f"✅ Monitoring complete. {anomaly_count} anomaly/anomalies found in {len(rows)} readings.")


if __name__ == "__main__":
    main()
