"""
🏭 Predictive Maintenance
=========================
Analyze machine sensor data with AI to predict potential failures
before they happen — saving cost and avoiding unplanned downtime.

Supports:
  - OpenAI GPT-4o-mini  (set OPENAI_API_KEY)
  - LLaMA 3 via Ollama  (set AI_PROVIDER=ollama)

Usage:
  python predictive_maintenance.py
"""

import os
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration  (change via environment variables, not here)
# ---------------------------------------------------------------------------
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")   # "openai" or "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sensor_data(filepath: str) -> list:
    """Read sensor readings from a CSV file and return a list of row dicts."""
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(sensor: dict) -> str:
    """
    Turn one sensor reading into an AI prompt.
    The AI is asked to respond with a strict JSON object.
    """
    return f"""You are an industrial AI assistant specialised in predictive maintenance.
Analyse the following machine sensor reading and assess the risk of failure.

Machine ID       : {sensor['machine_id']}
Temperature (C)  : {sensor['temperature']}
Vibration (mm/s) : {sensor['vibration']}
Pressure (bar)   : {sensor['pressure']}
Operating Hours  : {sensor['operating_hours']}

Respond ONLY with a JSON object in this exact format:
{{
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "predicted_failure_days": <integer or null>,
  "recommendation": "<short action for the maintenance team>",
  "reasoning": "<one or two sentences explaining your assessment>"
}}"""


# ---------------------------------------------------------------------------
# AI provider calls
# ---------------------------------------------------------------------------
def call_openai(prompt: str) -> str:
    """Send a prompt to OpenAI and return the raw text response."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Run: pip install openai") from exc

    if not OPENAI_API_KEY:
        raise ValueError("Set the OPENAI_API_KEY environment variable first.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,   # low temperature = more consistent, factual output
    )
    return response.choices[0].message.content


def call_ollama(prompt: str) -> str:
    """Send a prompt to a local Ollama instance and return the raw text response."""
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
    """
    Extract a JSON object from the AI response.
    Handles cases where the AI wraps the JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        # Return the raw text so the user can still see what the AI said
        return {"raw_response": raw.strip()}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse_machine(sensor: dict) -> dict:
    """Build a prompt, call the AI, and parse the result for one machine."""
    prompt = build_prompt(sensor)
    raw_response = call_ai(prompt)
    return parse_json_response(raw_response)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
RISK_EMOJI = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "CRITICAL": "🔴",
}


def display_result(sensor: dict, result: dict) -> None:
    """Print a human-readable summary of the AI analysis for one machine."""
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
        # Fallback: print whatever the AI returned
        print(f"   Response   : {result.get('raw_response', 'No response received')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_file = Path(__file__).parent / "data" / "sample_sensor_data.csv"

    print("🏭 Predictive Maintenance Analysis")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    sensors = load_sensor_data(data_file)
    print(f"   Loaded {len(sensors)} machine records.")

    for sensor in sensors:
        result = analyse_machine(sensor)
        display_result(sensor, result)

    print("\n" + "=" * 60)
    print("✅ Analysis complete.")


if __name__ == "__main__":
    main()
