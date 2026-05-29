"""
🔍 Quality Control Inspector
=============================
Use AI to evaluate product inspection reports and issue pass/fail verdicts.

Supports:
  - OpenAI GPT-4o-mini  (set OPENAI_API_KEY)
  - LLaMA 3 via Ollama  (set AI_PROVIDER=ollama)

Usage:
  python quality_control.py
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
def load_inspections(filepath: str) -> list:
    """Load product inspection records from a JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(record: dict) -> str:
    """
    Convert one inspection record into an AI prompt.
    We give the AI the measurements, the specifications, and the inspector notes.
    """
    measurements_text = "\n".join(
        f"  - {key}: {val['actual']} (spec: {val['spec']})"
        for key, val in record.get("measurements", {}).items()
    )

    return f"""You are a quality control AI for a manufacturing company.
Review the following product inspection record and decide if the batch passes or fails.

Batch ID       : {record['batch_id']}
Product        : {record['product']}
Inspector Notes: {record.get('visual_notes', 'None')}

Measurements vs Specifications:
{measurements_text}

Respond ONLY with a JSON object:
{{
  "verdict": "PASS" or "FAIL",
  "confidence_pct": <0-100>,
  "issues_found": ["<issue 1>", "<issue 2>"],
  "recommended_action": "<short instruction for the production team>",
  "reasoning": "<one or two sentences"
}}"""


# ---------------------------------------------------------------------------
# AI provider calls
# ---------------------------------------------------------------------------
def call_openai(prompt: str) -> str:
    """Call OpenAI and return the raw response text."""
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
        temperature=0.1,
    )
    return response.choices[0].message.content


def call_ollama(prompt: str) -> str:
    """Call local Ollama and return the raw response text."""
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
def display_result(record: dict, result: dict) -> None:
    """Print a formatted QC verdict for one batch."""
    print(f"\n📦 Batch: {record['batch_id']}  |  Product: {record['product']}")

    if "verdict" in result:
        icon = "✅" if result["verdict"] == "PASS" else "❌"
        conf = result.get("confidence_pct", "?")
        print(f"   Verdict    : {icon} {result['verdict']}  (confidence: {conf}%)")

        issues = result.get("issues_found", [])
        print(f"   Issues     : {', '.join(issues) if issues else 'None'}")
        print(f"   Action     : {result.get('recommended_action', 'N/A')}")
        print(f"   Reason     : {result.get('reasoning', 'N/A')}")
    else:
        print(f"   Response   : {result.get('raw_response', 'No response')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_file = Path(__file__).parent / "data" / "sample_inspections.json"

    print("🔍 Quality Control Inspector")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    records = load_inspections(data_file)
    print(f"   Loaded {len(records)} inspection records.")

    passed = 0
    for record in records:
        result = parse_json_response(call_ai(build_prompt(record)))
        display_result(record, result)
        if result.get("verdict") == "PASS":
            passed += 1

    print("\n" + "=" * 60)
    print(f"✅ Inspection complete. {passed}/{len(records)} batches passed.")


if __name__ == "__main__":
    main()
