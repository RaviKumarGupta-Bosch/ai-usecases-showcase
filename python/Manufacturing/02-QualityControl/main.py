"""
🔍 Quality Control Inspector
=============================
Use AI to evaluate product inspection reports and issue PASS / FAIL verdicts.

Run:
  python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared"))

from ai_service import call_ai, AI_PROVIDER
from data_loader import load_json
from json_helper import parse_json_response


def build_prompt(record: dict) -> str:
    measurements_text = "\n".join(
        f"  - {key}: {val['actual']} (spec: {val['spec']})"
        for key, val in record.get("measurements", {}).items()
    )
    return f"""You are a quality control AI for a manufacturing company.
Review the following product inspection record and decide if the batch passes or fails.

Batch ID        : {record['batch_id']}
Product         : {record['product']}
Inspector Notes : {record.get('visual_notes', 'None')}
Measurements vs Specifications:
{measurements_text}

Respond ONLY with a JSON object:
{{
  "verdict": "PASS" or "FAIL",
  "confidence_pct": <0-100>,
  "issues_found": ["<issue 1>", "<issue 2>"],
  "recommended_action": "<short instruction for the production team>",
  "reasoning": "<one or two sentences>"
}}"""


def display_result(record: dict, result: dict) -> None:
    print(f"\n📦 Batch: {record['batch_id']}  |  Product: {record['product']}")
    if "verdict" in result:
        icon = "✅" if result["verdict"] == "PASS" else "❌"
        print(f"   Verdict    : {icon} {result['verdict']}  (confidence: {result.get('confidence_pct', '?')}%)")
        issues = result.get("issues_found", [])
        print(f"   Issues     : {', '.join(issues) if issues else 'None'}")
        print(f"   Action     : {result.get('recommended_action', 'N/A')}")
        print(f"   Reason     : {result.get('reasoning', 'N/A')}")
    else:
        print(f"   Response   : {result.get('raw_response', 'No response')}")


def main():
    data_file = Path(__file__).parent / "data" / "sample_inspections.json"

    print("🔍 Quality Control Inspector")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    records = load_json(data_file)
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
