"""
Shared JSON response parser.

Strips markdown code fences (```json ... ```) that AI models often
wrap their JSON output in, then deserialises to a Python dict.
"""
import json


def parse_json_response(raw: str) -> dict:
    """Extract a JSON object from an AI response string."""
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"raw_response": raw.strip()}
