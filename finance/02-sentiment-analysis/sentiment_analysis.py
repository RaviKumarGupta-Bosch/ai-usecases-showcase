"""
📰 Financial Sentiment Analyser
================================
Score financial news headlines as BULLISH, BEARISH, or NEUTRAL using AI.

Supports:
  - OpenAI GPT-4o-mini  (set OPENAI_API_KEY)
  - LLaMA 3 via Ollama  (set AI_PROVIDER=ollama)

Usage:
  python sentiment_analysis.py
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
def load_news(filepath: str) -> list:
    """Load financial news records from a JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(article: dict) -> str:
    """Build a sentiment analysis prompt for one news article."""
    return f"""You are a financial sentiment AI used by professional traders and analysts.
Analyse the following news headline and assess the market sentiment.

Headline   : {article['headline']}
Source     : {article.get('source', 'Unknown')}
Published  : {article.get('published_at', 'Unknown')}

Respond ONLY with a JSON object:
{{
  "sentiment": "BULLISH | BEARISH | NEUTRAL",
  "confidence_pct": <0-100>,
  "affected_assets": "<comma-separated list of assets or sectors>",
  "market_impact_note": "<one sentence explaining the likely market effect>"
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
SENTIMENT_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}


def display_result(article: dict, result: dict, idx: int) -> str:
    """Print a formatted sentiment result and return the sentiment label."""
    print(f"\n[{article['id']}]  {article.get('published_at', '')}")
    print(f"Headline: \"{article['headline']}\"")

    if "sentiment" in result:
        emoji = SENTIMENT_EMOJI.get(result["sentiment"], "⚪")
        conf = result.get("confidence_pct", "?")
        print(f"  Sentiment  : {emoji} {result['sentiment']}  (confidence: {conf}%)")
        print(f"  Affects    : {result.get('affected_assets', 'N/A')}")
        print(f"  Impact     : {result.get('market_impact_note', 'N/A')}")
        return result["sentiment"]
    else:
        print(f"  Response   : {result.get('raw_response', 'No response')}")
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_file = Path(__file__).parent / "data" / "sample_news.json"

    print("📰 Financial Sentiment Analyser")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    articles = load_news(data_file)
    print(f"   Loaded {len(articles)} articles.")

    counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    for i, article in enumerate(articles):
        result = parse_json_response(call_ai(build_prompt(article)))
        sentiment = display_result(article, result, i)
        if sentiment in counts:
            counts[sentiment] += 1

    print("\n" + "=" * 60)
    print(f"✅ Analysis complete. Bullish: {counts['BULLISH']} | Bearish: {counts['BEARISH']} | Neutral: {counts['NEUTRAL']}")


if __name__ == "__main__":
    main()
