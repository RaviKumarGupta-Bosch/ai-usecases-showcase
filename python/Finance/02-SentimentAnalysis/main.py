"""
📰 Financial Sentiment Analyser
================================
Score financial news headlines as BULLISH, BEARISH, or NEUTRAL using AI.

Run:
  python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared"))

from ai_service import call_ai, AI_PROVIDER
from data_loader import load_json
from json_helper import parse_json_response

SENTIMENT_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}


def build_prompt(article: dict) -> str:
    return f"""You are a financial sentiment AI used by professional traders and analysts.
Analyse the following news headline and assess the market sentiment.

Headline  : {article['headline']}
Source    : {article.get('source', 'Unknown')}
Published : {article.get('published_at', 'Unknown')}

Respond ONLY with a JSON object:
{{
  "sentiment": "BULLISH | BEARISH | NEUTRAL",
  "confidence_pct": <0-100>,
  "affected_assets": "<comma-separated list of assets or sectors>",
  "market_impact_note": "<one sentence explaining the likely market effect>"
}}"""


def display_result(article: dict, result: dict) -> str:
    print(f"\n[{article['id']}]  {article.get('published_at', '')}")
    print(f"Headline: \"{article['headline']}\"")
    if "sentiment" in result:
        emoji = SENTIMENT_EMOJI.get(result["sentiment"], "⚪")
        print(f"  Sentiment  : {emoji} {result['sentiment']}  (confidence: {result.get('confidence_pct', '?')}%)")
        print(f"  Affects    : {result.get('affected_assets', 'N/A')}")
        print(f"  Impact     : {result.get('market_impact_note', 'N/A')}")
        return result["sentiment"]
    print(f"  Response   : {result.get('raw_response', 'No response')}")
    return "UNKNOWN"


def main():
    data_file = Path(__file__).parent / "data" / "sample_news.json"

    print("📰 Financial Sentiment Analyser")
    print(f"   AI Provider : {AI_PROVIDER.upper()}")
    print("=" * 60)

    articles = load_json(data_file)
    print(f"   Loaded {len(articles)} articles.")

    counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    for article in articles:
        result = parse_json_response(call_ai(build_prompt(article)))
        sentiment = display_result(article, result)
        if sentiment in counts:
            counts[sentiment] += 1

    print("\n" + "=" * 60)
    print(f"✅ Analysis complete. Bullish: {counts['BULLISH']} | Bearish: {counts['BEARISH']} | Neutral: {counts['NEUTRAL']}")


if __name__ == "__main__":
    main()
