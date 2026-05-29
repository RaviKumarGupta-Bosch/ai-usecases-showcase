namespace SentimentAnalysis;

public static class PromptBuilder
{
    public static string Build(NewsArticle a) => $$"""
        You are a financial sentiment AI used by professional traders.
        Analyse the following news headline and assess the market sentiment.

        Headline   : {{a.Headline}}
        Source     : {{a.Source}}
        Published  : {{a.PublishedAt}}

        Respond ONLY with a JSON object:
        {
          "sentiment": "BULLISH | BEARISH | NEUTRAL",
          "confidence_pct": <0-100>,
          "affected_assets": "<comma-separated assets or sectors>",
          "market_impact_note": "<one sentence>"
        }
        """;
}
