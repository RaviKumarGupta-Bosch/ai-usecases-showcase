namespace SentimentAnalysis;

public sealed record SentimentResult
{
    public string Sentiment { get; init; } = "";
    public int ConfidencePct { get; init; }
    public string AffectedAssets { get; init; } = "";
    public string MarketImpactNote { get; init; } = "";
}
