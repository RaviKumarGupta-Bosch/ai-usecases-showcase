namespace FraudDetection;

public sealed record FraudResult
{
    public int FraudProbabilityPct { get; init; }
    public string RiskLevel { get; init; } = "";
    public string[] RedFlags { get; init; } = [];
    public string RecommendedAction { get; init; } = "";
    public string Reasoning { get; init; } = "";
}
