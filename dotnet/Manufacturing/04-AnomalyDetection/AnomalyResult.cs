namespace AnomalyDetection;

public sealed record AnomalyResult
{
    public string LikelyCause { get; init; } = "";
    public string Urgency { get; init; } = "";
    public string RecommendedAction { get; init; } = "";
}
