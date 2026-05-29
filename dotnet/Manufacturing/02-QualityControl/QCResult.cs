namespace QualityControl;

public sealed record QCResult
{
    public string Verdict { get; init; } = "";
    public int ConfidencePct { get; init; }
    public string[] IssuesFound { get; init; } = [];
    public string RecommendedAction { get; init; } = "";
}
