namespace PortfolioAdvisor;

public sealed record PortfolioResult
{
    public string OverallRiskProfile { get; init; } = "";
    public Recommendation[] Recommendations { get; init; } = [];
    public string Summary { get; init; } = "";
}

public sealed record Recommendation
{
    public string Action { get; init; } = "";
    public string Ticker { get; init; } = "";
    public double CurrentAllocationPct { get; init; }
    public double SuggestedAllocationPct { get; init; }
    public string Rationale { get; init; } = "";
}
