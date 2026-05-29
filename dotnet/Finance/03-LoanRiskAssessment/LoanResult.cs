namespace LoanRiskAssessment;

public sealed record LoanResult
{
    public string RiskRating { get; init; } = "";
    public string Decision { get; init; } = "";
    public double? SuggestedInterestRatePct { get; init; }
    public string KeyFactors { get; init; } = "";
    public string Reasoning { get; init; } = "";
}
