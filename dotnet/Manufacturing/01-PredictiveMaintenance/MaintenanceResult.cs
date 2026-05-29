namespace PredictiveMaintenance;

public sealed record MaintenanceResult
{
    public string RiskLevel { get; init; } = "";
    public int PredictedFailureDays { get; init; }
    public string Recommendation { get; init; } = "";
    public string Reasoning { get; init; } = "";
}
