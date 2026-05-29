namespace AnomalyDetection;

public sealed record ProductionRecord
{
    public string Timestamp { get; init; } = "";
    public string LineId { get; init; } = "";
    public double CycleTimeSec { get; init; }
    public double RejectRatePct { get; init; }
    public double UtilisationPct { get; init; }
    public double EnergyKwh { get; init; }
}

public sealed record Baseline(
    double CycleMean, double CycleStd,
    double RejectMean, double RejectStd);
