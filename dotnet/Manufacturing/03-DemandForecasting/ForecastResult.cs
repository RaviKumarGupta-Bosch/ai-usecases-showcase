using System.Text.Json.Serialization;

namespace DemandForecasting;

public sealed record ForecastResult
{
    public string TrendSummary { get; init; } = "";
    public ForecastMonth[] Forecast { get; init; } = [];
    public string[] KeyRisks { get; init; } = [];
    public string ProductionRecommendation { get; init; } = "";
}

public sealed record ForecastMonth
{
    [JsonPropertyName("month")]
    public string Month { get; init; } = "";
    [JsonPropertyName("units")]
    public int Units { get; init; }
}
