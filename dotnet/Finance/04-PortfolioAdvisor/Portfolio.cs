using System.Text.Json.Serialization;

namespace PortfolioAdvisor;

public sealed record Portfolio
{
    [JsonPropertyName("investor_profile")]
    public InvestorProfile InvestorProfile { get; init; } = new();

    [JsonPropertyName("holdings")]
    public List<Holding> Holdings { get; init; } = new();
}

public sealed record InvestorProfile
{
    [JsonPropertyName("risk_tolerance")]
    public string RiskTolerance { get; init; } = "";

    [JsonPropertyName("horizon_years")]
    public int HorizonYears { get; init; }
}

public sealed record Holding
{
    [JsonPropertyName("ticker")]
    public string Ticker { get; init; } = "";

    [JsonPropertyName("name")]
    public string Name { get; init; } = "";

    [JsonPropertyName("asset_class")]
    public string AssetClass { get; init; } = "";

    [JsonPropertyName("value_usd")]
    public double ValueUsd { get; init; }

    [JsonPropertyName("allocation_pct")]
    public double AllocationPct { get; init; }
}
