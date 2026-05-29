namespace DemandForecasting;

public sealed record DemandRecord
{
    public string Month { get; init; } = "";
    public string Product { get; init; } = "";
    public int UnitsSold { get; init; }
    public double AvgPrice { get; init; }
    public int IsPromoMonth { get; init; }
}
