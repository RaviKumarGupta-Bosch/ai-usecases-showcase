namespace FraudDetection;

public sealed record TransactionRecord
{
    public string TransactionId { get; init; } = "";
    public string UserId { get; init; } = "";
    public double AmountUsd { get; init; }
    public string MerchantCategory { get; init; } = "";
    public string LocationCountry { get; init; } = "";
    public string UserHomeCountry { get; init; } = "";
    public int HourOfDay { get; init; }
    public int IsOnline { get; init; }
    public double Prev30dAvgUsd { get; init; }
}
