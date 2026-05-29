using System.Text.Json.Serialization;

namespace QualityControl;

public sealed record BatchRecord
{
    [JsonPropertyName("batch_id")]
    public string BatchId { get; init; } = "";

    [JsonPropertyName("measurements")]
    public Dictionary<string, MeasurementPair> Measurements { get; init; } = new();

    [JsonPropertyName("visual_notes")]
    public string VisualNotes { get; init; } = "";
}

public sealed record MeasurementPair
{
    [JsonPropertyName("actual")]
    public double Actual { get; init; }

    [JsonPropertyName("spec")]
    public double Spec { get; init; }
}
