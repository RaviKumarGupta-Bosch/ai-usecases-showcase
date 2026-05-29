using System.Text.Json.Serialization;

namespace SentimentAnalysis;

public sealed record NewsArticle
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = "";
    [JsonPropertyName("headline")]
    public string Headline { get; init; } = "";
    [JsonPropertyName("source")]
    public string Source { get; init; } = "";
    [JsonPropertyName("published_at")]
    public string PublishedAt { get; init; } = "";
}
