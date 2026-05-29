using System.Net.Http.Json;
using System.Text.Json;

namespace AIUsecasesShowcase.Core;

/// <summary>Calls a local Ollama instance (llama3 by default).</summary>
public sealed class OllamaService : IAIService
{
    private readonly HttpClient _http;
    private readonly string _model;
    private readonly string _baseUrl;

    public OllamaService()
    {
        _baseUrl = Environment.GetEnvironmentVariable("OLLAMA_URL") ?? "http://localhost:11434";
        _model   = Environment.GetEnvironmentVariable("OLLAMA_MODEL") ?? "llama3";
        _http    = new HttpClient { Timeout = TimeSpan.FromSeconds(180) };
    }

    public async Task<string> CompleteAsync(string prompt, CancellationToken ct = default)
    {
        var payload = new { model = _model, prompt, stream = false };
        using var response = await _http.PostAsJsonAsync($"{_baseUrl}/api/generate", payload, ct);
        response.EnsureSuccessStatusCode();
        var json = await JsonDocument.ParseAsync(await response.Content.ReadAsStreamAsync(ct), cancellationToken: ct);
        return json.RootElement.GetProperty("response").GetString() ?? string.Empty;
    }
}
