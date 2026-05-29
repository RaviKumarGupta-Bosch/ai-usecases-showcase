using Azure;
using Azure.AI.OpenAI;

namespace AIUsecasesShowcase.Core;

/// <summary>Calls Azure OpenAI (GPT-4o-mini by default).</summary>
public sealed class OpenAIService : IAIService
{
    private readonly OpenAIClient _client;
    private readonly string _deployment;

    public OpenAIService()
    {
        var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")
            ?? throw new InvalidOperationException("Set AZURE_OPENAI_ENDPOINT");
        var key = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY")
            ?? throw new InvalidOperationException("Set AZURE_OPENAI_KEY");
        _deployment = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "gpt-4o-mini";
        _client = new OpenAIClient(new Uri(endpoint), new AzureKeyCredential(key));
    }

    public async Task<string> CompleteAsync(string prompt, CancellationToken ct = default)
    {
        var options = new ChatCompletionsOptions
        {
            DeploymentName = _deployment,
            Messages = { new ChatRequestUserMessage(prompt) },
            Temperature = 0.1f,
        };
        var response = await _client.GetChatCompletionsAsync(options, ct);
        return response.Value.Choices[0].Message.Content;
    }
}
