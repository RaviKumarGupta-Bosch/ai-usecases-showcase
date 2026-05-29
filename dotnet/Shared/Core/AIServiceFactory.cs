namespace AIUsecasesShowcase.Core;

/// <summary>Creates the correct AI service based on the AI_PROVIDER environment variable.</summary>
public static class AIServiceFactory
{
    public static IAIService Create()
    {
        var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToLowerInvariant();
        return provider switch
        {
            "ollama" => new OllamaService(),
            _ => new OpenAIService()
        };
    }
}
