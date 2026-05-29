namespace AIUsecasesShowcase.Core;

/// <summary>Common AI backend abstraction used by every use-case.</summary>
public interface IAIService
{
    Task<string> CompleteAsync(string prompt, CancellationToken ct = default);
}
