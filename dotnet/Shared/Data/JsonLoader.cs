using System.Text.Json;

namespace AIUsecasesShowcase.Data;

/// <summary>Generic JSON file loader.</summary>
public static class JsonLoader
{
    public static T Load<T>(string path)
    {
        var text = File.ReadAllText(path);
        return JsonSerializer.Deserialize<T>(text,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidDataException($"Failed to deserialise {path}");
    }
}
