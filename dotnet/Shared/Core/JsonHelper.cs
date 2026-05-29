using System.Text.Json;

namespace AIUsecasesShowcase.Core;

/// <summary>Strips markdown code fences and deserialises AI JSON responses.</summary>
public static class JsonHelper
{
    public static T? Parse<T>(string raw)
    {
        var text = raw.Trim();
        if (text.Contains("```json"))
            text = text.Split("```json")[1].Split("```")[0];
        else if (text.Contains("```"))
            text = text.Split("```")[1].Split("```")[0];

        return JsonSerializer.Deserialize<T>(text.Trim(),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
    }
}
