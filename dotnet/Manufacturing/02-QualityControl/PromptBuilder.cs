using System.Text;

namespace QualityControl;

public static class PromptBuilder
{
    public static string Build(BatchRecord batch)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Batch ID: {batch.BatchId}");
        sb.AppendLine("Measurements (actual vs spec):");
        foreach (var (name, pair) in batch.Measurements)
            sb.AppendLine($"  {name}: actual={pair.Actual}, spec={pair.Spec}");
        sb.AppendLine($"Visual Notes: {batch.VisualNotes}");

        return $$"""
            You are a manufacturing quality control AI.
            Inspect the following batch measurements and determine PASS or FAIL.

            {{sb}}

            Respond ONLY with a JSON object:
            {
              "verdict": "PASS | FAIL",
              "confidence_pct": <0-100>,
              "issues_found": ["<issue1>", "<issue2>"],
              "recommended_action": "<short action>"
            }
            """;
    }
}
