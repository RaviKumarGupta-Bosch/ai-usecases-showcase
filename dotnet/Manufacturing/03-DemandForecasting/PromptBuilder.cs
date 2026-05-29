using System.Text;

namespace DemandForecasting;

public static class PromptBuilder
{
    public static string Build(string product, List<DemandRecord> history)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Month       | Units Sold | Avg Price | Promo");
        sb.AppendLine(new string('-', 50));
        foreach (var r in history)
            sb.AppendLine($"{r.Month,-12}| {r.UnitsSold,10} | {r.AvgPrice,9:F2} | {(r.IsPromoMonth == 1 ? "Yes" : "No")}");

        return $"""
            You are a demand forecasting AI for a manufacturing company.
            Analyse the following 12-month sales history for {product} and forecast the next 3 months.

            {sb}

            Respond ONLY with a JSON object:
            {{
              "trend_summary": "<one sentence>",
              "forecast": [
                {{"month": "Month+1", "units": <int>}},
                {{"month": "Month+2", "units": <int>}},
                {{"month": "Month+3", "units": <int>}}
              ],
              "key_risks": ["<risk1>", "<risk2>"],
              "production_recommendation": "<short recommendation>"
            }}
            """;
    }
}
