using AIUsecasesShowcase.Core;
using AIUsecasesShowcase.Data;
using DemandForecasting;

var ai = AIServiceFactory.Create();
var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToUpper();

Console.WriteLine("\ud83d\udcc8 Demand Forecasting");
Console.WriteLine($"   AI Provider : {provider}");
Console.WriteLine(new string('=', 60));

var dataFile = Path.Combine(AppContext.BaseDirectory, "data", "sample_demand_data.csv");
var rows = CsvLoader.Load<DemandRecord>(dataFile);

var byProduct = rows.GroupBy(r => r.Product);

foreach (var group in byProduct)
{
    Console.WriteLine($"\nProduct: {group.Key}");
    var prompt = PromptBuilder.Build(group.Key, group.ToList());
    var raw = await ai.CompleteAsync(prompt);
    var result = JsonHelper.Parse<ForecastResult>(raw);

    if (result is null) { Console.WriteLine("  Could not parse response."); continue; }

    Console.WriteLine($"  Trend    : {result.TrendSummary}");
    Console.WriteLine($"  Forecast : {string.Join(", ", result.Forecast.Select(f => $"{f.Month}: {f.Units} units"))}");
    Console.WriteLine($"  Risks    : {string.Join("; ", result.KeyRisks)}");
    Console.WriteLine($"  Action   : {result.ProductionRecommendation}");
}

Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("\u2705 Forecasting complete.");
