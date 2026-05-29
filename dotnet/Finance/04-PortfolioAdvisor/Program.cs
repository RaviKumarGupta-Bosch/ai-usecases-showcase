using AIUsecasesShowcase.Core;
using AIUsecasesShowcase.Data;
using PortfolioAdvisor;

var ai = AIServiceFactory.Create();
var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToUpper();

Console.WriteLine("\ud83d\udcbc Portfolio Advisor");
Console.WriteLine($"   AI Provider : {provider}");

var dataFile = Path.Combine(AppContext.BaseDirectory, "data", "sample_portfolio.json");
var portfolio = JsonLoader.Load<Portfolio>(dataFile);

var total = portfolio.Holdings.Sum(h => h.ValueUsd);
Console.WriteLine($"   Investor Profile : {portfolio.InvestorProfile.RiskTolerance.ToUpper()} risk | {portfolio.InvestorProfile.HorizonYears}-year horizon");
Console.WriteLine($"   Portfolio Total  : ${total:N0}");
Console.WriteLine(new string('=', 60));

var prompt = PromptBuilder.Build(portfolio);
var raw = await ai.CompleteAsync(prompt);
var result = JsonHelper.Parse<PortfolioResult>(raw);

if (result is null)
{
    Console.WriteLine("Could not parse AI response.");
    return;
}

Console.WriteLine($"\nOverall Risk Profile : {result.OverallRiskProfile}");
Console.WriteLine("\nRebalancing Recommendations:");

var actionEmoji = new Dictionary<string, string>
{
    ["REDUCE"]   = "\ud83d\udd3b",
    ["INCREASE"] = "\ud83d\udd3a",
    ["ADD"]      = "\u2795",
    ["REMOVE"]   = "\u2796",
    ["HOLD"]     = "\u23f8\ufe0f"
};

for (int i = 0; i < result.Recommendations.Length; i++)
{
    var rec = result.Recommendations[i];
    var emoji = actionEmoji.GetValueOrDefault(rec.Action, "\u25b6\ufe0f");
    Console.WriteLine($"  {i + 1}. {emoji} {rec.Action,-8} {rec.Ticker,-6} ({rec.CurrentAllocationPct:F0}% -> {rec.SuggestedAllocationPct:F0}%) - {rec.Rationale}");
}

Console.WriteLine($"\nSummary: {result.Summary}");
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("\u2705 Advice generated.");
