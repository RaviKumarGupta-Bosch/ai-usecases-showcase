using AIUsecasesShowcase.Core;
using AIUsecasesShowcase.Data;
using SentimentAnalysis;

var ai = AIServiceFactory.Create();
var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToUpper();

Console.WriteLine("\ud83d\udcf0 Financial Sentiment Analyser");
Console.WriteLine($"   AI Provider : {provider}");
Console.WriteLine(new string('=', 60));

var dataFile = Path.Combine(AppContext.BaseDirectory, "data", "sample_news.json");
var articles = JsonLoader.Load<List<NewsArticle>>(dataFile);
Console.WriteLine($"   Loaded {articles.Count} articles.\n");

var counts = new Dictionary<string, int> { ["BULLISH"] = 0, ["BEARISH"] = 0, ["NEUTRAL"] = 0 };

foreach (var article in articles)
{
    var prompt = PromptBuilder.Build(article);
    var raw = await ai.CompleteAsync(prompt);
    var result = JsonHelper.Parse<SentimentResult>(raw);

    if (result is null) { Console.WriteLine($"  [{article.Id}] Could not parse response."); continue; }

    var emoji = result.Sentiment switch
    {
        "BULLISH" => "\ud83d\udfe2",
        "BEARISH" => "\ud83d\udd34",
        _         => "\u26aa"
    };

    Console.WriteLine($"  [{article.Id}]  {article.PublishedAt}");
    Console.WriteLine($"  Headline   : \"{article.Headline}\"");
    Console.WriteLine($"  Sentiment  : {emoji} {result.Sentiment}  (confidence: {result.ConfidencePct}%)");
    Console.WriteLine($"  Affects    : {result.AffectedAssets}");
    Console.WriteLine($"  Impact     : {result.MarketImpactNote}");
    Console.WriteLine();

    if (counts.ContainsKey(result.Sentiment)) counts[result.Sentiment]++;
}

Console.WriteLine(new string('=', 60));
Console.WriteLine($"\u2705 Analysis complete. Bullish: {counts["BULLISH"]} | Bearish: {counts["BEARISH"]} | Neutral: {counts["NEUTRAL"]}");
