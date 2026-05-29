using AIUsecasesShowcase.Core;
using AIUsecasesShowcase.Data;
using QualityControl;

var ai = AIServiceFactory.Create();
var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToUpper();

Console.WriteLine("\ud83d\udd0d Quality Control Inspector");
Console.WriteLine($"   AI Provider : {provider}");
Console.WriteLine(new string('=', 60));

var dataFile = Path.Combine(AppContext.BaseDirectory, "data", "sample_inspections.json");
var batches = JsonLoader.Load<List<BatchRecord>>(dataFile);
Console.WriteLine($"   Loaded {batches.Count} batches.\n");

int passed = 0, failed = 0;
foreach (var batch in batches)
{
    var prompt = PromptBuilder.Build(batch);
    var raw = await ai.CompleteAsync(prompt);
    var result = JsonHelper.Parse<QCResult>(raw);

    if (result is null) { Console.WriteLine($"  [{batch.BatchId}] Could not parse response."); continue; }

    var emoji = result.Verdict == "PASS" ? "\u2705" : "\u274c";
    Console.WriteLine($"  Batch     : {batch.BatchId}");
    Console.WriteLine($"  Verdict   : {emoji} {result.Verdict}  (confidence: {result.ConfidencePct}%)");
    Console.WriteLine($"  Issues    : {(result.IssuesFound.Length > 0 ? string.Join("; ", result.IssuesFound) : "None")}");
    Console.WriteLine($"  Action    : {result.RecommendedAction}");
    Console.WriteLine();

    if (result.Verdict == "PASS") passed++; else failed++;
}

Console.WriteLine(new string('=', 60));
Console.WriteLine($"\u2705 Inspection complete. Passed: {passed} | Failed: {failed}");
