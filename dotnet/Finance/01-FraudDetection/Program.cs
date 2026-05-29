using AIUsecasesShowcase.Core;
using AIUsecasesShowcase.Data;
using FraudDetection;

var ai = AIServiceFactory.Create();
var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToUpper();

Console.WriteLine("\ud83d\udd75\ufe0f Fraud Detection System");
Console.WriteLine($"   AI Provider : {provider}");
Console.WriteLine(new string('=', 60));

var dataFile = Path.Combine(AppContext.BaseDirectory, "data", "sample_transactions.csv");
var transactions = CsvLoader.Load<TransactionRecord>(dataFile);
Console.WriteLine($"   Loaded {transactions.Count} transactions.\n");

int highRisk = 0;
foreach (var txn in transactions)
{
    var prompt = PromptBuilder.Build(txn);
    var raw = await ai.CompleteAsync(prompt);
    var result = JsonHelper.Parse<FraudResult>(raw);

    if (result is null) { Console.WriteLine($"  [{txn.TransactionId}] Could not parse response."); continue; }

    var emoji = result.RiskLevel switch
    {
        "HIGH"   => "\ud83d\udd34",
        "MEDIUM" => "\ud83d\udfe1",
        _        => "\ud83d\udfe2"
    };

    Console.WriteLine($"  Transaction : {txn.TransactionId}  |  User: {txn.UserId}  |  Amount: ${txn.AmountUsd:N2}");
    Console.WriteLine($"  Risk        : {emoji} {result.RiskLevel}  (fraud probability: {result.FraudProbabilityPct}%)");
    Console.WriteLine($"  Red Flags   : {(result.RedFlags.Length > 0 ? string.Join("; ", result.RedFlags) : "None")}");
    Console.WriteLine($"  Action      : {result.RecommendedAction}");
    Console.WriteLine();

    if (result.RiskLevel == "HIGH") highRisk++;
}

Console.WriteLine(new string('=', 60));
Console.WriteLine($"\u2705 Screening complete. {highRisk}/{transactions.Count} transactions flagged HIGH risk.");
