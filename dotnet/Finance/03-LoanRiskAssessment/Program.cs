using AIUsecasesShowcase.Core;
using AIUsecasesShowcase.Data;
using LoanRiskAssessment;

var ai = AIServiceFactory.Create();
var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToUpper();

Console.WriteLine("\ud83c\udfe6 Loan Risk Assessment System");
Console.WriteLine($"   AI Provider : {provider}");
Console.WriteLine(new string('=', 60));

var dataFile = Path.Combine(AppContext.BaseDirectory, "data", "sample_loan_applications.csv");
var applications = CsvLoader.Load<LoanApplication>(dataFile);
Console.WriteLine($"   Loaded {applications.Count} applications.\n");

int approved = 0, rejected = 0;
foreach (var app in applications)
{
    var prompt = PromptBuilder.Build(app);
    var raw = await ai.CompleteAsync(prompt);
    var result = JsonHelper.Parse<LoanResult>(raw);

    if (result is null) { Console.WriteLine($"  [{app.ApplicationId}] Could not parse response."); continue; }

    var emoji = result.RiskRating switch
    {
        "VERY_HIGH" => "\ud83d\udd34",
        "HIGH"      => "\ud83d\udfe0",
        "MEDIUM"    => "\ud83d\udfe1",
        _           => "\ud83d\udfe2"
    };
    var rate = result.SuggestedInterestRatePct.HasValue ? $"{result.SuggestedInterestRatePct}%" : "N/A";

    Console.WriteLine($"  Application : {app.ApplicationId}  |  Loan: ${app.LoanAmountUsd:N0}  |  Purpose: {app.LoanPurpose}");
    Console.WriteLine($"  Risk Rating : {emoji} {result.RiskRating}");
    Console.WriteLine($"  Decision    : {result.Decision}");
    Console.WriteLine($"  Rate        : {rate}");
    Console.WriteLine($"  Key Factors : {result.KeyFactors}");
    Console.WriteLine();

    if (result.Decision == "REJECT") rejected++;
    else approved++;
}

Console.WriteLine(new string('=', 60));
Console.WriteLine($"\u2705 Assessment complete. Approved: {approved} | Rejected: {rejected}");
