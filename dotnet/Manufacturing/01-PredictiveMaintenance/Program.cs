using AIUsecasesShowcase.Core;
using AIUsecasesShowcase.Data;
using PredictiveMaintenance;

var ai = AIServiceFactory.Create();
var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToUpper();

Console.WriteLine("\u2699\ufe0f  Predictive Maintenance Analysis");
Console.WriteLine($"   AI Provider : {provider}");
Console.WriteLine(new string('=', 60));

var dataFile = Path.Combine(AppContext.BaseDirectory, "data", "sample_sensor_data.csv");
var sensors = CsvLoader.Load<SensorRecord>(dataFile);
Console.WriteLine($"   Loaded {sensors.Count} machine records.\n");

foreach (var sensor in sensors)
{
    var prompt = PromptBuilder.Build(sensor);
    var raw = await ai.CompleteAsync(prompt);
    var result = JsonHelper.Parse<MaintenanceResult>(raw);

    if (result is null) { Console.WriteLine($"  [{sensor.MachineId}] Could not parse response."); continue; }

    var emoji = result.RiskLevel switch
    {
        "CRITICAL" => "\ud83d\udd34",
        "HIGH"     => "\ud83d\udfe0",
        "MEDIUM"   => "\ud83d\udfe1",
        _          => "\ud83d\udfe2"
    };

    Console.WriteLine($"  Machine : {sensor.MachineId}");
    Console.WriteLine($"  Risk    : {emoji} {result.RiskLevel}  (failure in ~{result.PredictedFailureDays} days)");
    Console.WriteLine($"  Action  : {result.Recommendation}");
    Console.WriteLine($"  Reason  : {result.Reasoning}");
    Console.WriteLine();
}

Console.WriteLine(new string('=', 60));
Console.WriteLine("\u2705 Analysis complete.");
