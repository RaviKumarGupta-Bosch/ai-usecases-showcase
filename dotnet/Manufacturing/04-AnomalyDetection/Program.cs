using AIUsecasesShowcase.Core;
using AIUsecasesShowcase.Data;
using AnomalyDetection;

const double AnomalyThresholdStd = 2.0;

var ai = AIServiceFactory.Create();
var provider = (Environment.GetEnvironmentVariable("AI_PROVIDER") ?? "openai").ToUpper();

Console.WriteLine("\ud83d\udea8 Anomaly Detection System");
Console.WriteLine($"   AI Provider      : {provider}");
Console.WriteLine($"   Threshold (stdev): {AnomalyThresholdStd}");
Console.WriteLine(new string('=', 60));

var dataFile = Path.Combine(AppContext.BaseDirectory, "data", "sample_production_data.csv");
var rows = CsvLoader.Load<ProductionRecord>(dataFile);
Console.WriteLine($"   Loaded {rows.Count} records.\n");

// Build statistical baseline from first half of data per line
var baselines = rows
    .GroupBy(r => r.LineId)
    .ToDictionary(
        g => g.Key,
        g =>
        {
            var half = g.Take(g.Count() / 2).ToList();
            return new Baseline(
                CycleMean: half.Average(r => r.CycleTimeSec),
                CycleStd:  StdDev(half.Select(r => r.CycleTimeSec)),
                RejectMean: half.Average(r => r.RejectRatePct),
                RejectStd:  StdDev(half.Select(r => r.RejectRatePct))
            );
        });

int anomalyCount = 0;
foreach (var row in rows)
{
    var b = baselines[row.LineId];
    bool cycleAnomaly  = Math.Abs(row.CycleTimeSec  - b.CycleMean)  > AnomalyThresholdStd * b.CycleStd;
    bool rejectAnomaly = Math.Abs(row.RejectRatePct - b.RejectMean) > AnomalyThresholdStd * b.RejectStd;

    if (!cycleAnomaly && !rejectAnomaly) continue;

    anomalyCount++;
    Console.WriteLine($"  \u26a0\ufe0f  Anomaly detected: {row.LineId} @ {row.Timestamp}");
    Console.WriteLine($"      CycleTime={row.CycleTimeSec}s (mean={b.CycleMean:F1}, std={b.CycleStd:F1})");
    Console.WriteLine($"      RejectRate={row.RejectRatePct}% (mean={b.RejectMean:F2}, std={b.RejectStd:F2})");

    var prompt = PromptBuilder.Build(row, b);
    var raw = await ai.CompleteAsync(prompt);
    var result = JsonHelper.Parse<AnomalyResult>(raw);

    if (result is not null)
    {
        Console.WriteLine($"      LikelyCause : {result.LikelyCause}");
        Console.WriteLine($"      Urgency     : {result.Urgency}");
        Console.WriteLine($"      Action      : {result.RecommendedAction}");
    }
    Console.WriteLine();
}

Console.WriteLine(new string('=', 60));
Console.WriteLine($"\u2705 Detection complete. {anomalyCount} anomalies found.");

static double StdDev(IEnumerable<double> values)
{
    var list = values.ToList();
    var mean = list.Average();
    return Math.Sqrt(list.Sum(v => Math.Pow(v - mean, 2)) / list.Count);
}
