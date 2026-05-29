namespace AnomalyDetection;

public static class PromptBuilder
{
    public static string Build(ProductionRecord row, Baseline baseline) => $$"""
        You are a manufacturing anomaly detection AI.
        An anomaly was flagged on production line {{row.LineId}} at {{row.Timestamp}}.

        Observed values:
          Cycle Time : {{row.CycleTimeSec}} sec  (baseline mean: {{baseline.CycleMean:F1}}, stdev: {{baseline.CycleStd:F1}})
          Reject Rate: {{row.RejectRatePct}}%     (baseline mean: {{baseline.RejectMean:F2}}, stdev: {{baseline.RejectStd:F2}})
          Utilisation: {{row.UtilisationPct}}%
          Energy     : {{row.EnergyKwh}} kWh

        Respond ONLY with a JSON object:
        {
          "likely_cause": "<short description>",
          "urgency": "LOW | MEDIUM | HIGH | CRITICAL",
          "recommended_action": "<short action>"
        }
        """;
}
