namespace PredictiveMaintenance;

public static class PromptBuilder
{
    public static string Build(SensorRecord s) => $$"""
        You are a predictive maintenance AI for industrial machinery.
        Analyse the following sensor readings and predict maintenance needs.

        Machine ID       : {{s.MachineId}}
        Temperature (C)  : {{s.Temperature}}
        Vibration (mm/s) : {{s.Vibration}}
        Pressure (bar)   : {{s.Pressure}}
        Operating Hours  : {{s.OperatingHours}}

        Typical safe ranges: temperature < 85 C, vibration < 7 mm/s, pressure 2-8 bar.

        Respond ONLY with a JSON object:
        {
          "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
          "predicted_failure_days": <integer>,
          "recommendation": "<short action>",
          "reasoning": "<one sentence>"
        }
        """;
}
