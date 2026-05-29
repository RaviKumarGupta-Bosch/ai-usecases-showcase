using System.Text;

namespace PortfolioAdvisor;

public static class PromptBuilder
{
    public static string Build(Portfolio portfolio)
    {
        var profile = portfolio.InvestorProfile;
        var total = portfolio.Holdings.Sum(h => h.ValueUsd);

        var sb = new StringBuilder();
        sb.AppendLine("Ticker | Name                     | Class        | Value (USD) | Alloc%");
        sb.AppendLine(new string('-', 72));
        foreach (var h in portfolio.Holdings)
            sb.AppendLine($"{h.Ticker,-6} | {h.Name,-24} | {h.AssetClass,-12} | ${h.ValueUsd,10:N0} | {h.AllocationPct,5:F1}%");

        return $"""
            You are a professional portfolio advisor AI.
            Review the following investment portfolio and provide rebalancing recommendations.

            Investor Profile:
              Risk Tolerance       : {profile.RiskTolerance}
              Investment Horizon   : {profile.HorizonYears} years
              Total Portfolio Value: ${total:N0}

            Current Holdings:
            {sb}

            Respond ONLY with a JSON object:
            {{
              "overall_risk_profile": "<brief assessment>",
              "recommendations": [
                {{
                  "action": "REDUCE | INCREASE | ADD | REMOVE | HOLD",
                  "ticker": "<symbol>",
                  "current_allocation_pct": <number>,
                  "suggested_allocation_pct": <number>,
                  "rationale": "<one sentence>"
                }}
              ],
              "summary": "<two to three sentences>"
            }}
            """;
    }
}
