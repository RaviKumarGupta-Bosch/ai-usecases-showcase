namespace FraudDetection;

public static class PromptBuilder
{
    public static string Build(TransactionRecord t) => $$"""
        You are a fraud detection AI for a financial institution.
        Analyse the following transaction and assess the likelihood of fraud.

        Transaction ID       : {{t.TransactionId}}
        User ID              : {{t.UserId}}
        Amount               : ${{t.AmountUsd:N2}}
        Merchant Category    : {{t.MerchantCategory}}
        Transaction Country  : {{t.LocationCountry}}
        User Home Country    : {{t.UserHomeCountry}}
        Hour of Day          : {{t.HourOfDay}}:00
        Channel              : {{(t.IsOnline == 1 ? "Online" : "In-Person")}}
        User 30-day Avg Spend: ${{t.Prev30dAvgUsd:N2}}

        Respond ONLY with a JSON object:
        {
          "fraud_probability_pct": <0-100>,
          "risk_level": "LOW | MEDIUM | HIGH",
          "red_flags": ["<flag1>", "<flag2>"],
          "recommended_action": "APPROVE | FLAG_FOR_REVIEW | BLOCK",
          "reasoning": "<one or two sentences>"
        }
        """;
}
