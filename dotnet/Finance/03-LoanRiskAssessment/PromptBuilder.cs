namespace LoanRiskAssessment;

public static class PromptBuilder
{
    public static string Build(LoanApplication app)
    {
        double dti = (app.ExistingDebtUsd / app.AnnualIncomeUsd) * 100;
        return $"""
            You are a credit risk AI for a financial institution.
            Evaluate the following loan application.

            Application ID    : {app.ApplicationId}
            Applicant Age     : {app.Age}
            Annual Income     : ${app.AnnualIncomeUsd:N0}
            Loan Amount       : ${app.LoanAmountUsd:N0}
            Loan Purpose      : {app.LoanPurpose}
            Credit Score      : {app.CreditScore} / 850
            Existing Debt     : ${app.ExistingDebtUsd:N0}  (DTI: {dti:F0}%)
            Employment        : {app.EmploymentYears} years
            Has Collateral    : {(app.HasCollateral == 1 ? "Yes" : "No")}

            Respond ONLY with a JSON object:
            {{
              "risk_rating": "LOW | MEDIUM | HIGH | VERY_HIGH",
              "decision": "APPROVE | APPROVE_WITH_CONDITIONS | REJECT",
              "suggested_interest_rate_pct": <float or null>,
              "key_factors": "<two or three key factors>",
              "reasoning": "<one or two sentences>"
            }}
            """;
    }
}
