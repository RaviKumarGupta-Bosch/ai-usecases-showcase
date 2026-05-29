namespace LoanRiskAssessment;

public sealed record LoanApplication
{
    public string ApplicationId { get; init; } = "";
    public int Age { get; init; }
    public double AnnualIncomeUsd { get; init; }
    public double LoanAmountUsd { get; init; }
    public string LoanPurpose { get; init; } = "";
    public int CreditScore { get; init; }
    public double ExistingDebtUsd { get; init; }
    public int EmploymentYears { get; init; }
    public int HasCollateral { get; init; }
}
