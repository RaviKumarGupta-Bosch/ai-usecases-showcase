namespace PredictiveMaintenance;

public sealed record SensorRecord
{
    public string MachineId { get; init; } = "";
    public double Temperature { get; init; }
    public double Vibration { get; init; }
    public double Pressure { get; init; }
    public int OperatingHours { get; init; }
}
