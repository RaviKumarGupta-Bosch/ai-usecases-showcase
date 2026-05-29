using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

namespace AIUsecasesShowcase.Data;

/// <summary>Generic CSV loader using CsvHelper.</summary>
public static class CsvLoader
{
    public static List<T> Load<T>(string path)
    {
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            PrepareHeaderForMatch = args => args.Header.Trim().ToLowerInvariant()
        };
        using var reader = new StreamReader(path);
        using var csv = new CsvReader(reader, config);
        return csv.GetRecords<T>().ToList();
    }
}
