# 🔮 Future: .NET / C# Implementations

This section outlines the plan to port all AI use-cases in this repository
into **.NET 8 / C#** using Microsoft's AI ecosystem.

> **Status:** Roadmap / Coming Soon

---

## Why .NET?

- Enterprise-grade, strongly-typed, high-performance
- Native integration with **Azure OpenAI Service**
- **Microsoft Semantic Kernel** provides a clean abstraction layer for LLMs
- **ML.NET** supports local model inference without cloud dependencies
- Ideal for teams already building microservices or APIs in .NET

---

## Planned Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| AI Orchestration | [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel) | Prompt management, chaining, memory |
| Cloud AI | [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service) | GPT-4o via Azure |
| Local AI | [Ollama .NET client](https://github.com/awaescher/OllamaSharp) | LLaMA 3 locally via Ollama |
| Local ML | [ML.NET](https://dotnet.microsoft.com/en-us/apps/machinelearning-ai/ml-dotnet) | Classical ML (regression, classification) |
| HTTP | `HttpClient` / `IHttpClientFactory` | REST API calls |
| Data | `CsvHelper`, `System.Text.Json` | CSV and JSON parsing |
| Output | [Spectre.Console](https://spectreconsole.net/) | Colour-coded terminal output |

---

## Planned Project Structure

```
dotnet-future/
│
├── AIUsecasesShowcase.sln          # Solution file
│
├── Shared/
│   ├── AIUsecasesShowcase.Core/       # Shared AI client abstractions
│   └── AIUsecasesShowcase.Data/       # CSV / JSON loaders
│
├── Manufacturing/
│   ├── PredictiveMaintenance/         # Use-case 01
│   ├── QualityControl/                # Use-case 02
│   ├── DemandForecasting/             # Use-case 03
│   └── AnomalyDetection/              # Use-case 04
│
└── Finance/
    ├── FraudDetection/                # Use-case 01
    ├── SentimentAnalysis/             # Use-case 02
    ├── LoanRiskAssessment/            # Use-case 03
    └── PortfolioAdvisor/              # Use-case 04
```

---

## Architecture Pattern (per use-case)

Each .NET use-case will follow this clean architecture pattern:

```
UseCaseName/
├── Program.cs                     # Entry point: load data, call service, display output
├── Models/
│   └── InputRecord.cs             # Strongly-typed input model
├── Services/
│   ├── IAIService.cs              # Interface for AI provider
│   ├── OpenAIService.cs           # Azure OpenAI implementation
│   └── OllamaService.cs           # Local Ollama implementation
├── Prompts/
│   └── AnalysisPrompt.cs          # Prompt templates
└── Data/
    └── sample_data.csv            # Same sample data as Python version
```

---

## Code Preview — AI Client Interface

All use-cases will share a common AI abstraction:

```csharp
// Shared/AIUsecasesShowcase.Core/IAIService.cs
namespace AIUsecasesShowcase.Core;

public interface IAIService
{
    /// <summary>Send a prompt and get a response string.</summary>
    Task<string> CompleteAsync(string prompt, CancellationToken ct = default);
}
```

```csharp
// Shared/AIUsecasesShowcase.Core/OpenAIService.cs
using Azure.AI.OpenAI;

public class OpenAIService : IAIService
{
    private readonly OpenAIClient _client;
    private readonly string _model;

    public OpenAIService(string endpoint, string apiKey, string model = "gpt-4o-mini")
    {
        _client = new OpenAIClient(new Uri(endpoint), new AzureKeyCredential(apiKey));
        _model = model;
    }

    public async Task<string> CompleteAsync(string prompt, CancellationToken ct = default)
    {
        var options = new ChatCompletionsOptions
        {
            Messages = { new ChatRequestUserMessage(prompt) },
            Temperature = 0.2f,
        };
        var response = await _client.GetChatCompletionsAsync(_model, options, ct);
        return response.Value.Choices[0].Message.Content;
    }
}
```

---

## Code Preview — Predictive Maintenance (C#)

```csharp
// Manufacturing/PredictiveMaintenance/Program.cs
using AIUsecasesShowcase.Core;
using CsvHelper;

var aiService = AiServiceFactory.Create(); // reads AI_PROVIDER env var
var records = LoadSensorData("data/sample_sensor_data.csv");

Console.WriteLine("\ud83c\udfed Predictive Maintenance Analysis");

foreach (var sensor in records)
{
    var prompt = PromptBuilder.ForSensor(sensor);
    var rawResponse = await aiService.CompleteAsync(prompt);
    var result = JsonSerializer.Deserialize<MaintenanceResult>(rawResponse);

    Console.WriteLine($"  Machine: {sensor.MachineId}");
    Console.WriteLine($"  Risk   : {result.RiskLevel}");
    Console.WriteLine($"  Action : {result.Recommendation}");
}
```

---

## Semantic Kernel Integration

For more advanced use-cases, [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)
will be used as an orchestration layer:

```csharp
using Microsoft.SemanticKernel;

var kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion(
        deploymentName: "gpt-4o-mini",
        endpoint: Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!,
        apiKey: Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY")!)
    .Build();

// Prompt as an inline function
var analyse = kernel.CreateFunctionFromPrompt(
    """Analyse this machine sensor: {{$sensor_data}}
       Reply with JSON: {risk_level, recommendation}"""
);

var result = await kernel.InvokeAsync(analyse,
    new KernelArguments {{ "sensor_data", sensorJson }});

Console.WriteLine(result.GetValue<string>());
```

---

## Getting Started (once implemented)

```bash
# Prerequisites: .NET 8 SDK, Azure OpenAI key or Ollama running

cd dotnet-future
dotnet restore

# Set environment variables
set AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
set AZURE_OPENAI_KEY=your-key

# Run a use-case
cd Manufacturing/PredictiveMaintenance
dotnet run
```

---

## Contribution Welcome!

If you'd like to help build the .NET version, see [CONTRIBUTING.md](../CONTRIBUTING.md)
and open a Pull Request with a use-case implementation.

The Python version serves as the specification — match its input format, output structure,
and AI prompt design.
