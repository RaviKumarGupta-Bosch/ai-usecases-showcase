"""
MCP Basics 01 — MCP Concepts and Architecture
==============================================
Topics covered:
  1. What is MCP and the problem it solves
  2. MCP primitives: Tools, Resources, Prompts
  3. MCP transport layers: stdio and SSE
  4. Client-server architecture overview
  5. When to use MCP vs direct API calls

This file is conceptual — it explains MCP through code-based examples
without requiring a running server.

Run:
  python 01_mcp_concepts.py
"""

import json
from typing import Any


# ── 1. MCP primitives as plain Python structures ──────────────────────────────
def explain_mcp_primitives():
    print("\n=== 1. MCP Primitives ===")
    print("""
MCP defines 3 primitives that servers can expose:

┌──────────────────────────────────────────────────────────────────────┐
│ TOOL       │ A function the AI model can call to take action         │
│            │ Examples: search web, run SQL, send email, calculate    │
├──────────────────────────────────────────────────────────────────────┤
│ RESOURCE   │ Data the AI model can read                              │
│            │ Examples: file contents, DB rows, API results, configs  │
├──────────────────────────────────────────────────────────────────────┤
│ PROMPT     │ Reusable prompt template with named parameters           │
│            │ Examples: code review template, summarisation prompt    │
└──────────────────────────────────────────────────────────────────────┘
""")

    # What a Tool definition looks like (JSON Schema format)
    tool_definition = {
        "name": "search_docs",
        "description": "Search the documentation for a query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Max results", "default": 5},
            },
            "required": ["query"],
        },
    }
    print("Tool definition example:")
    print(json.dumps(tool_definition, indent=2))

    # What a Resource definition looks like
    resource_definition = {
        "uri":         "file:///project/README.md",
        "name":        "Project README",
        "description": "The main README file for this project",
        "mimeType":    "text/markdown",
    }
    print("\nResource definition example:")
    print(json.dumps(resource_definition, indent=2))

    # What a Prompt definition looks like
    prompt_definition = {
        "name":        "code_review",
        "description": "Review code for quality, bugs, and best practices",
        "arguments": [
            {"name": "code",     "description": "The code to review", "required": True},
            {"name": "language", "description": "Programming language", "required": False},
        ],
    }
    print("\nPrompt definition example:")
    print(json.dumps(prompt_definition, indent=2))


# ── 2. MCP transport layers ───────────────────────────────────────────────────
def explain_transport_layers():
    print("\n=== 2. MCP Transport Layers ===")
    print("""
MCP supports two transport mechanisms:

┌─────────┬──────────────────────────────────────────────────────────┐
│ STDIO   │ Communication via stdin/stdout                           │
│         │ Best for: local tools, CLI integrations                   │
│         │ The client spawns the server as a subprocess              │
├─────────┬──────────────────────────────────────────────────────────┤
│ SSE     │ Server-Sent Events over HTTP                             │
│         │ Best for: remote servers, web deployments                │
│         │ Server runs independently, client connects via URL        │
└─────────┴──────────────────────────────────────────────────────────┘

Stdio example (client config):
{
    "command": "python",
    "args": ["my_mcp_server.py"],
    "env": {"API_KEY": "..."}
}

SSE example (client config):
{
    "url": "http://localhost:8000/sse"
}
""")


# ── 3. MCP vs direct API calls ────────────────────────────────────────────────
def explain_mcp_vs_direct():
    print("\n=== 3. MCP vs Direct API Calls ===")
    print("""
Without MCP (direct function calling):
  ┌────────┐  hardcoded tools  ┌─────┐
  │  LLM   │ ────────────────► │Tools│
  └────────┘                   └─────┘
  • Each AI app reimplements tool integration
  • Tools are tightly coupled to the application
  • No standardised discovery or protocol

With MCP:
  ┌────────┐  MCP protocol  ┌────────────┐  implements  ┌──────────┐
  │  LLM   │ ─────────────► │ MCP Client │ ────────────► │MCP Server│
  └────────┘                └────────────┘               └──────────┘
                                                            (Tools +
                                                           Resources +
                                                            Prompts)
  • Standardised protocol — any MCP client works with any MCP server
  • Tools are decoupled from AI application
  • Servers can be shared, published, reused
  • Dynamic capability discovery
  
Real-world analogy: MCP is to AI tools what USB is to peripherals —
a standard interface so any device works with any computer.
""")


# ── 4. Simulated MCP message flow ─────────────────────────────────────────────
def demonstrate_message_flow():
    print("\n=== 4. MCP Message Flow (Simulated) ===")

    # Simulate the JSON-RPC messages MCP uses internally
    messages = [
        {
            "description": "1. Client initialises connection",
            "direction": "client → server",
            "message": {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "my-app", "version": "1.0"},
                },
            },
        },
        {
            "description": "2. Client requests list of tools",
            "direction": "client → server",
            "message": {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        },
        {
            "description": "3. Server returns available tools",
            "direction": "server → client",
            "message": {
                "jsonrpc": "2.0", "id": 2, "result": {
                    "tools": [
                        {"name": "calculate", "description": "Evaluate math expressions"},
                        {"name": "search",    "description": "Search the web"},
                    ],
                },
            },
        },
        {
            "description": "4. Client calls a tool",
            "direction": "client → server",
            "message": {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "calculate", "arguments": {"expression": "sqrt(144)"}},
            },
        },
        {
            "description": "5. Server returns tool result",
            "direction": "server → client",
            "message": {
                "jsonrpc": "2.0", "id": 3, "result": {
                    "content": [{"type": "text", "text": "12.0"}],
                    "isError": False,
                },
            },
        },
    ]

    for msg in messages:
        print(f"\n  {msg['description']} ({msg['direction']})")
        print(f"  {json.dumps(msg['message'], indent=4)}")


if __name__ == "__main__":
    explain_mcp_primitives()
    explain_transport_layers()
    explain_mcp_vs_direct()
    demonstrate_message_flow()
