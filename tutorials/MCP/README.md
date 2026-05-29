# MCP Tutorial — Model Context Protocol

MCP (Model Context Protocol) is an open standard by Anthropic that defines how AI models
communicate with external tools, resources, and systems. This tutorial covers building
MCP servers (exposing tools/resources) and MCP clients (connecting AI models to them).

## Curriculum

```
01-basics/
  01_mcp_concepts.py          — MCP architecture: servers, tools, resources, prompts
  02_simple_server.py         — Build a minimal MCP server with FastMCP
  03_client_connection.py     — Connect a client to an MCP server

02-intermediate/
  01_tools.py                 — Exposing tools via MCP (with type validation)
  02_resources.py             — Exposing resources (files, databases, APIs)
  03_prompts.py               — Reusable prompt templates via MCP

03-advanced/
  01_claude_integration.py    — Connect Claude to an MCP server
  02_multi_server.py          — Aggregating multiple MCP servers
  03_auth_and_security.py     — MCP authentication and security patterns

04-UseCases/
  01_filesystem_server.py     — File system MCP server (read/write/search files)
  02_database_server.py       — Database query MCP server (SQLite)
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Start a server in one terminal:
python 01-basics/02_simple_server.py
# Connect from another terminal:
python 01-basics/03_client_connection.py
```

## MCP Primitives

| Primitive | Purpose |
|-----------|---------|
| **Tool** | Function the model can call (e.g., search, calculate) |
| **Resource** | Data the model can read (files, DB rows, API results) |
| **Prompt** | Reusable prompt template with parameters |

## Prerequisites
- Python 3.10+
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
