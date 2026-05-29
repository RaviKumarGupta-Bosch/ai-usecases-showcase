"""
MCP 03-Advanced — Multi-Server Setup and Architecture Patterns
===============================================================
Topics covered:
  1. Multi-server client architecture — connecting to N servers
  2. Tool namespacing across servers
  3. Error handling and per-server fallback
  4. Server composition patterns (proxy / aggregator)
  5. Authentication and security patterns
  6. Performance: connection pooling and caching
  7. Practical: AI assistant with multiple specialised MCP servers

This file demonstrates MCP advanced architecture patterns.
Concepts are illustrated with complete, runnable code examples.

Prerequisites:
  pip install mcp python-dotenv

Run:
  python 01_multi_server_setup.py
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()


# ── 1. Multi-server client architecture ──────────────────────────────────────
def explain_multi_server():
    print("\n=== 1. Multi-Server Client Architecture ===")
    print("""
A single MCP client can maintain simultaneous connections to multiple servers.

                    ┌─────────────────┐
                    │   AI Client     │
                    │  (LLM + Agent)  │
                    └────────┬────────┘
                             │  manages N connections
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
    │  File MCP   │  │  DB MCP     │  │  Web MCP     │
    │  Server     │  │  Server     │  │  Server      │
    │             │  │             │  │              │
    │ list_files  │  │ query_db    │  │ web_search   │
    │ read_file   │  │ list_tables │  │ fetch_url    │
    │ write_file  │  │ run_query   │  │ take_screenshot│
    └─────────────┘  └─────────────┘  └──────────────┘

Each server is independent — different transports, auth, capabilities.

Connection types:
  stdio  → spawn subprocess, communicate over stdin/stdout
  SSE    → HTTP Server-Sent Events (remote servers)
  websocket → bidirectional streaming (planned)
""")


# ── 2. Multi-server connection manager ───────────────────────────────────────
def demo_connection_manager():
    print("\n=== 2. Multi-Server Connection Manager ===")

    code = '''
import asyncio
import json
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

@dataclass
class ServerConfig:
    name: str
    transport: str            # "stdio" or "sse"
    command: list[str] = field(default_factory=list)   # for stdio
    url: str = ""             # for SSE
    env: dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0


class MultiServerClient:
    """Manages connections to multiple MCP servers."""

    def __init__(self, configs: list[ServerConfig]):
        self.configs = configs
        self._sessions: dict[str, ClientSession] = {}
        self._tools_index: dict[str, str] = {}   # tool_name → server_name

    async def connect_all(self):
        """Connect to all configured servers concurrently."""
        results = await asyncio.gather(
            *[self._connect(cfg) for cfg in self.configs],
            return_exceptions=True,
        )
        for cfg, result in zip(self.configs, results):
            if isinstance(result, Exception):
                print(f"[WARN] Failed to connect to {cfg.name!r}: {result}")
            else:
                print(f"[OK] Connected to {cfg.name!r}")

    async def _connect(self, cfg: ServerConfig):
        if cfg.transport == "stdio":
            params = StdioServerParameters(command=cfg.command[0], args=cfg.command[1:])
            read, write = await stdio_client(params).__aenter__()
        else:
            read, write = await sse_client(cfg.url).__aenter__()

        session = ClientSession(read, write)
        await session.initialize()
        self._sessions[cfg.name] = session

        # Index available tools
        tools = await session.list_tools()
        for tool in tools.tools:
            self._tools_index[tool.name] = cfg.name
            # Namespace: also register as "server_name.tool_name"
            self._tools_index[f"{cfg.name}.{tool.name}"] = cfg.name

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the appropriate server (auto-routes by name)."""
        server_name = self._tools_index.get(tool_name)
        if not server_name:
            raise ValueError(f"Unknown tool: {tool_name!r}")
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name.split(".")[-1], arguments)
        return result

    async def list_all_tools(self) -> list[dict]:
        """Aggregate tools from all servers with namespace prefix."""
        all_tools = []
        for server_name, session in self._sessions.items():
            tools = await session.list_tools()
            for t in tools.tools:
                all_tools.append({
                    "name": f"{server_name}.{t.name}",
                    "description": t.description,
                    "server": server_name,
                })
        return all_tools
'''
    print(code)


# ── 3. Tool namespacing ───────────────────────────────────────────────────────
def demo_tool_namespacing():
    print("\n=== 3. Tool Namespacing Across Servers ===")
    print("""
When multiple servers expose tools, name collisions can occur.
Namespacing prevents conflicts and makes tool origin clear.

Strategies:

  A) PREFIX strategy: "<server>.<tool>"
     files.read_file    db.read_file    →  both "read_file" coexist
     files.write_file   db.query        →  origin is always clear

  B) REGISTRY strategy: centrally map user-facing names to (server, tool)
     registry = {
         "search_code":   ("git_server",  "search"),
         "search_web":    ("web_server",  "search"),
         "search_docs":   ("docs_server", "search"),
     }

  C) CAPABILITY-BASED routing: route by what the tool does
     capability_map = {
         "file_ops":    "file_server",
         "db_queries":  "db_server",
         "web_access":  "browser_server",
     }
""")

    # Demonstrate a simple tool registry
    class ToolRegistry:
        def __init__(self):
            self._registry: dict[str, tuple[str, str]] = {}

        def register(self, namespace: str, tool_name: str, server: str, original_name: str):
            key = f"{namespace}.{tool_name}"
            self._registry[key] = (server, original_name)

        def resolve(self, tool_name: str) -> tuple[str, str] | None:
            """Returns (server_name, original_tool_name) or None."""
            return self._registry.get(tool_name)

        def list_tools(self) -> list[str]:
            return sorted(self._registry.keys())

    registry = ToolRegistry()
    registry.register("files", "read",   "file_server",    "read_file")
    registry.register("files", "write",  "file_server",    "write_file")
    registry.register("db",    "query",  "postgres_server","execute_query")
    registry.register("db",    "schema", "postgres_server","describe_table")
    registry.register("web",   "search", "browser_server", "web_search")
    registry.register("web",   "fetch",  "browser_server", "fetch_url")

    print("\n  Registry contents:")
    for name in registry.list_tools():
        server, orig = registry.resolve(name)
        print(f"    {name:<25} → {server}::{orig}")

    print(f"\n  Resolve 'db.query' → {registry.resolve('db.query')}")
    print(f"  Resolve 'web.search' → {registry.resolve('web.search')}")


# ── 4. Error handling and fallback ────────────────────────────────────────────
def demo_error_handling():
    print("\n=== 4. Error Handling and Fallback Patterns ===")

    code = '''
class ResilientMultiServerClient:
    """Client with per-server health tracking and fallbacks."""

    def __init__(self):
        self._sessions = {}
        self._health: dict[str, bool] = {}
        self._fallbacks: dict[str, list[str]] = {}  # server → ordered fallback list

    def register_fallback(self, primary: str, fallbacks: list[str]):
        self._fallbacks[primary] = fallbacks

    async def call_tool_with_fallback(
        self, tool_name: str, arguments: dict, servers: list[str]
    ) -> tuple[Any, str]:
        """
        Try each server in order; return first successful response.
        Returns (result, winning_server_name).
        """
        last_error = None
        for server in servers:
            if not self._health.get(server, True):
                continue   # skip known-unhealthy server
            try:
                result = await self._sessions[server].call_tool(tool_name, arguments)
                self._health[server] = True
                return result, server
            except asyncio.TimeoutError:
                self._health[server] = False
                last_error = f"{server}: timeout"
            except Exception as e:
                last_error = f"{server}: {e}"
        raise RuntimeError(f"All servers failed. Last error: {last_error}")

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        server = self._route(tool_name)
        fallback_chain = [server] + self._fallbacks.get(server, [])
        result, winner = await self.call_tool_with_fallback(tool_name, arguments, fallback_chain)
        return result

    async def health_check_all(self):
        for server, session in self._sessions.items():
            try:
                await asyncio.wait_for(session.ping(), timeout=2.0)
                self._health[server] = True
            except Exception:
                self._health[server] = False
        return {s: ("healthy" if h else "UNHEALTHY") for s, h in self._health.items()}
'''
    print(code)

    # Demonstrate error classification
    error_categories = {
        "ConnectionError":   "Server unreachable — mark unhealthy, try fallback",
        "TimeoutError":      "Slow server — mark unhealthy, circuit-break for N seconds",
        "McpError (code 404)": "Tool not found — update tool registry",
        "McpError (code 500)": "Server internal error — may retry, log for debugging",
        "ValidationError":   "Bad arguments from AI — fix prompt, do NOT retry as-is",
    }
    print("\n  Error classification guide:")
    for err, action in error_categories.items():
        print(f"    {err:<30} → {action}")


# ── 5. Server composition patterns ───────────────────────────────────────────
def demo_composition():
    print("\n=== 5. Server Composition Patterns ===")
    print("""
Three main patterns for composing MCP servers:

──────────────────────────────────────────────────────────
Pattern A: PROXY SERVER
  Wraps an upstream server — adds auth, rate-limiting, caching.

  Client → Proxy MCP → Upstream MCP

  from mcp.server.fastmcp import FastMCP
  from mcp import ClientSession

  proxy = FastMCP("Proxy")
  upstream: ClientSession = ...  # connection to real server

  @proxy.tool()
  async def read_file(path: str) -> str:
      # Add auth check before forwarding
      if not is_allowed(path):
          raise PermissionError(f"Access denied: {path}")
      result = await upstream.call_tool("read_file", {"path": path})
      return result.content[0].text

──────────────────────────────────────────────────────────
Pattern B: AGGREGATOR SERVER
  Exposes tools from multiple upstreams under one connection.

  Client → Aggregator MCP → Server A
                           → Server B
                           → Server C

  aggregator = FastMCP("Aggregator")
  file_session, db_session, web_session = ...

  @aggregator.tool()
  async def search(query: str) -> list[dict]:
      # Fan-out to all servers
      results = await asyncio.gather(
          db_session.call_tool("db_search",  {"query": query}),
          web_session.call_tool("web_search", {"query": query}),
      )
      return merge_and_rank(results)

──────────────────────────────────────────────────────────
Pattern C: ROUTER SERVER
  Routes tool calls to specialised servers based on content.

  @router.tool()
  async def query(input: str) -> str:
      intent = classify_intent(input)   # "db" | "files" | "web"
      target = {"db": db_session, "files": file_session, "web": web_session}
      return await target[intent].call_tool("query", {"input": input})
""")


# ── 6. Authentication and security ────────────────────────────────────────────
def demo_auth_security():
    print("\n=== 6. Authentication and Security Patterns ===")
    print("""
MCP servers should enforce authentication and authorisation.

──────────────────────────────────────────────────────────
A) API KEY authentication (SSE transport):

  from mcp.server.fastmcp import FastMCP
  from starlette.requests import Request
  from starlette.middleware.base import BaseHTTPMiddleware

  class ApiKeyMiddleware(BaseHTTPMiddleware):
      VALID_KEYS = {os.getenv("MCP_API_KEY")}
      async def dispatch(self, request, call_next):
          key = request.headers.get("X-API-Key")
          if key not in self.VALID_KEYS:
              return Response("Unauthorized", status_code=401)
          return await call_next(request)

  mcp = FastMCP("Secure Server")
  app = mcp.sse_app()
  app.add_middleware(ApiKeyMiddleware)

──────────────────────────────────────────────────────────
B) Per-context authorisation (tool-level):

  @mcp.tool()
  async def delete_file(ctx: Context, path: str) -> str:
      user_role = ctx.request_context.meta.get("role", "read-only")
      if user_role not in ("admin", "write"):
          raise PermissionError("Deletion requires admin role")
      os.unlink(path)
      return f"Deleted {path}"

──────────────────────────────────────────────────────────
C) Input sanitisation:

  import re
  SAFE_PATH = re.compile(r'^[a-zA-Z0-9/_.-]+$')

  @mcp.tool()
  def read_file(path: str) -> str:
      # Prevent path traversal
      if ".." in path or not SAFE_PATH.match(path):
          raise ValueError("Invalid path")
      resolved = (BASE_DIR / path).resolve()
      if not resolved.is_relative_to(BASE_DIR):
          raise PermissionError("Path outside allowed directory")
      return resolved.read_text()

──────────────────────────────────────────────────────────
D) Rate limiting:

  from collections import defaultdict
  import time

  _call_counts: dict[str, list[float]] = defaultdict(list)

  def rate_limit(client_id: str, limit: int = 10, window: float = 60.0):
      now = time.time()
      calls = [t for t in _call_counts[client_id] if now - t < window]
      _call_counts[client_id] = calls
      if len(calls) >= limit:
          raise RuntimeError(f"Rate limit exceeded ({limit}/{window}s)")
      _call_counts[client_id].append(now)
""")


# ── 7. Practical: multi-server AI assistant ───────────────────────────────────
def demo_multi_server_assistant():
    print("\n=== 7. Practical: Multi-Server AI Assistant Architecture ===")

    architecture = '''
"""
multi_server_assistant.py
─────────────────────────
An AI assistant backed by 4 specialised MCP servers.

Servers:
  1. file_server.py   — file I/O, directory listing, code reading
  2. db_server.py     — PostgreSQL queries, schema inspection
  3. web_server.py    — web search, URL fetching, screenshot
  4. code_server.py   — code execution (sandboxed), linting, formatting
"""

import asyncio
import json
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS = {
    "files": ["python", "file_server.py"],
    "db":    ["python", "db_server.py"],
    "web":   ["python", "web_server.py"],
    "code":  ["python", "code_server.py"],
}

class Assistant:
    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tools: list[dict] = []
        self.client = openai.AsyncOpenAI()

    async def start(self):
        """Connect to all MCP servers and build unified tool list."""
        for name, cmd in SERVERS.items():
            params = StdioServerParameters(command=cmd[0], args=cmd[1:])
            read, write = await stdio_client(params).__aenter__()
            session = ClientSession(read, write)
            await session.initialize()
            self.sessions[name] = session

            # Collect and namespace tools
            tools = await session.list_tools()
            for t in tools.tools:
                self.tools.append({
                    "type": "function",
                    "function": {
                        "name":        f"{name}__{t.name}",   # "files__read_file"
                        "description": f"[{name}] {t.description}",
                        "parameters":  t.inputSchema,
                    }
                })
        print(f"Ready: {len(self.tools)} tools across {len(self.sessions)} servers")

    async def chat(self, user_message: str) -> str:
        messages = [{"role": "user", "content": user_message}]

        while True:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return msg.content   # final answer

            # Execute all tool calls
            for tc in msg.tool_calls:
                server, tool = tc.function.name.split("__", 1)
                args = json.loads(tc.function.arguments)
                try:
                    result = await self.sessions[server].call_tool(tool, args)
                    content = result.content[0].text
                except Exception as e:
                    content = f"Error: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })

async def main():
    assistant = Assistant()
    await assistant.start()
    answer = await assistant.chat(
        "Search the codebase for async functions, then show me today\'s DB schema."
    )
    print("Assistant:", answer)

if __name__ == "__main__":
    asyncio.run(main())
'''
    for line in architecture.strip().split("\n"):
        print(f"  {line}")


if __name__ == "__main__":
    print("MCP 03-Advanced — Multi-Server Setup and Architecture Patterns")
    print("=" * 62)
    explain_multi_server()
    demo_connection_manager()
    demo_tool_namespacing()
    demo_error_handling()
    demo_composition()
    demo_auth_security()
    demo_multi_server_assistant()
