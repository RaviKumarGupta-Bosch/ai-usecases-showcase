"""
MCP Use Case 01 — File System MCP Server
=========================================
Topics covered:
  1. Building a file system MCP server
  2. Safe sandboxed file operations
  3. File read/write/search/list tools
  4. File metadata as resources
  5. Using the server with an LLM for file-based Q&A

Run the server (stdio mode):
  python 01_filesystem_server.py --server

Run the client demo:
  python 01_filesystem_server.py --client

Run both (demo mode — spawns server as subprocess):
  python 01_filesystem_server.py
"""

import os
import sys
import json
import glob
import asyncio
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Server mode ───────────────────────────────────────────────────────────────
def run_server():
    """Run as an MCP server exposing file system tools."""
    from mcp.server.fastmcp import FastMCP

    # Sandbox: only operate within a temp working directory
    SANDBOX = Path(tempfile.mkdtemp(prefix="mcp_fs_"))
    print(f"[Server] Sandbox directory: {SANDBOX}", file=sys.stderr)

    # Pre-populate sandbox with sample files
    (SANDBOX / "notes.txt").write_text("Meeting notes:\n- Discussed Q4 targets\n- Assigned action items\n- Next meeting Friday")
    (SANDBOX / "data.json").write_text(json.dumps({"users": 142, "revenue": 58400, "growth": "12%"}))
    (SANDBOX / "report.md").write_text("# Q4 Report\n\n## Summary\nExceeded targets by 8%.\n\n## Details\nSee data.json for numbers.")
    (SANDBOX / "tasks.txt").write_text("TODO:\n[x] Deploy v2\n[ ] Write tests\n[ ] Update docs\n[ ] Review PRs")

    mcp = FastMCP(name="FileSystemServer", description="Safe sandboxed file system access")

    def _safe_path(filename: str) -> Path:
        """Ensure the path is within the sandbox."""
        p = (SANDBOX / filename).resolve()
        if not str(p).startswith(str(SANDBOX)):
            raise ValueError(f"Access denied: '{filename}' is outside sandbox")
        return p

    @mcp.tool()
    def list_files(pattern: str = "*") -> str:
        """List files in the workspace. Use patterns like '*.txt', '*.json', or '*' for all."""
        matches = glob.glob(str(SANDBOX / pattern))
        files = []
        for path in matches:
            p = Path(path)
            if p.is_file():
                stat = p.stat()
                files.append({
                    "name":     p.name,
                    "size":     stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
        return json.dumps(files, indent=2) if files else "No files found"

    @mcp.tool()
    def read_file(filename: str) -> str:
        """Read the contents of a file by name."""
        try:
            path = _safe_path(filename)
            if not path.exists():
                return f"Error: File '{filename}' does not exist"
            return path.read_text(encoding="utf-8")
        except ValueError as e:
            return f"Error: {e}"

    @mcp.tool()
    def write_file(filename: str, content: str) -> str:
        """Write content to a file (creates it if it doesn't exist)."""
        try:
            path = _safe_path(filename)
            path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to '{filename}'"
        except ValueError as e:
            return f"Error: {e}"

    @mcp.tool()
    def append_to_file(filename: str, content: str) -> str:
        """Append content to an existing file."""
        try:
            path = _safe_path(filename)
            if not path.exists():
                return f"Error: File '{filename}' does not exist. Use write_file to create it."
            with path.open("a", encoding="utf-8") as f:
                f.write("\n" + content)
            return f"Appended {len(content)} characters to '{filename}'"
        except ValueError as e:
            return f"Error: {e}"

    @mcp.tool()
    def search_files(query: str, extension: str = "") -> str:
        """Search file contents for a query string. Optionally filter by extension (e.g. '.txt')."""
        results = []
        pattern = f"*{extension}" if extension else "*"
        for path in SANDBOX.glob(pattern):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
                if query.lower() in text.lower():
                    # Find matching lines
                    lines = [
                        f"  Line {i+1}: {line.strip()}"
                        for i, line in enumerate(text.split("\n"))
                        if query.lower() in line.lower()
                    ]
                    results.append(f"{path.name}:\n" + "\n".join(lines[:3]))
            except Exception:
                continue
        return "\n\n".join(results) if results else f"No files contain '{query}'"

    @mcp.tool()
    def get_file_info(filename: str) -> str:
        """Get detailed metadata about a file."""
        try:
            path = _safe_path(filename)
            if not path.exists():
                return f"Error: File '{filename}' does not exist"
            stat = path.stat()
            content = path.read_text(encoding="utf-8")
            return json.dumps({
                "name":       path.name,
                "extension":  path.suffix,
                "size_bytes": stat.st_size,
                "lines":      len(content.splitlines()),
                "words":      len(content.split()),
                "created":    datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified":   datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "md5":        hashlib.md5(content.encode()).hexdigest(),
            }, indent=2)
        except ValueError as e:
            return f"Error: {e}"

    @mcp.resource("workspace://files")
    def workspace_index() -> str:
        """Index of all files in the workspace."""
        files = list(SANDBOX.glob("*"))
        index = [
            {"name": f.name, "type": "file" if f.is_file() else "dir"}
            for f in sorted(files)
        ]
        return json.dumps(index, indent=2)

    mcp.run(transport="stdio")


# ── Client demo ───────────────────────────────────────────────────────────────
async def run_client():
    """Connect to the filesystem server and perform file-based Q&A with an LLM."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    print("\n=== File System MCP Server Demo ===")
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    server_params = StdioServerParameters(command="python", args=[__file__, "--server"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get tools
            tools_result = await session.list_tools()
            openai_tools = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}}
                for t in tools_result.tools
            ]

            questions = [
                "What files are in the workspace?",
                "What are the contents of the report file?",
                "Search for any files mentioning targets or revenue",
                "Create a new file called summary.txt with a one-line summary of what you found",
            ]

            messages = [{"role": "system", "content": "You are a file system assistant. Use the provided tools to answer questions about files."}]

            for question in questions:
                print(f"\n--- Q: {question} ---")
                messages.append({"role": "user", "content": question})

                response = openai.chat.completions.create(
                    model="gpt-4o-mini", messages=messages, tools=openai_tools, tool_choice="auto"
                )

                while response.choices[0].message.tool_calls:
                    messages.append(response.choices[0].message)
                    for tc in response.choices[0].message.tool_calls:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)
                        print(f"  [Tool] {fn_name}({fn_args})")
                        result = await session.call_tool(fn_name, fn_args)
                        output = result.content[0].text
                        print(f"  [Result] {output[:150]}{'...' if len(output) > 150 else ''}")
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini", messages=messages, tools=openai_tools, tool_choice="auto"
                    )

                answer = response.choices[0].message.content
                messages.append({"role": "assistant", "content": answer})
                print(f"A: {answer}")


if __name__ == "__main__":
    if "--server" in sys.argv:
        run_server()
    elif "--client" in sys.argv:
        asyncio.run(run_client())
    else:
        # Default: run client (which spawns the server automatically)
        asyncio.run(run_client())
