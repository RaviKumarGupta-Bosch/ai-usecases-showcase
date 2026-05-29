"""
MCP Basics 03 — Connecting a Client to an MCP Server
=====================================================
Topics covered:
  1. Connecting to an MCP server via stdio
  2. Listing available tools, resources, and prompts
  3. Calling tools and reading resources
  4. Using an LLM with MCP tools (OpenAI function calling)
  5. Building a simple MCP-powered chat loop

Prerequisites:
  - 02_simple_server.py must be available for subprocess
  - OPENAI_API_KEY in .env

Run:
  python 03_client_connection.py
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "02_simple_server.py")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── 1. List all server capabilities ──────────────────────────────────────────
async def demo_list_capabilities():
    print("\n=== 1. List Server Capabilities ===")

    server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools_result = await session.list_tools()
            print(f"\nTools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"  • {tool.name}: {tool.description}")

            # List resources
            resources_result = await session.list_resources()
            print(f"\nResources ({len(resources_result.resources)}):")
            for r in resources_result.resources:
                print(f"  • {r.uri}: {r.name}")

            # List prompts
            prompts_result = await session.list_prompts()
            print(f"\nPrompts ({len(prompts_result.prompts)}):")
            for p in prompts_result.prompts:
                print(f"  • {p.name}: {p.description}")


# ── 2. Call tools directly ────────────────────────────────────────────────────
async def demo_call_tools():
    print("\n=== 2. Calling MCP Tools Directly ===")

    server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call calculate
            result = await session.call_tool("calculate", {"expression": "sqrt(2) * pi"})
            print(f"\ncalculate('sqrt(2) * pi') = {result.content[0].text}")

            # Call get_current_time
            result = await session.call_tool("get_current_time", {})
            print(f"get_current_time() = {result.content[0].text}")

            # Call search_wikipedia
            result = await session.call_tool("search_wikipedia", {"query": "Python programming language"})
            print(f"search_wikipedia('Python...') =\n{result.content[0].text[:200]}...")

            # Call convert_units
            result = await session.call_tool("convert_units", {
                "value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"
            })
            print(f"convert_units(100, C→F) = {result.content[0].text}")


# ── 3. Read resources ─────────────────────────────────────────────────────────
async def demo_read_resources():
    print("\n=== 3. Reading MCP Resources ===")

    server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Read capabilities resource
            result = await session.read_resource("info://server/capabilities")
            print("\nServer capabilities resource:")
            print(result.contents[0].text[:300])

            # Read math problems resource
            result = await session.read_resource("data://examples/math-problems")
            problems = json.loads(result.contents[0].text)
            print(f"\nMath problems resource ({len(problems)} problems):")
            for p in problems:
                print(f"  {p['problem']} → {p['expression']}")


# ── 4. Get prompt templates ───────────────────────────────────────────────────
async def demo_get_prompts():
    print("\n=== 4. Getting MCP Prompt Templates ===")

    server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get explain_concept prompt
            result = await session.get_prompt("explain_concept", {
                "concept": "recursion",
                "audience": "intermediate",
            })
            print("\nexplain_concept prompt:")
            print(result.messages[0].content.text)


# ── 5. LLM + MCP tools (OpenAI function calling) ─────────────────────────────
async def demo_llm_with_mcp():
    print("\n=== 5. LLM with MCP Tools ===")

    server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get tool schemas from MCP server
            tools_result = await session.list_tools()

            # Convert MCP tool schemas to OpenAI function format
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name":        tool.name,
                        "description": tool.description,
                        "parameters":  tool.inputSchema,
                    },
                }
                for tool in tools_result.tools
            ]

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
                {"role": "user",   "content": "What is 15% of 250, and what time is it now? Also convert 37 Celsius to Fahrenheit."},
            ]

            print(f"\nUser: {messages[-1]['content']}")

            # LLM decides which tools to call
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            # Process tool calls
            while response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls
                messages.append(response.choices[0].message)

                for tc in tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    print(f"\n  [Tool call] {fn_name}({fn_args})")

                    # Execute tool via MCP
                    result = await session.call_tool(fn_name, fn_args)
                    tool_output = result.content[0].text
                    print(f"  [Tool result] {tool_output}")

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      tool_output,
                    })

                # Get next LLM response
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )

            print(f"\nAssistant: {response.choices[0].message.content}")


async def main():
    await demo_list_capabilities()
    await demo_call_tools()
    await demo_read_resources()
    await demo_get_prompts()
    await demo_llm_with_mcp()


if __name__ == "__main__":
    asyncio.run(main())
