"""
MCP Basics 02 — Building a Simple MCP Server with FastMCP
===========================================================
Topics covered:
  1. Creating an MCP server with FastMCP
  2. Registering tools (functions the LLM can call)
  3. Registering resources (data the LLM can read)
  4. Registering prompt templates
  5. Running the server via stdio and SSE transports

This server can be connected to by any MCP client including:
  - Claude Desktop
  - The MCP client in 03_client_connection.py
  - LangChain/LlamaIndex MCP adapters

Run (stdio mode — for testing with client):
  python 02_simple_server.py

Run (SSE mode — for HTTP connections):
  python 02_simple_server.py --sse
"""

import math
import json
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ── Create the MCP server ─────────────────────────────────────────────────────
mcp = FastMCP(
    name="TutorialServer",
    version="1.0.0",
    description="A tutorial MCP server with math, search, and text tools",
)


# ── Tool definitions ──────────────────────────────────────────────────────────

@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e, abs, round
    Examples: 'sqrt(144)', '2**10 + 1', 'sin(pi/4)'
    """
    try:
        safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_globals.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, safe_globals)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_current_time() -> str:
    """Return the current UTC date and time."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


@mcp.tool()
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about a topic.
    Returns a brief summary of the topic.
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return f"Title: {data.get('title')}\n\n{data.get('extract', 'No summary available.')[:600]}"
        return f"No Wikipedia article found for '{query}'"
    except Exception as e:
        return f"Search failed: {e}"


@mcp.tool()
def analyse_text(text: str) -> str:
    """
    Analyse text and return statistics: word count, sentence count, average word length, top words.
    """
    words = text.lower().split()
    sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
    avg_len = sum(len(w.strip(".,!?;:")) for w in words) / max(len(words), 1)

    stop = {"the", "a", "an", "is", "in", "on", "and", "or", "of", "to", "it", "for"}
    freq: dict[str, int] = {}
    for w in words:
        clean = w.strip(".,!?;:")
        if clean not in stop and clean:
            freq[clean] = freq.get(clean, 0) + 1

    top5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    return json.dumps({
        "word_count":      len(words),
        "sentence_count":  sentences,
        "avg_word_length": round(avg_len, 1),
        "top_words":       dict(top5),
    })


@mcp.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    Supported conversions:
      - Temperature: celsius↔fahrenheit, celsius↔kelvin
      - Length: meters↔feet, km↔miles
      - Weight: kg↔pounds
    """
    conversions: dict[tuple[str, str], float | None] = {
        ("celsius",    "fahrenheit"): None,   # special formula
        ("fahrenheit", "celsius"):    None,
        ("celsius",    "kelvin"):     None,
        ("kelvin",     "celsius"):    None,
        ("meters",     "feet"):       3.28084,
        ("feet",       "meters"):     0.30480,
        ("km",         "miles"):      0.62137,
        ("miles",      "km"):         1.60934,
        ("kg",         "pounds"):     2.20462,
        ("pounds",     "kg"):         0.45359,
    }

    key = (from_unit.lower(), to_unit.lower())

    if key == ("celsius", "fahrenheit"):
        result = value * 9/5 + 32
    elif key == ("fahrenheit", "celsius"):
        result = (value - 32) * 5/9
    elif key == ("celsius", "kelvin"):
        result = value + 273.15
    elif key == ("kelvin", "celsius"):
        result = value - 273.15
    elif key in conversions and conversions[key] is not None:
        factor = conversions[key]
        assert isinstance(factor, float)
        result = value * factor
    else:
        return f"Conversion from {from_unit} to {to_unit} not supported"

    return f"{value} {from_unit} = {result:.4f} {to_unit}"


# ── Resource definitions ──────────────────────────────────────────────────────

@mcp.resource("info://server/capabilities")
def server_capabilities() -> str:
    """Returns a description of what this server can do."""
    return """# TutorialServer Capabilities

## Tools Available
- **calculate**: Evaluate mathematical expressions
- **get_current_time**: Return the current UTC time  
- **search_wikipedia**: Search Wikipedia for any topic
- **analyse_text**: Get statistics about a piece of text
- **convert_units**: Convert between measurement units

## Use Cases
This server is ideal for:
1. Mathematical calculations during conversations
2. Looking up factual information
3. Analysing text content
4. Unit conversions in technical contexts
"""


@mcp.resource("data://examples/math-problems")
def math_problems() -> str:
    """A set of sample math problems for testing the calculate tool."""
    return json.dumps([
        {"problem": "Area of a circle with radius 5", "expression": "pi * 5**2"},
        {"problem": "Compound interest: $1000 at 5% for 10 years", "expression": "1000 * (1 + 0.05)**10"},
        {"problem": "Hypotenuse of 3-4 right triangle", "expression": "sqrt(3**2 + 4**2)"},
        {"problem": "Log base 2 of 1024", "expression": "log2(1024)"},
    ], indent=2)


# ── Prompt definitions ────────────────────────────────────────────────────────

@mcp.prompt()
def explain_concept(concept: str, audience: str = "beginner") -> str:
    """Generate a prompt to explain a concept at a given level."""
    return f"""Explain '{concept}' to a {audience} audience.
Structure your explanation as:
1. Simple one-sentence definition
2. Analogy that makes it intuitive
3. Two concrete real-world examples
4. One common misconception to avoid

Keep the entire explanation under 200 words."""


@mcp.prompt()
def debug_code(code: str, error_message: str, language: str = "Python") -> str:
    """Generate a prompt to debug code with an error."""
    return f"""Debug this {language} code that produces an error.

**Code:**
```{language.lower()}
{code}
```

**Error:**
```
{error_message}
```

1. Identify the root cause of the error
2. Explain why it occurs
3. Provide the corrected code
4. Suggest how to avoid this error in the future"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--sse" in sys.argv:
        print("Starting MCP server in SSE mode at http://localhost:8000")
        mcp.run(transport="sse")
    else:
        print("Starting MCP server in stdio mode (for subprocess clients)")
        mcp.run(transport="stdio")
