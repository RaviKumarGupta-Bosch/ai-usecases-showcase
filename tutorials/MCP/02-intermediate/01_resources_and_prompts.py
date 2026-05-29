"""
MCP 02-Intermediate — Resources, Prompts & Sampling
=====================================================
Topics covered:
  1. MCP Resources — static and dynamic content
  2. Resource URI templates (parameterised resources)
  3. MCP Prompt templates — reusable prompt patterns
  4. Prompt arguments and interpolation
  5. Resource subscriptions and change notifications
  6. Sampling requests — server-initiated LLM calls
  7. Practical: document knowledge-base server with resources and prompts

This file demonstrates MCP intermediate concepts with both
explanatory code and a runnable FastMCP server example.

Prerequisites:
  pip install mcp python-dotenv

Run:
  python 01_resources_and_prompts.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()


# ── 1. MCP Resources — what they are ─────────────────────────────────────────
def explain_resources():
    print("\n=== 1. MCP Resources ===")
    print("""
Resources expose DATA to the AI model (read-only, like GET endpoints).

┌─────────────────────────────────────────────────────────────────────┐
│ Resource URI          │ Type     │ Description                      │
├─────────────────────────────────────────────────────────────────────┤
│ docs://readme         │ static   │ Always returns the same content  │
│ db://users/{id}       │ dynamic  │ Fetches live data per request    │
│ file://logs/today     │ dynamic  │ Current date's log file          │
│ metrics://cpu         │ dynamic  │ Real-time system metrics         │
└─────────────────────────────────────────────────────────────────────┘

Resources differ from Tools:
  Tool     → AI calls to perform an ACTION (side effects OK)
  Resource → AI reads DATA (should be idempotent, no side effects)

Resource content types:
  text/plain           → plain text documents
  application/json     → structured data
  text/markdown        → formatted documentation
  application/pdf      → binary (base64 encoded)
""")


# ── 2. Implementing resources in FastMCP ─────────────────────────────────────
def demo_resource_server():
    print("\n=== 2. Resource Server Implementation ===")

    server_code = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Knowledge Base Server")

# ── Static resource — same content every time ──────────────────────────────
@mcp.resource("docs://readme")
def get_readme() -> str:
    """Project README always available to the model."""
    return """
# AI Project
This project demonstrates RAG patterns.
## Architecture
- Vector store: Chroma
- Embeddings: text-embedding-3-small
- LLM: gpt-4o-mini
"""

# ── Dynamic resource — computed on every access ────────────────────────────
@mcp.resource("metrics://status")
def get_status() -> str:
    """Live system status."""
    import json, datetime
    return json.dumps({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "status": "healthy",
        "queue_depth": 42,
        "cache_hit_rate": 0.87,
    })

# ── Resource with MIME type ────────────────────────────────────────────────
@mcp.resource("docs://architecture", mime_type="text/markdown")
def get_architecture() -> str:
    return "# Architecture\\n\\n```mermaid\\ngraph TD\\n  A-->B\\n```"
'''
    print("  FastMCP resource decorators:")
    print(server_code)


# ── 3. Resource URI templates ─────────────────────────────────────────────────
def demo_uri_templates():
    print("\n=== 3. Resource URI Templates ===")

    template_code = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Document Server")

# URI template — {doc_id} is extracted from the URI and passed as parameter
@mcp.resource("docs://{doc_id}")
def get_document(doc_id: str) -> str:
    """Fetch a document by ID. URI: docs://intro, docs://quickstart, etc."""
    docs = {
        "intro":      "# Introduction\\nWelcome to the AI platform.",
        "quickstart": "# Quickstart\\n1. Install\\n2. Configure\\n3. Run",
        "faq":        "# FAQ\\nQ: What is RAG?\\nA: Retrieval-Augmented Generation.",
    }
    if doc_id not in docs:
        raise ValueError(f"Document {doc_id!r} not found. Available: {list(docs)}")
    return docs[doc_id]

# Multi-parameter template
@mcp.resource("db://{schema}/{table}/{row_id}")
def get_row(schema: str, table: str, row_id: str) -> str:
    """Fetch a specific DB row. URI: db://public/users/42"""
    return json.dumps({
        "schema": schema, "table": table,
        "row_id": row_id, "data": {"name": "Alice", "role": "admin"}
    })

# List ALL resources so the client can discover them
@mcp.resource("docs://index")
def list_docs() -> str:
    return json.dumps(["docs://intro", "docs://quickstart", "docs://faq"])
'''
    print(template_code)

    # Show URI template matching logic
    import re

    def match_uri_template(template: str, uri: str) -> dict[str, str] | None:
        """Extract params from a URI template match."""
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", re.escape(template))
        m = re.fullmatch(pattern, uri)
        return m.groupdict() if m else None

    cases = [
        ("docs://{doc_id}", "docs://quickstart"),
        ("db://{schema}/{table}/{row_id}", "db://public/users/42"),
        ("docs://{doc_id}", "metrics://cpu"),   # no match
    ]
    print("  URI template matching:")
    for tmpl, uri in cases:
        params = match_uri_template(tmpl, uri)
        print(f"    {tmpl!r} vs {uri!r} → {params}")


# ── 4. MCP Prompt templates ───────────────────────────────────────────────────
def demo_prompt_templates():
    print("\n=== 4. MCP Prompt Templates ===")

    prompt_code = '''
from mcp.server.fastmcp import FastMCP
from mcp.types import GetPromptResult, PromptMessage, TextContent

mcp = FastMCP("Prompt Library Server")

# Simple prompt — no arguments
@mcp.prompt()
def summarise_document() -> str:
    """Instruct the model to summarise the current document."""
    return "Please read the attached document and provide a concise 3-bullet summary."

# Prompt with arguments — arguments are passed in by the client
@mcp.prompt()
def analyse_code(language: str, focus: str = "quality") -> list[dict]:
    """Multi-turn prompt for code analysis."""
    return [
        {
            "role": "user",
            "content": f"Analyse the following {language} code for {focus} issues.",
        },
        {
            "role": "assistant",
            "content": f"I will review your {language} code focusing on {focus}.",
        },
    ]

# Structured prompt with system + user messages
@mcp.prompt()
def rag_query(question: str, context_resource: str = "docs://readme") -> list[dict]:
    """RAG-style prompt that references a resource."""
    return [
        {
            "role": "system",
            "content": "You answer questions using only the provided context.",
        },
        {
            "role": "user",
            "content": (
                f"Context: {{resource: {context_resource}}}\\n\\n"
                f"Question: {question}"
            ),
        },
    ]
'''
    print(prompt_code)

    # Demonstrate prompt interpolation
    def render_prompt(template: str, **kwargs) -> str:
        return template.format(**kwargs)

    template = "Analyse this {language} code for {focus} issues:\n\n```{language}\n{code}\n```"
    rendered = render_prompt(template, language="Python", focus="performance",
                             code="def fib(n): return fib(n-1)+fib(n-2) if n>1 else n")
    print("  Rendered prompt:")
    for line in rendered.split("\n"):
        print(f"    {line}")


# ── 5. Resource subscriptions ─────────────────────────────────────────────────
def demo_subscriptions():
    print("\n=== 5. Resource Subscriptions ===")
    print("""
Resource subscriptions let clients receive notifications when resource
content changes — enabling reactive AI workflows.

Pattern:
  client.subscribe_resource("metrics://cpu")
  → server sends resources/updated notification when CPU metric changes
  → client re-reads the resource and updates context

Server-side (FastMCP):

    @mcp.resource("metrics://cpu")
    def cpu_metrics() -> str:
        return json.dumps({"usage": psutil.cpu_percent()})

    # Trigger a change notification from server code:
    async def monitor():
        while True:
            await asyncio.sleep(5)
            # Notify all subscribers that this resource changed
            await mcp.notify_resource_updated("metrics://cpu")

Client-side:

    async with mcp.ClientSession(read, write) as session:
        await session.subscribe_resource("metrics://cpu")
        
        async def on_notification(notif):
            if notif.method == "notifications/resources/updated":
                content = await session.read_resource(notif.params.uri)
                print("Updated:", content)
        
        # Register callback
        session.on_notification = on_notification

Use cases:
  • Live dashboard data fed to AI analysis loop
  • Config hot-reload — AI adapts when settings change
  • Event-driven pipelines — agent reacts to new data
""")


# ── 6. Sampling — server-initiated LLM calls ─────────────────────────────────
def demo_sampling():
    print("\n=== 6. Sampling — Server-Initiated LLM Calls ===")
    print("""
Sampling lets an MCP SERVER ask the CLIENT to make an LLM call.
This is the reverse of the normal flow (client → server).

Why useful:
  Server can request LLM completions without managing its own LLM client.
  Human in the loop: client can review/approve before sending to LLM.

Server requesting a sample:

    from mcp.types import SamplingMessage, TextContent

    result = await ctx.sample(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text="Classify this support ticket: " + ticket_text,
                ),
            )
        ],
        max_tokens=100,
        system_prompt="You are a support ticket classifier. Reply with one word: BUG, FEATURE, or BILLING.",
    )
    classification = result.content.text.strip()

Sequence:
  [Server Tool] → sampling/createMessage → [Client] → [LLM] → response → [Server]

Client configuration (must enable sampling):

    capabilities = ClientCapabilities(
        sampling={}     # opt-in — client controls which LLM is used
    )
""")


# ── 7. Practical: document knowledge-base server ─────────────────────────────
def demo_knowledge_base_server():
    print("\n=== 7. Practical: Document Knowledge-Base Server ===")

    full_server = '''
"""
knowledge_base_server.py — A complete MCP server with resources and prompts.

Usage (stdio):
  python knowledge_base_server.py

Usage (SSE):
  mcp.run(transport="sse", host="0.0.0.0", port=8000)
"""
import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Knowledge Base", version="1.0.0")

# ── In-memory document store ────────────────────────────────────────────────
DOCUMENTS: dict[str, dict] = {
    "python-async": {
        "title": "Python Async Guide",
        "content": "Use asyncio for I/O-bound tasks. Avoid blocking calls.",
        "tags": ["python", "async", "performance"],
        "updated": "2025-05-01",
    },
    "rag-intro": {
        "title": "RAG Introduction",
        "content": "RAG combines retrieval with generation for accurate answers.",
        "tags": ["rag", "llm", "retrieval"],
        "updated": "2025-04-15",
    },
}

# ── Resources ───────────────────────────────────────────────────────────────
@mcp.resource("docs://index")
def list_documents() -> str:
    """List all available documents."""
    return json.dumps([
        {"id": k, "title": v["title"], "tags": v["tags"]}
        for k, v in DOCUMENTS.items()
    ])

@mcp.resource("docs://{doc_id}")
def get_document(doc_id: str) -> str:
    """Get a document by ID."""
    if doc_id not in DOCUMENTS:
        raise ValueError(f"Unknown doc: {doc_id!r}")
    return json.dumps(DOCUMENTS[doc_id])

@mcp.resource("docs://{doc_id}/metadata")
def get_metadata(doc_id: str) -> str:
    """Get only metadata (no content)."""
    doc = DOCUMENTS.get(doc_id, {})
    return json.dumps({k: v for k, v in doc.items() if k != "content"})

@mcp.resource("search://{tag}")
def search_by_tag(tag: str) -> str:
    """Search documents by tag."""
    results = [
        {"id": k, "title": v["title"]}
        for k, v in DOCUMENTS.items()
        if tag in v["tags"]
    ]
    return json.dumps(results)

# ── Prompts ─────────────────────────────────────────────────────────────────
@mcp.prompt()
def answer_from_docs(question: str) -> list[dict]:
    """Prompt to answer a question using knowledge base documents."""
    return [
        {"role": "system", "content": "Answer using only information from the provided documents."},
        {"role": "user",   "content": f"Documents: {{resource: docs://index}}\\nQuestion: {question}"},
    ]

@mcp.prompt()
def compare_documents(doc_id_1: str, doc_id_2: str) -> list[dict]:
    """Prompt to compare two knowledge base documents."""
    return [
        {"role": "user", "content": (
            f"Compare these two documents:\\n"
            f"Doc 1: {{resource: docs://{doc_id_1}}}\\n"
            f"Doc 2: {{resource: docs://{doc_id_2}}}\\n"
            "Highlight key similarities and differences."
        )},
    ]

# ── Tools ────────────────────────────────────────────────────────────────────
@mcp.tool()
def add_document(doc_id: str, title: str, content: str, tags: list[str]) -> str:
    """Add a new document to the knowledge base."""
    if doc_id in DOCUMENTS:
        return f"Error: document {doc_id!r} already exists"
    DOCUMENTS[doc_id] = {
        "title": title, "content": content,
        "tags": tags, "updated": datetime.utcnow().date().isoformat(),
    }
    return f"Added document {doc_id!r} successfully"

if __name__ == "__main__":
    mcp.run()
'''
    # Print the server code in sections
    print("  Full server with resources + prompts + tools:")
    lines = full_server.strip().split("\n")
    for line in lines:
        print(f"  {line}")


if __name__ == "__main__":
    print("MCP 02-Intermediate — Resources, Prompts & Sampling")
    print("=" * 54)
    explain_resources()
    demo_resource_server()
    demo_uri_templates()
    demo_prompt_templates()
    demo_subscriptions()
    demo_sampling()
    demo_knowledge_base_server()
