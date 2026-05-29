"""
Ollama 04-UseCases — Private Offline Assistant
================================================
Topics covered:
  1. Conversational assistant with persistent message history
  2. Simulated tool use via prompt (calculator, date/time, unit converter)
  3. Persona and system-level instructions
  4. Graceful handling of off-topic or unsafe requests
  5. Session summary on exit

This assistant runs 100% locally:
  - No API keys
  - No internet after model download
  - All data stays on your machine

Prerequisites:
  - Ollama running: `ollama serve`
  - Model pulled: `ollama pull llama3.2`

Run:
  python 01_private_assistant.py
"""

import os
import sys
import json
import re
import math
from datetime import datetime, timezone
from dotenv import load_dotenv
import ollama

load_dotenv()

MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are a helpful, private, offline AI assistant called 'Aria'.

Key behaviours:
- Be concise and practical. Avoid unnecessary filler text.
- You have access to a few built-in tools. When the user asks for a calculation,
  conversion, or the current date/time, use the TOOL syntax below.
- Privacy: remind users that everything stays local — no cloud, no logging.

Built-in tools (use EXACTLY this format when needed):
  TOOL:calculate:<expression>          — e.g. TOOL:calculate:sqrt(144)
  TOOL:datetime:<timezone>             — e.g. TOOL:datetime:UTC
  TOOL:convert:<value> <from> to <to>  — e.g. TOOL:convert:100 km to miles

You do NOT have web access. If asked for live information (news, stock prices,
current weather), explain that you operate offline and suggest alternatives."""


# ── Tool executor ──────────────────────────────────────────────────────────────
def execute_tool(tool_call: str) -> str:
    """Parse and execute a TOOL:... directive, return result string."""
    try:
        parts = tool_call.split(":", 2)
        if len(parts) < 3:
            return "[Tool error: malformed call]"
        _, tool, args = parts[0], parts[1].lower(), parts[2].strip()

        if tool == "calculate":
            # Safe evaluation — allow only math expressions
            allowed = set("0123456789+-*/.() ")
            allowed_funcs = {"sqrt", "pi", "e", "sin", "cos", "tan", "log", "abs", "pow", "round"}
            expr = args.replace("^", "**")
            # Replace function names
            safe_expr = expr
            for fn in allowed_funcs:
                safe_expr = safe_expr.replace(fn, f"math.{fn}")
            result = eval(safe_expr, {"__builtins__": {}, "math": math})  # noqa: S307
            return f"= {result}"

        elif tool == "datetime":
            tz_name = args.upper() or "UTC"
            now = datetime.now(timezone.utc)
            return f"Current date/time ({tz_name}): {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"

        elif tool == "convert":
            # Simple unit conversions
            conversions = {
                ("km", "miles"):    lambda v: v * 0.621371,
                ("miles", "km"):    lambda v: v * 1.60934,
                ("kg", "lb"):       lambda v: v * 2.20462,
                ("lb", "kg"):       lambda v: v * 0.453592,
                ("c", "f"):         lambda v: v * 9/5 + 32,
                ("f", "c"):         lambda v: (v - 32) * 5/9,
                ("m", "ft"):        lambda v: v * 3.28084,
                ("ft", "m"):        lambda v: v * 0.3048,
                ("l", "gallon"):    lambda v: v * 0.264172,
                ("gallon", "l"):    lambda v: v * 3.78541,
            }
            # Parse "100 km to miles"
            match = re.match(r"([\d.]+)\s+(\w+)\s+to\s+(\w+)", args, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                from_unit = match.group(2).lower()
                to_unit   = match.group(3).lower()
                fn = conversions.get((from_unit, to_unit))
                if fn:
                    result = fn(value)
                    return f"{value} {from_unit} = {result:.4f} {to_unit}"
                return f"[Conversion not supported: {from_unit} → {to_unit}]"
            return "[Could not parse conversion]"

        return f"[Unknown tool: {tool}]"
    except Exception as e:
        return f"[Tool error: {e}]"


def process_tool_calls(text: str) -> str:
    """Find all TOOL: directives in model output and replace with results."""
    pattern = re.compile(r"TOOL:[a-z]+:[^\n]+", re.IGNORECASE)
    def replace(match):
        result = execute_tool(match.group())
        return f"[{match.group()} → {result}]"
    return pattern.sub(replace, text)


# ── Chat session ───────────────────────────────────────────────────────────────
class PrivateAssistant:
    def __init__(self):
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.turn_count = 0
        self.start_time = datetime.now()

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        self.turn_count += 1

        response = ollama.chat(model=MODEL, messages=self.history)
        raw_reply = (
            response.message.content
            if hasattr(response, "message")
            else response["message"]["content"]
        )

        # Process any tool calls embedded in the reply
        reply = process_tool_calls(raw_reply)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def session_summary(self) -> str:
        duration = (datetime.now() - self.start_time).seconds
        return (
            f"\n  Session summary\n"
            f"  ─────────────────────────────\n"
            f"  Turns:       {self.turn_count}\n"
            f"  Duration:    {duration}s\n"
            f"  Model:       {MODEL}\n"
            f"  Privacy:     100% local — no data sent externally\n"
        )


# ── Demo: scripted conversation ───────────────────────────────────────────────
def demo_scripted_conversation():
    print("\n=== Scripted Demo Conversation ===")
    print("  (In real use, replace with input() for interactive mode)\n")

    assistant = PrivateAssistant()

    scripted_inputs = [
        "Hello! What can you help me with?",
        "What is the square root of 1764?",
        "Convert 72 fahrenheit to celsius.",
        "What's today's date?",
        "I'm building a Python REST API. What framework would you recommend?",
        "Can you check today's stock price for Apple?",
        "Thanks, that's helpful. Keep a note: remind me to review this chat.",
    ]

    for user_msg in scripted_inputs:
        print(f"  You:   {user_msg}")
        reply = assistant.chat(user_msg)
        print(f"  Aria:  {reply.strip()[:300]}")
        print()

    print(assistant.session_summary())


# ── Interactive mode ──────────────────────────────────────────────────────────
def run_interactive():
    print("\n╔══════════════════════════════════════════╗")
    print("║   Aria — Private Offline AI Assistant   ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Model:   {MODEL}")
    print("  Privacy: 100% local — nothing sent to the cloud")
    print("  Type 'quit' or 'exit' to end the session.\n")

    assistant = PrivateAssistant()

    while True:
        try:
            user_input = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            break

        print("  Aria: ", end="", flush=True)
        try:
            reply = assistant.chat(user_input)
            print(reply.strip())
        except Exception as e:
            print(f"[Error: {e}]")
        print()

    print(assistant.session_summary())


if __name__ == "__main__":
    print("Ollama 04-UseCases — Private Offline Assistant")
    print("=" * 48)

    try:
        import ollama as _test
        _test.list()
    except Exception:
        print("\n  ERROR: Cannot connect to Ollama. Start with: ollama serve")
        sys.exit(1)

    # Run scripted demo by default.
    # Change to run_interactive() for a live chat experience.
    demo_scripted_conversation()

    # Uncomment for interactive chat:
    # run_interactive()
