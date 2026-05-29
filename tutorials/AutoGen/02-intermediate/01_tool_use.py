"""
AutoGen Intermediate 01 — Tool Use
=====================================
Topics covered:
  1. Registering tools with @register_for_llm and @register_for_execution
  2. Built-in tool calling pattern
  3. Agents with multiple tools
  4. Tool result handling and error recovery
  5. Combining tools with code execution

Run:
  python 01_tool_use.py
"""

import os
import math
import json
import requests
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent, register_function

load_dotenv()

LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0,
}


# ── Tool definitions ──────────────────────────────────────────────────────────

def calculate(
    expression: Annotated[str, "A mathematical expression to evaluate, e.g. '2 ** 10 + sqrt(144)'"]
) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def get_current_datetime(
    timezone: Annotated[str, "Timezone name, e.g. 'UTC', 'US/Eastern'"] = "UTC"
) -> str:
    """Return the current date and time."""
    now = datetime.utcnow()
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def search_wikipedia(
    query: Annotated[str, "Search term to look up on Wikipedia"]
) -> str:
    """Fetch a Wikipedia summary for the given query."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("extract", "No summary found.")[:500]
        return f"Wikipedia returned status {resp.status_code}"
    except Exception as e:
        return f"Search failed: {e}"


def word_count(
    text: Annotated[str, "Text to count words in"]
) -> str:
    """Count words and characters in a text."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


# ── 1. Single tool registration ───────────────────────────────────────────────
def demo_single_tool():
    print("\n=== 1. Single Tool (Calculator) ===")

    assistant = AssistantAgent(
        name="MathAssistant",
        llm_config=LLM_CONFIG,
        system_message="You are a math assistant. Use the calculate tool to evaluate expressions.",
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=2,
    )

    # Register the tool: tell LLM about it AND bind execution
    register_function(
        calculate,
        caller=assistant,
        executor=user,
        name="calculate",
        description="Evaluate a mathematical expression. Supports sqrt, sin, cos, log, etc.",
    )

    user.initiate_chat(
        assistant,
        message="What is sqrt(144) + 2^8, and what is sin(pi/4) to 4 decimal places?",
    )


# ── 2. Multiple tools ─────────────────────────────────────────────────────────
def demo_multiple_tools():
    print("\n=== 2. Multiple Tools ===")

    assistant = AssistantAgent(
        name="ResearchAssistant",
        llm_config=LLM_CONFIG,
        system_message="""You are a research assistant with access to tools.
Use them to answer questions accurately.""",
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=5,
    )

    for fn, name, desc in [
        (calculate,         "calculate",        "Evaluate mathematical expressions"),
        (get_current_datetime, "get_datetime",  "Get current date and time"),
        (search_wikipedia,  "search_wikipedia", "Search Wikipedia for information"),
        (word_count,        "word_count",       "Count words, chars, sentences in text"),
    ]:
        register_function(fn, caller=assistant, executor=user, name=name, description=desc)

    user.initiate_chat(
        assistant,
        message=(
            "1) What is the square root of 2048? "
            "2) Look up 'machine learning' on Wikipedia and count the words in the summary. "
            "3) What is the current time?"
        ),
    )


# ── 3. Tool with error handling ───────────────────────────────────────────────
def demo_tool_error_recovery():
    print("\n=== 3. Tool Error Recovery ===")

    def risky_divide(
        a: Annotated[float, "Numerator"],
        b: Annotated[float, "Denominator"],
    ) -> str:
        """Divide two numbers. Returns error if division by zero."""
        if b == 0:
            return "ERROR: Division by zero is undefined"
        return f"Result: {a / b:.6f}"

    assistant = AssistantAgent(
        name="Calculator",
        llm_config=LLM_CONFIG,
        system_message="Use the divide tool. If you get an error, explain why mathematically.",
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=3,
    )

    register_function(
        risky_divide,
        caller=assistant,
        executor=user,
        name="divide",
        description="Divide two numbers",
    )

    user.initiate_chat(
        assistant,
        message="Calculate: 100 / 4, then 50 / 0, then 7 / 3.",
    )


# ── 4. Stateful tool (maintains state between calls) ─────────────────────────
def demo_stateful_tool():
    print("\n=== 4. Stateful Tool (Running Totals) ===")

    ledger = {"balance": 0.0, "transactions": []}

    def bank_transaction(
        amount: Annotated[float, "Amount to deposit (positive) or withdraw (negative)"],
        description: Annotated[str, "Transaction description"],
    ) -> str:
        """Process a bank transaction and return the new balance."""
        ledger["balance"] += amount
        ledger["transactions"].append({
            "amount":      amount,
            "description": description,
            "balance":     ledger["balance"],
        })
        action = "Deposit" if amount > 0 else "Withdrawal"
        return (
            f"{action} of ${abs(amount):.2f} ({description}). "
            f"New balance: ${ledger['balance']:.2f}"
        )

    def get_statement() -> str:
        """Get the full transaction statement."""
        lines = [f"Balance: ${ledger['balance']:.2f}", "Transactions:"]
        for t in ledger["transactions"]:
            lines.append(f"  {t['description']}: ${t['amount']:+.2f} → ${t['balance']:.2f}")
        return "\n".join(lines)

    assistant = AssistantAgent(
        name="BankBot",
        llm_config=LLM_CONFIG,
        system_message="You manage a bank account using the provided tools.",
    )

    user = UserProxyAgent(
        name="Customer",
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=6,
    )

    register_function(bank_transaction, caller=assistant, executor=user,
                      name="transaction", description="Deposit or withdraw money")
    register_function(get_statement, caller=assistant, executor=user,
                      name="get_statement", description="Get full account statement")

    user.initiate_chat(
        assistant,
        message=(
            "Please: deposit $1000 (salary), withdraw $250 (rent), "
            "deposit $50 (interest), withdraw $75 (groceries). "
            "Then show me the full statement."
        ),
    )


if __name__ == "__main__":
    demo_single_tool()
    demo_multiple_tools()
    demo_tool_error_recovery()
    demo_stateful_tool()
