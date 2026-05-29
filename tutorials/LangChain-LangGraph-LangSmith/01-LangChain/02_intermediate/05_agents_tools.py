"""
05 - Agents & Tools
====================
Agents use an LLM to decide which tools to call and in what order.
LangChain supports the ReAct pattern: Reason → Act → Observe → repeat.

Topics covered:
  1. @tool decorator — simplest way to create a tool
  2. Tool class — more control over name/description
  3. StructuredTool — tools with complex Pydantic schemas
  4. Built-in tools: TavilySearchResults, WikipediaQueryRun
  5. create_react_agent + AgentExecutor
  6. Agent with memory (conversation history)
  7. Custom tools with validation
"""

import math
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool, Tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. @tool decorator ───────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e.
    Example: '2 ** 10 + sqrt(144)'
    """
    safe_names = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "pi": math.pi, "e": math.e, "abs": abs, "round": round,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """
    Return the current date and time.
    Args:
        timezone: Timezone label (informational only, system time is used).
    """
    now = datetime.now()
    return f"Current date/time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def word_count(text: str) -> str:
    """
    Count the number of words, sentences, and characters in the given text.
    """
    words = len(text.split())
    sentences = text.count(".") + text.count("!") + text.count("?")
    return f"Words: {words}, Sentences: {sentences}, Characters: {len(text)}"


def demo_tool_decorator():
    print("=== 1. @tool Decorator ===")
    print("Name:", calculator.name)
    print("Description:", calculator.description[:80])
    print("Invoke:", calculator.invoke({"expression": "sqrt(144) + 2**8"}))
    print("Datetime:", get_current_datetime.invoke({"timezone": "UTC"}))
    print("Word count:", word_count.invoke({"text": "Hello world. This is a test!"}))


# ── 2. Tool class ────────────────────────────────────────────────────────────
def reverse_string(text: str) -> str:
    return text[::-1]


def demo_tool_class():
    reverse_tool = Tool(
        name="reverse_string",
        func=reverse_string,
        description="Reverse the characters in a string. Input: any text string.",
    )

    print("\n=== 2. Tool Class ===")
    print(reverse_tool.invoke("LangChain is great!"))


# ── 3. StructuredTool — complex input schema ─────────────────────────────────
class UnitConversionInput(BaseModel):
    value: float = Field(description="Numeric value to convert")
    from_unit: str = Field(description="Source unit (e.g. 'km', 'miles', 'celsius', 'fahrenheit')")
    to_unit: str = Field(description="Target unit")


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("km", "miles"):        value * 0.621371,
        ("miles", "km"):        value * 1.60934,
        ("celsius", "fahrenheit"): value * 9 / 5 + 32,
        ("fahrenheit", "celsius"): (value - 32) * 5 / 9,
        ("kg", "lbs"):          value * 2.20462,
        ("lbs", "kg"):          value * 0.453592,
        ("meters", "feet"):     value * 3.28084,
        ("feet", "meters"):     value * 0.3048,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported."


unit_converter = StructuredTool.from_function(
    func=convert_units,
    name="unit_converter",
    description="Convert between units of measurement (distance, temperature, weight).",
    args_schema=UnitConversionInput,
)


def demo_structured_tool():
    print("\n=== 3. StructuredTool ===")
    result = unit_converter.invoke({"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"})
    print(result)
    result2 = unit_converter.invoke({"value": 26.2, "from_unit": "miles", "to_unit": "km"})
    print(result2)


# ── 4. React Agent with custom tools ─────────────────────────────────────────
def demo_react_agent():
    """
    ReAct agent uses the LLM to reason about which tool to call next.
    Uses LangChain Hub's react prompt (includes Thought/Action/Observation format).
    """
    tools = [calculator, get_current_datetime, word_count, unit_converter]

    # LangChain Hub prompt for ReAct — includes the required scratchpad variable
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )

    print("\n=== 4. ReAct Agent ===")
    result = agent_executor.invoke({
        "input": (
            "What is the square root of 256, and how many miles is 42.195 km "
            "(a marathon)? Also tell me today's date."
        )
    })
    print("\nFinal Answer:", result["output"])


# ── 5. Agent with built-in tools ─────────────────────────────────────────────
def demo_builtin_tools():
    """Wikipedia tool — no API key needed."""
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))

        tools = [wiki_tool, calculator]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5, handle_parsing_errors=True)

        print("\n=== 5. Built-in Wikipedia Tool ===")
        result = executor.invoke({"input": "What year was Python programming language created, and what is that year squared?"})
        print("Answer:", result["output"])
    except ImportError:
        print("\n=== 5. Built-in Wikipedia Tool === (install: pip install wikipedia)")


# ── 6. Agent with memory ─────────────────────────────────────────────────────
def demo_agent_with_memory():
    """
    Agents can maintain conversation history across turns.
    Uses a chat-style prompt with history placeholder.
    """
    tools = [calculator, word_count]

    # Agent prompt that supports multi-turn conversation
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant with access to tools.\n"
         "Always think step by step. Use tools when they help you compute or look things up.\n"
         "Conversation history is provided for context.\n\n"
         "Tools available: {tools}\n"
         "Tool names: {tool_names}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    from langchain.agents.format_scratchpad import format_log_to_str
    from langchain.agents.output_parsers import ReActSingleInputOutputParser
    from langchain.tools.render import render_text_description

    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
            "agent_scratchpad": lambda x: format_log_to_str(x.get("intermediate_steps", [])),
            "tools": lambda _: render_text_description(tools),
            "tool_names": lambda _: ", ".join([t.name for t in tools]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session(sid):
        if sid not in store:
            store[sid] = InMemoryChatMessageHistory()
        return store[sid]

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
    )

    print("\n=== 6. Agent with Memory ===")
    cfg = {"configurable": {"session_id": "agent-session"}}
    session_history = InMemoryChatMessageHistory()

    def chat(user_input: str) -> str:
        history = session_history.messages
        result = executor.invoke({"input": user_input, "chat_history": history})
        session_history.add_user_message(user_input)
        session_history.add_ai_message(result["output"])
        return result["output"]

    print("Turn 1:", chat("What is 15 factorial? (15!)"))
    print("Turn 2:", chat("Divide that result by 1000000"))
    print("Turn 3:", chat("How many words are in 'The quick brown fox jumps over the lazy dog'?"))


if __name__ == "__main__":
    demo_tool_decorator()
    demo_tool_class()
    demo_structured_tool()
    demo_react_agent()
    demo_builtin_tools()
    demo_agent_with_memory()
