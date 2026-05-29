"""
CrewAI Basics 03 — Tools
==========================
Topics covered:
  1. Using built-in CrewAI tools (FileReadTool, DirectoryReadTool)
  2. Creating tools with the @tool decorator
  3. Creating tools with BaseTool subclass
  4. Assigning tools to agents
  5. Agent using multiple tools in a task

Run:
  python 03_tools.py
"""

import os
import math
import json
import requests
from datetime import datetime
from typing import Type
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)


# ── 1. @tool decorator — simple function-based tools ─────────────────────────
from crewai.tools import tool


@tool("Calculator")
def calculator_tool(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e
    Example: 'sqrt(144) + 2**8'
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{result}"
    except Exception as e:
        return f"Error: {e}"


@tool("Wikipedia Search")
def wikipedia_tool(query: str) -> str:
    """
    Search Wikipedia for a topic and return a brief summary.
    Best for: factual questions, definitions, historical facts.
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("extract", "No summary found.")[:600]
        return f"No Wikipedia article found for '{query}'"
    except Exception as e:
        return f"Search failed: {e}"


@tool("Current Time")
def current_time_tool(timezone: str = "UTC") -> str:
    """Return the current UTC date and time."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


# ── 2. BaseTool subclass — for tools with complex logic ──────────────────────
class WordAnalysisInput(BaseModel):
    text: str = Field(..., description="The text to analyse")


class WordAnalysisTool(BaseTool):
    name: str = "Word Analysis"
    description: str = (
        "Analyse text for word count, sentence count, average word length, "
        "and the top 5 most frequent words. Input: the text string."
    )
    args_schema: Type[BaseModel] = WordAnalysisInput

    def _run(self, text: str) -> str:
        words = text.lower().split()
        sentences = text.count(".") + text.count("!") + text.count("?")
        avg_len = sum(len(w.strip(".,!?;:\"'()")) for w in words) / max(len(words), 1)

        # Word frequency (ignore stop words)
        stop = {"the", "a", "an", "is", "it", "in", "on", "at", "to", "and", "or", "of", "for"}
        freq: dict[str, int] = {}
        for w in words:
            clean = w.strip(".,!?;:\"'()")
            if clean and clean not in stop:
                freq[clean] = freq.get(clean, 0) + 1

        top5 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]

        return json.dumps({
            "word_count":      len(words),
            "sentence_count":  max(sentences, 1),
            "avg_word_length": round(avg_len, 1),
            "top_5_words":     dict(top5),
        }, indent=2)


# ── 3. Demo: single tool ──────────────────────────────────────────────────────
def demo_single_tool():
    print("\n=== 1. Agent with Single Tool (Calculator) ===")

    math_agent = Agent(
        role="Maths Tutor",
        goal="Solve mathematical problems step by step",
        backstory="You are a maths teacher who shows your working clearly.",
        llm=llm,
        tools=[calculator_tool],
        verbose=False,
    )

    task = Task(
        description=(
            "Solve these problems: "
            "a) What is 2^10? "
            "b) What is sqrt(1764)? "
            "c) What is sin(pi/6) rounded to 4 decimal places?"
        ),
        expected_output="Three answers with brief explanations, using the calculator tool.",
        agent=math_agent,
    )

    crew = Crew(agents=[math_agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    print(result.raw)


# ── 4. Demo: multiple tools ───────────────────────────────────────────────────
def demo_multiple_tools():
    print("\n=== 2. Agent with Multiple Tools ===")

    word_tool = WordAnalysisTool()

    research_agent = Agent(
        role="Research Analyst",
        goal="Research topics and provide data-backed summaries",
        backstory="You are a thorough analyst who uses every available tool.",
        llm=llm,
        tools=[wikipedia_tool, word_tool, current_time_tool],
        verbose=False,
    )

    task = Task(
        description=(
            "1. Look up 'artificial intelligence' on Wikipedia. "
            "2. Analyse the text you get back — report word count and top words. "
            "3. Note the current time. "
            "Present findings as a structured report."
        ),
        expected_output="A structured report with Wikipedia summary, word analysis stats, and timestamp.",
        agent=research_agent,
    )

    crew = Crew(agents=[research_agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    print(result.raw)


# ── 5. Demo: tools with multiple agents ──────────────────────────────────────
def demo_tools_with_crew():
    print("\n=== 3. Tools Distributed Across Crew ===")

    researcher = Agent(
        role="Topic Researcher",
        goal="Gather accurate information on technology topics",
        backstory="You specialise in finding and verifying technical information.",
        llm=llm,
        tools=[wikipedia_tool, current_time_tool],
        verbose=False,
    )

    analyst = Agent(
        role="Content Analyst",
        goal="Analyse and score content quality",
        backstory="You evaluate writing quality and provide improvement scores.",
        llm=llm,
        tools=[calculator_tool, WordAnalysisTool()],
        verbose=False,
    )

    research_task = Task(
        description="Research 'neural network' on Wikipedia and return the full summary.",
        expected_output="Wikipedia summary of neural networks (full extract).",
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "Analyse the research text: count words, find top 5 terms, "
            "then calculate reading time (words / 200 words per minute)."
        ),
        expected_output="Word stats, top 5 terms, estimated reading time in minutes.",
        agent=analyst,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    print(result.raw)


if __name__ == "__main__":
    demo_single_tool()
    demo_multiple_tools()
    demo_tools_with_crew()
