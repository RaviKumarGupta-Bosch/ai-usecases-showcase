"""
CrewAI Use Case 02 — Research Analyst Crew
===========================================
A multi-agent deep research pipeline:
  - Research Planner   : decomposes the topic into sub-questions
  - Wikipedia Researcher: searches Wikipedia for each sub-topic
  - Data Analyst       : synthesises findings into insights
  - Report Writer      : produces a structured research report

Run:
  python 02_research_analyst_crew.py
"""

import os
import json
import requests
from typing import Type
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.tools import BaseTool, tool
from pydantic import BaseModel, Field

load_dotenv()

llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("Wikipedia Multi-Search")
def wikipedia_multi_search(queries: str) -> str:
    """
    Search Wikipedia for multiple comma-separated topics.
    Returns a combined research summary.
    Input: comma-separated list of search terms, e.g. 'Python programming, machine learning, neural networks'
    """
    results = []
    for query in queries.split(","):
        q = query.strip()
        if not q:
            continue
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q.replace(' ', '_')}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                results.append(f"## {data.get('title', q)}\n{data.get('extract', '')[:400]}")
            else:
                results.append(f"## {q}\nNo article found.")
        except Exception as e:
            results.append(f"## {q}\nError: {e}")
    return "\n\n".join(results)


class StatisticsInput(BaseModel):
    numbers: str = Field(..., description="Comma-separated list of numbers to compute stats for")


class StatisticsTool(BaseTool):
    name: str = "Statistics Calculator"
    description: str = (
        "Calculate basic statistics (mean, median, min, max, range) "
        "for a comma-separated list of numbers."
    )
    args_schema: Type[BaseModel] = StatisticsInput

    def _run(self, numbers: str) -> str:
        try:
            nums = [float(x.strip()) for x in numbers.split(",") if x.strip()]
            if not nums:
                return "No numbers provided"
            nums_sorted = sorted(nums)
            n = len(nums_sorted)
            mean = sum(nums_sorted) / n
            median = nums_sorted[n // 2] if n % 2 == 1 else (nums_sorted[n // 2 - 1] + nums_sorted[n // 2]) / 2
            return json.dumps({
                "count": n, "mean": round(mean, 2), "median": median,
                "min": nums_sorted[0], "max": nums_sorted[-1],
                "range": nums_sorted[-1] - nums_sorted[0],
            })
        except Exception as e:
            return f"Error: {e}"


def run_research(topic: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Research Crew: {topic}")
    print("=" * 60)

    stats_tool = StatisticsTool()

    # ── Agents ────────────────────────────────────────────────────────────────
    planner = Agent(
        role="Research Planner",
        goal="Break complex topics into clear, answerable sub-questions",
        backstory=(
            "You are a research director who has led 200+ research projects. "
            "You know how to decompose any topic into a structured investigation plan."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    researcher = Agent(
        role="Wikipedia Researcher",
        goal="Find accurate, comprehensive information on each research sub-topic",
        backstory=(
            "You are a research librarian with expertise in quickly finding "
            "and extracting the most relevant information from any source."
        ),
        llm=llm,
        tools=[wikipedia_multi_search],
        verbose=False,
        allow_delegation=False,
    )

    analyst = Agent(
        role="Research Analyst",
        goal="Synthesise raw research into meaningful insights and patterns",
        backstory=(
            "You are a senior analyst who transforms raw information into "
            "clear insights with supporting evidence and statistical backing."
        ),
        llm=llm,
        tools=[stats_tool],
        verbose=False,
        allow_delegation=False,
    )

    writer = Agent(
        role="Research Report Writer",
        goal="Produce clear, structured research reports for professional audiences",
        backstory=(
            "You have written 100+ research reports for Fortune 500 companies. "
            "Your reports are known for clarity, depth, and actionable conclusions."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ── Tasks ─────────────────────────────────────────────────────────────────
    plan_task = Task(
        description=(
            f"Create a research plan for: '{topic}'\n"
            "Identify 4 key sub-topics or questions to investigate. "
            "For each, specify what we need to know and why it matters."
        ),
        expected_output="Research plan with 4 sub-topics, each with a 'What to find' and 'Why it matters' bullet.",
        agent=planner,
    )

    research_task = Task(
        description=(
            f"Using the research plan, search Wikipedia for all 4 sub-topics related to '{topic}'. "
            "Compile the raw research findings."
        ),
        expected_output="Raw research: Wikipedia excerpts for each of the 4 sub-topics, clearly labelled.",
        agent=researcher,
        context=[plan_task],
    )

    analysis_task = Task(
        description=(
            "Analyse the research findings. Identify: "
            "(1) 3 key trends or patterns, "
            "(2) 2 surprising or counter-intuitive facts, "
            "(3) any quantitative data worth computing statistics on."
        ),
        expected_output="Analysis: 3 trends, 2 surprising facts, and any statistical findings.",
        agent=analyst,
        context=[research_task],
    )

    report_task = Task(
        description=(
            f"Write a professional research report on '{topic}'. Structure:\n"
            "- Executive Summary (2 sentences)\n"
            "- Key Findings (3-4 bullet points with evidence)\n"
            "- Detailed Analysis (2 paragraphs)\n"
            "- Conclusions & Implications (3 bullet points)\n"
            "- Recommended Next Steps (2 actions)"
        ),
        expected_output=(
            "Full structured research report with all 5 sections, "
            "incorporating findings and analysis."
        ),
        agent=writer,
        context=[research_task, analysis_task],
    )

    crew = Crew(
        agents=[planner, researcher, analyst, writer],
        tasks=[plan_task, research_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    print("\n--- RESEARCH REPORT ---")
    print(result.raw)
    print(f"\nPipeline: {len(result.tasks_output)} stages completed")


if __name__ == "__main__":
    topics = [
        "Quantum Computing in cryptography",
        "Large Language Models and their societal impact",
    ]

    run_research(topics[0])
