"""
CrewAI Basics 01 — Agents and Tasks
=====================================
Topics covered:
  1. Creating an Agent with role, goal, backstory
  2. Creating a Task with description and expected output
  3. Running a minimal single-agent Crew
  4. Passing context between tasks
  5. Inspecting crew output

Run:
  python 01_agents_and_tasks.py
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM

load_dotenv()

llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)


# ── 1. Minimal single-agent crew ──────────────────────────────────────────────
def demo_single_agent():
    print("\n=== 1. Single Agent Crew ===")

    writer = Agent(
        role="Technical Writer",
        goal="Write clear, accurate technical explanations for developers",
        backstory=(
            "You are an experienced technical writer with 10 years of experience "
            "explaining complex software concepts in plain English. "
            "You love bullet points and concrete examples."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description="Explain what a REST API is in 3 bullet points suitable for a junior developer.",
        expected_output="3 concise bullet points explaining REST APIs, each under 30 words.",
        agent=writer,
    )

    crew = Crew(agents=[writer], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()

    print("Output:")
    print(result.raw)


# ── 2. Task context — passing output from one task to the next ────────────────
def demo_task_context():
    print("\n=== 2. Task Context (Output Chaining) ===")

    researcher = Agent(
        role="Research Analyst",
        goal="Find and summarise key facts about technology topics",
        backstory="You are a meticulous researcher who always backs claims with data.",
        llm=llm,
        verbose=False,
    )

    writer = Agent(
        role="Content Writer",
        goal="Transform research into engaging blog content",
        backstory="You write viral tech blog posts read by 100k engineers monthly.",
        llm=llm,
        verbose=False,
    )

    research_task = Task(
        description=(
            "Research the top 3 benefits of using Python for data science. "
            "Include one real statistic or survey result for each benefit."
        ),
        expected_output="3 benefits with supporting data points, in bullet format.",
        agent=researcher,
    )

    writing_task = Task(
        description=(
            "Using the research provided, write a short blog introduction (2 paragraphs) "
            "that would hook a data scientist into reading more."
        ),
        expected_output="2 engaging paragraphs, each 3-4 sentences, citing the research.",
        agent=writer,
        context=[research_task],   # <-- output of research_task feeds in here
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    print("Final blog intro:")
    print(result.raw)
    print("\nTask outputs:")
    for task_output in result.tasks_output:
        print(f"  [{task_output.agent}] {task_output.raw[:80]}...")


# ── 3. Multi-task sequential crew ─────────────────────────────────────────────
def demo_multi_task_crew():
    print("\n=== 3. Multi-Task Sequential Crew ===")

    analyst = Agent(
        role="Market Analyst",
        goal="Analyse market trends and provide actionable insights",
        backstory="You have an MBA from Wharton and 15 years of market analysis experience.",
        llm=llm,
        verbose=False,
    )

    task1 = Task(
        description="List 5 top tech trends for 2025 with a one-sentence explanation each.",
        expected_output="Numbered list of 5 tech trends, each with a 1-sentence explanation.",
        agent=analyst,
    )

    task2 = Task(
        description=(
            "From the trends identified, pick the ONE with the highest business impact. "
            "Write a 150-word executive summary explaining why."
        ),
        expected_output="150-word executive summary about the most impactful 2025 tech trend.",
        agent=analyst,
        context=[task1],
    )

    task3 = Task(
        description=(
            "Based on the top trend, recommend 3 concrete actions a startup could take "
            "in the next 6 months to capitalise on it."
        ),
        expected_output="3 numbered action items, each with a timeline and expected outcome.",
        agent=analyst,
        context=[task1, task2],
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    print("Final recommendations:")
    print(result.raw)
    print(f"\nTotal tasks completed: {len(result.tasks_output)}")
    print(f"Token usage: {result.token_usage}")


if __name__ == "__main__":
    demo_single_agent()
    demo_task_context()
    demo_multi_task_crew()
