"""
CrewAI Basics 02 — Sequential and Hierarchical Crews
======================================================
Topics covered:
  1. Sequential process — tasks run in order
  2. Hierarchical process — manager LLM delegates to agents
  3. Agent delegation between agents (allow_delegation)
  4. Crew configuration options (memory, cache, max_rpm)

Run:
  python 02_crew_processes.py
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM

load_dotenv()

llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
manager_llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)


# ── 1. Sequential process ─────────────────────────────────────────────────────
def demo_sequential_process():
    print("\n=== 1. Sequential Process ===")

    planner = Agent(
        role="Content Planner",
        goal="Plan structured content outlines",
        backstory="You excel at organising ideas into logical, reader-friendly structures.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    writer = Agent(
        role="Content Writer",
        goal="Write compelling content following a given outline",
        backstory="You craft engaging content that resonates with technical audiences.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    editor = Agent(
        role="Editor",
        goal="Polish and improve content quality",
        backstory="You have an eagle eye for clarity, grammar, and tone consistency.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    plan_task = Task(
        description="Create a 4-section outline for an article: 'Why Every Developer Should Learn SQL'",
        expected_output="4-section outline with section titles and 2-3 bullet points each.",
        agent=planner,
    )

    write_task = Task(
        description="Write the introduction section (100 words) based on the provided outline.",
        expected_output="100-word article introduction, engaging and technically accurate.",
        agent=writer,
        context=[plan_task],
    )

    edit_task = Task(
        description="Edit the introduction for clarity and punch. Suggest 2 improvements.",
        expected_output="Polished introduction with 2 inline improvement suggestions marked [EDIT].",
        agent=editor,
        context=[write_task],
    )

    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan_task, write_task, edit_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    print("Final edited introduction:")
    print(result.raw)


# ── 2. Hierarchical process ───────────────────────────────────────────────────
def demo_hierarchical_process():
    print("\n=== 2. Hierarchical Process (Manager delegates) ===")

    # Specialist agents
    data_analyst = Agent(
        role="Data Analyst",
        goal="Analyse numerical data and spot trends",
        backstory="You turn raw numbers into actionable insights using statistical thinking.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    visualization_expert = Agent(
        role="Visualisation Expert",
        goal="Describe ideal chart types and layouts for data",
        backstory="You specialise in data visualisation best practices and UX for dashboards.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    storyteller = Agent(
        role="Data Storyteller",
        goal="Turn data insights into compelling narratives for non-technical audiences",
        backstory="You bridge the gap between analysts and executives with clear stories.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # In hierarchical mode, the manager LLM orchestrates — no explicit agent assignment needed
    analysis_task = Task(
        description=(
            "Analyse this sales data and provide key insights:\n"
            "Q1: $120k, Q2: $145k, Q3: $98k, Q4: $210k\n"
            "Annual target: $500k. Top product: SaaS subscription (60% of revenue)."
        ),
        expected_output="3-5 key business insights from the sales data with percentages.",
    )

    viz_task = Task(
        description="Recommend the best chart types to visualise the quarterly sales data and why.",
        expected_output="3 chart recommendations with chart type, reason, and key variable to highlight.",
    )

    story_task = Task(
        description="Write a 2-sentence executive summary of the year's sales performance for the CEO.",
        expected_output="2-sentence executive summary suitable for a C-level audience.",
    )

    crew = Crew(
        agents=[data_analyst, visualization_expert, storyteller],
        tasks=[analysis_task, viz_task, story_task],
        process=Process.hierarchical,
        manager_llm=manager_llm,  # Manager LLM orchestrates delegation
        verbose=False,
    )

    result = crew.kickoff()
    print("Final output:")
    print(result.raw)


# ── 3. Agent delegation ───────────────────────────────────────────────────────
def demo_agent_delegation():
    print("\n=== 3. Agent Delegation ===")

    senior = Agent(
        role="Senior Developer",
        goal="Oversee code quality and mentor junior developers",
        backstory="10-year veteran who has shipped systems used by millions of users.",
        llm=llm,
        verbose=False,
        allow_delegation=True,   # <-- can delegate tasks
    )

    junior = Agent(
        role="Junior Developer",
        goal="Implement specific code tasks as directed",
        backstory="2-year developer who writes clean Python and is eager to learn.",
        llm=llm,
        verbose=False,
        allow_delegation=False,  # <-- cannot delegate further
    )

    task = Task(
        description=(
            "Implement a Python function `fibonacci(n)` that returns the nth Fibonacci number. "
            "Include a docstring and handle edge cases. Then review the implementation."
        ),
        expected_output="Working Python function with docstring, edge case handling, and a brief code review.",
        agent=senior,  # Senior agent may delegate the implementation
    )

    crew = Crew(
        agents=[senior, junior],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    print("Result:")
    print(result.raw)


if __name__ == "__main__":
    demo_sequential_process()
    demo_hierarchical_process()
    demo_agent_delegation()
