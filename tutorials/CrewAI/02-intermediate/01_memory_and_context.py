"""
CrewAI 02-Intermediate — Memory, Context & Task Dependencies
==============================================================
Topics covered:
  1. Task context passing — using previous task output in next task
  2. Task dependencies (context=[task1, task2])
  3. Pydantic output schemas for structured task results
  4. Task callbacks for monitoring and logging
  5. Async crew execution (kickoff_async)
  6. Human-in-the-loop tasks
  7. Practical: document analysis pipeline with context flow

Prerequisites:
  pip install crewai openai python-dotenv pydantic

Run:
  python 01_memory_and_context.py
"""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from pydantic import BaseModel, Field

load_dotenv()

llm = LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
          api_key=os.getenv("OPENAI_API_KEY"), temperature=0)


# ── 1. Task context passing ───────────────────────────────────────────────────
def demo_task_context():
    print("\n=== 1. Task Context Passing ===")

    researcher = Agent(
        role="Research Analyst",
        goal="Extract key facts and insights from documents",
        backstory="Expert at reading and distilling complex documents into facts.",
        llm=llm, verbose=False,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write clear summaries from research notes",
        backstory="Skilled at turning research notes into readable summaries.",
        llm=llm, verbose=False,
    )

    research_task = Task(
        description="Research the topic: Python async programming. List 5 key facts.",
        expected_output="A numbered list of 5 key facts about Python async programming.",
        agent=researcher,
    )

    # context=[research_task] — writer sees research_task's output automatically
    summary_task = Task(
        description=(
            "Write a 2-paragraph summary for a developer blog post "
            "using the research facts provided in context."
        ),
        expected_output="A 2-paragraph blog post summary.",
        agent=writer,
        context=[research_task],   # ← injects research_task output into writer's context
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, summary_task],
        process=Process.sequential,
        verbose=False,
    )

    print(f"  Tasks: research_task → summary_task")
    print(f"  summary_task.context = [research_task]  → output injected automatically")
    print("  Crew configured. Run: crew.kickoff()")


# ── 2. Multi-task dependency chains ──────────────────────────────────────────
def demo_task_dependencies():
    print("\n=== 2. Task Dependency Chains ===")

    analyst = Agent(
        role="Data Analyst",
        goal="Analyse data and produce insights",
        backstory="Expert in data analysis and pattern recognition.",
        llm=llm, verbose=False,
    )

    task_a = Task(
        description="Summarise Q1 sales data: [100, 200, 150, 300] units per week.",
        expected_output="A brief Q1 sales summary with trends.",
        agent=analyst,
    )
    task_b = Task(
        description="Identify the top-performing week from the Q1 summary.",
        expected_output="The best-performing week and its unit count.",
        agent=analyst,
        context=[task_a],
    )
    task_c = Task(
        description="Write a 1-sentence executive recommendation based on Q1 analysis.",
        expected_output="One actionable recommendation sentence.",
        agent=analyst,
        context=[task_a, task_b],   # ← depends on both previous tasks
    )

    crew = Crew(agents=[analyst], tasks=[task_a, task_b, task_c],
                process=Process.sequential, verbose=False)

    print("  Chain: task_a → task_b (uses a) → task_c (uses a + b)")
    print("  context=[task_a, task_b] makes both outputs available to task_c")
    print("  Crew configured. Run: result = crew.kickoff()")


# ── 3. Pydantic output schemas ────────────────────────────────────────────────
def demo_pydantic_output():
    print("\n=== 3. Pydantic Output Schemas ===")

    class SentimentResult(BaseModel):
        sentiment: str = Field(description="POSITIVE, NEGATIVE, or NEUTRAL")
        confidence: float = Field(description="Confidence score 0.0-1.0")
        key_phrases: list[str] = Field(description="Top 3 phrases driving the sentiment")
        summary: str = Field(description="One-sentence summary")

    class MarketAnalysis(BaseModel):
        ticker: str
        recommendation: str = Field(description="BUY, HOLD, or SELL")
        price_target: Optional[float] = Field(default=None, description="Target price in USD")
        rationale: str

    analyst = Agent(
        role="Financial Analyst",
        goal="Analyse financial news and provide structured insights",
        backstory="Expert in financial sentiment analysis and market research.",
        llm=llm, verbose=False,
    )

    sentiment_task = Task(
        description="Analyse sentiment of: 'The company reported record earnings, beating estimates.'",
        expected_output="Structured sentiment analysis.",
        agent=analyst,
        output_pydantic=SentimentResult,   # ← enforce structured output
    )

    market_task = Task(
        description="Analyse: 'TechCorp stock up 15% on strong AI product launches.'",
        expected_output="Structured market analysis for TechCorp.",
        agent=analyst,
        output_pydantic=MarketAnalysis,
    )

    print("  SentimentResult schema:", list(SentimentResult.model_fields.keys()))
    print("  MarketAnalysis schema: ", list(MarketAnalysis.model_fields.keys()))
    print()
    print("  output_pydantic=SentimentResult → task output parsed into model")
    print("  Access: result.pydantic.sentiment, result.pydantic.confidence, ...")
    print("  Also available as dict: result.to_dict()")


# ── 4. Task callbacks ─────────────────────────────────────────────────────────
def demo_callbacks():
    print("\n=== 4. Task Callbacks ===")

    import time

    task_timings: dict[str, float] = {}

    def on_task_start(task_output):
        """Called when a task starts (step_callback on Crew)."""
        task_timings["start"] = time.time()
        print(f"    [callback] task started")

    def on_task_end(task_output):
        """task_callback is called when each task completes."""
        elapsed = time.time() - task_timings.get("start", time.time())
        print(f"    [callback] task complete in {elapsed:.2f}s")
        print(f"    [callback] output[:60]: {str(task_output)[:60]!r}")

    researcher = Agent(
        role="Researcher", goal="Research and summarise topics",
        backstory="Research specialist.", llm=llm, verbose=False,
    )

    task = Task(
        description="List 3 benefits of using type hints in Python.",
        expected_output="Numbered list of 3 benefits.",
        agent=researcher,
        callback=on_task_end,   # ← fires when this task finishes
    )

    crew = Crew(
        agents=[researcher], tasks=[task],
        process=Process.sequential, verbose=False,
        step_callback=on_task_start,    # ← fires on every agent step
    )

    print("  task.callback      → fires when this specific task finishes")
    print("  crew.step_callback → fires on every LLM step across all tasks")
    print("  Use for: logging, timing, cost tracking, alerting")
    print(f"  Crew configured. Run: crew.kickoff()")


# ── 5. Async crew execution ───────────────────────────────────────────────────
def demo_async_kickoff():
    print("\n=== 5. Async Crew Execution ===")

    async def run_two_crews_concurrently():
        agent = Agent(
            role="Analyst", goal="Produce analysis",
            backstory="Experienced analyst.", llm=llm, verbose=False,
        )

        task1 = Task(description="Analyse Python trends in 2025.",
                     expected_output="3 key trends.", agent=agent)
        task2 = Task(description="Analyse JavaScript trends in 2025.",
                     expected_output="3 key trends.", agent=agent)

        crew1 = Crew(agents=[agent], tasks=[task1], process=Process.sequential, verbose=False)
        crew2 = Crew(agents=[agent], tasks=[task2], process=Process.sequential, verbose=False)

        # Run concurrently — saves wall-clock time when tasks are independent
        print("  Concurrent execution:")
        print("    results = await asyncio.gather(")
        print("        crew1.kickoff_async(),")
        print("        crew2.kickoff_async(),")
        print("    )")
        # Actual call (commented to avoid API usage in demo):
        # results = await asyncio.gather(crew1.kickoff_async(), crew2.kickoff_async())

    asyncio.run(run_two_crews_concurrently())

    # kickoff_for_each — run same crew against multiple input sets
    print()
    print("  kickoff_for_each_async: run crew N times with different inputs")
    print("    inputs = [{'topic': 'Python'}, {'topic': 'Rust'}, {'topic': 'Go'}]")
    print("    results = crew.kickoff_for_each_async(inputs=inputs)")


# ── 6. Human-in-the-loop tasks ───────────────────────────────────────────────
def demo_human_in_loop():
    print("\n=== 6. Human-in-the-Loop Tasks ===")

    # human_input=True — agent pauses and asks human for input mid-task
    reviewer_agent = Agent(
        role="Code Reviewer",
        goal="Review code with optional human feedback",
        backstory="Expert code reviewer who values human oversight.",
        llm=llm, verbose=False,
    )

    review_task = Task(
        description="Review this Python function for correctness and style: def add(a,b): return a+b",
        expected_output="Code review with pass/fail verdict.",
        agent=reviewer_agent,
        human_input=True,   # ← pauses after agent draft, asks human to approve/modify
    )

    crew = Crew(agents=[reviewer_agent], tasks=[review_task],
                process=Process.sequential, verbose=False)

    print("  human_input=True → crew pauses after agent produces draft output")
    print("  Human is shown the draft and can:")
    print("    - Press Enter to accept")
    print("    - Type feedback to send back to the agent for revision")
    print("  Use for: quality gates, approval workflows, sensitive decisions")
    print(f"  Crew configured. Run: crew.kickoff()")


# ── 7. Practical: document analysis pipeline ─────────────────────────────────
def demo_document_pipeline():
    print("\n=== 7. Practical: Document Analysis Pipeline ===")

    class DocumentInsights(BaseModel):
        title: str
        key_topics: list[str] = Field(description="Top 5 topics")
        sentiment: str = Field(description="POSITIVE, NEGATIVE, NEUTRAL")
        action_items: list[str] = Field(description="Recommended actions")
        confidence: float

    extractor = Agent(
        role="Document Extractor",
        goal="Extract structured information from documents",
        backstory="Expert at reading documents and identifying key information.",
        llm=llm, verbose=False,
    )
    summariser = Agent(
        role="Executive Summariser",
        goal="Produce concise executive summaries",
        backstory="Senior analyst who writes clear, actionable executive briefs.",
        llm=llm, verbose=False,
    )
    recommender = Agent(
        role="Strategy Advisor",
        goal="Provide strategic recommendations based on analysis",
        backstory="Strategic consultant who translates insights into action plans.",
        llm=llm, verbose=False,
    )

    doc_text = (
        "Q2 2025 Performance Report: Revenue grew 22% YoY driven by AI product adoption. "
        "Customer churn increased 3% in SMB segment. R&D costs up 40%. "
        "New enterprise contracts signed: 15. Action required on pricing strategy."
    )

    extract_task = Task(
        description=f"Extract key information from this document:\n\n{doc_text}",
        expected_output="Structured document insights.",
        agent=extractor,
        output_pydantic=DocumentInsights,
    )
    summary_task = Task(
        description="Write a 3-sentence executive summary from the extracted insights.",
        expected_output="3-sentence executive summary.",
        agent=summariser,
        context=[extract_task],
    )
    recommend_task = Task(
        description="Provide 3 strategic recommendations based on the full analysis.",
        expected_output="3 prioritised strategic recommendations.",
        agent=recommender,
        context=[extract_task, summary_task],
    )

    pipeline = Crew(
        agents=[extractor, summariser, recommender],
        tasks=[extract_task, summary_task, recommend_task],
        process=Process.sequential,
        verbose=False,
    )

    print(f"  Document: {doc_text[:60]!r}...")
    print(f"  Pipeline: Extractor → Summariser → Recommender")
    print(f"  Output schema: {list(DocumentInsights.model_fields.keys())}")
    print("  Run: result = pipeline.kickoff()")
    print("  Access typed output: result.tasks_output[0].pydantic.key_topics")


if __name__ == "__main__":
    print("CrewAI 02-Intermediate — Memory, Context & Task Dependencies")
    print("=" * 60)
    demo_task_context()
    demo_task_dependencies()
    demo_pydantic_output()
    demo_callbacks()
    demo_async_kickoff()
    demo_human_in_loop()
    demo_document_pipeline()
