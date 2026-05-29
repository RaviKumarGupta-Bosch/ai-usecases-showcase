"""
CrewAI Use Case 01 — Content Creation Crew
============================================
A full content creation pipeline:
  - Researcher   : gathers topic background from Wikipedia
  - Planner      : builds a structured content outline
  - Writer       : writes the article draft
  - SEO Analyst  : optimises for search and adds meta data
  - Editor       : final polish and quality check

Run:
  python 01_content_creation_crew.py
"""

import os
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.tools import tool

load_dotenv()

llm = LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)


@tool("Wikipedia Research")
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for background information on a topic."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            data = r.json()
            return f"Title: {data.get('title')}\n\n{data.get('extract', '')[:800]}"
        return f"No Wikipedia article found for '{query}'"
    except Exception as e:
        return f"Research failed: {e}"


def create_content_crew(topic: str) -> Crew:
    # ── Agents ────────────────────────────────────────────────────────────────
    researcher = Agent(
        role="Content Researcher",
        goal=f"Gather accurate background facts about: {topic}",
        backstory=(
            "You are a meticulous researcher who digs deep into topics to find "
            "the most relevant and interesting facts. You always cite your sources."
        ),
        llm=llm,
        tools=[wikipedia_search],
        verbose=False,
        allow_delegation=False,
    )

    planner = Agent(
        role="Content Strategist",
        goal="Create structured outlines that make complex topics accessible",
        backstory=(
            "You have planned 500+ successful blog posts and know exactly how "
            "to structure content for maximum reader engagement and SEO."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    writer = Agent(
        role="Senior Content Writer",
        goal="Write engaging, informative articles that developers love to read",
        backstory=(
            "You have 8 years of technical writing experience with bylines in "
            "major tech publications. Your articles rank #1 on Google regularly."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    seo_analyst = Agent(
        role="SEO Specialist",
        goal="Optimise content for search engines while maintaining readability",
        backstory=(
            "You're a data-driven SEO expert who has grown organic traffic by 300% "
            "for multiple SaaS companies. You balance keywords with user intent."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    editor = Agent(
        role="Senior Editor",
        goal="Ensure content excellence: clarity, accuracy, style consistency",
        backstory=(
            "Former editor at a top tech publication with 15 years of experience. "
            "You catch every inconsistency and make every sentence earn its place."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # ── Tasks ─────────────────────────────────────────────────────────────────
    research_task = Task(
        description=f"Research '{topic}' thoroughly. Find: key concepts, history, current applications, interesting statistics.",
        expected_output="500-word research brief with key facts, dates, statistics, and concepts about the topic.",
        agent=researcher,
    )

    outline_task = Task(
        description=f"Create a detailed blog post outline for an article about '{topic}' aimed at software developers.",
        expected_output="Blog post outline with: title, subtitle, 5 sections (each with 3 sub-points), and a conclusion hook.",
        agent=planner,
        context=[research_task],
    )

    writing_task = Task(
        description=(
            f"Write a complete 400-word blog post about '{topic}' following the provided outline. "
            "Use the research facts. Include code examples or analogies where relevant."
        ),
        expected_output="400-word blog post with title, 4 sections, and a strong conclusion call-to-action.",
        agent=writer,
        context=[research_task, outline_task],
    )

    seo_task = Task(
        description=(
            "Analyse the article for SEO and provide: "
            "(1) suggested primary keyword, "
            "(2) 4 secondary keywords, "
            "(3) meta description (150 chars), "
            "(4) 2 headline alternatives optimised for click-through rate."
        ),
        expected_output="SEO brief with keyword strategy, meta description, and headline variants.",
        agent=seo_analyst,
        context=[writing_task],
    )

    editing_task = Task(
        description=(
            "Final edit of the article. Check: clarity, grammar, flow, factual accuracy. "
            "Incorporate SEO recommendations. Provide the polished final article + an editor's note."
        ),
        expected_output=(
            "Final polished article (400 words) with title and meta description "
            "incorporated, plus 3-sentence editor's note on changes made."
        ),
        agent=editor,
        context=[writing_task, seo_task],
    )

    return Crew(
        agents=[researcher, planner, writer, seo_analyst, editor],
        tasks=[research_task, outline_task, writing_task, seo_task, editing_task],
        process=Process.sequential,
        verbose=False,
    )


if __name__ == "__main__":
    TOPICS = [
        "WebAssembly",
        "Edge Computing",
    ]

    for topic in TOPICS[:1]:  # Run one topic in demo
        print("\n" + "=" * 60)
        print(f"Content Creation Crew: {topic}")
        print("=" * 60)

        crew = create_content_crew(topic)
        result = crew.kickoff()

        print("\n--- FINAL ARTICLE ---")
        print(result.raw)

        print(f"\nPipeline stats:")
        print(f"  Tasks completed: {len(result.tasks_output)}")
        for t in result.tasks_output:
            print(f"  [{t.agent}] → {len(t.raw.split())} words")
