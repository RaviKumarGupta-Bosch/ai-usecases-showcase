"""
Use Case 03 — Research Assistant
===================================
An autonomous research assistant that:
1. Decomposes a research topic into focused sub-queries (Planner)
2. Searches each sub-query using Tavily (or simulated search if no key)
3. Synthesises all findings into a structured markdown report (Synthesiser)
4. Persists progress with LangGraph MemorySaver checkpoints

Architecture (LangGraph):
  plan_research → execute_searches (loop over sub-queries) → synthesise_report → END

State:
  - topic         : original research question
  - sub_queries   : list of focused search queries
  - search_results: accumulated search findings
  - report        : final synthesised markdown report
  - step          : current sub-query index

Run:
  python main.py
  (Set TAVILY_API_KEY for live search; otherwise uses built-in simulated results)
"""

import os
import operator
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Check for Tavily ───────────────────────────────────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HAS_TAVILY = bool(TAVILY_API_KEY)

if HAS_TAVILY:
    from langchain_community.tools import TavilySearchResults
    search_tool = TavilySearchResults(max_results=3)
    print("  Live search: TavilySearchResults enabled")
else:
    print("  Simulated search: set TAVILY_API_KEY for live web search")


# ── Simulated search fallback ─────────────────────────────────────────────────
SIMULATED_RESULTS = {
    "default": [
        {"url": "https://research.example.com/article1", "content":
         "Recent studies indicate significant advances in AI capabilities, "
         "with large language models demonstrating emergent reasoning abilities "
         "previously unseen in smaller models. Researchers note that scale "
         "continues to be a key driver of performance improvements."},
        {"url": "https://academic.example.com/paper42", "content":
         "Meta-analysis of 50 studies on AI adoption shows that organisations "
         "implementing AI tools see an average 23% productivity improvement. "
         "Key success factors include data quality, change management, and "
         "clear ROI measurement frameworks."},
        {"url": "https://industry.example.com/report", "content":
         "Industry experts predict AI will contribute $15.7 trillion to the "
         "global economy by 2030. Healthcare, finance, and manufacturing "
         "are identified as the sectors with highest transformation potential."},
    ]
}


def simulated_search(query: str) -> list[dict]:
    """Return realistic-looking simulated search results."""
    # Customise based on query keywords
    results = []
    query_lower = query.lower()

    if "history" in query_lower or "origin" in query_lower:
        results = [
            {"url": "https://history.example.com/timeline",
             "content": f"Historical overview of '{query}': The field traces its origins "
                        "to foundational research in the 1950s-1970s. Key milestones include "
                        "seminal papers that established core theoretical frameworks still "
                        "used today. The field gained mainstream attention in the early 2010s."},
        ]
    elif "application" in query_lower or "use case" in query_lower or "example" in query_lower:
        results = [
            {"url": "https://applications.example.com/usecases",
             "content": f"Practical applications of '{query}': Real-world deployments span "
                        "healthcare diagnostics, financial fraud detection, autonomous vehicles, "
                        "natural language interfaces, and personalised recommendation systems. "
                        "Enterprise adoption has accelerated significantly since 2020."},
        ]
    elif "challenge" in query_lower or "limitation" in query_lower or "problem" in query_lower:
        results = [
            {"url": "https://challenges.example.com/analysis",
             "content": f"Key challenges in '{query}': Researchers identify data quality, "
                        "interpretability, computational costs, and ethical concerns as primary "
                        "barriers to broader adoption. Regulatory frameworks are still evolving "
                        "in most jurisdictions."},
        ]
    elif "future" in query_lower or "trend" in query_lower or "next" in query_lower:
        results = [
            {"url": "https://trends.example.com/forecast",
             "content": f"Future directions for '{query}': Experts anticipate continued "
                        "convergence of multiple AI modalities, improved energy efficiency, "
                        "and tighter human-AI collaboration frameworks. Industry investment "
                        "is projected to triple over the next five years."},
        ]
    else:
        results = SIMULATED_RESULTS["default"]

    return results[:3]


def do_search(query: str) -> str:
    """Execute a search (live or simulated) and return formatted results."""
    try:
        if HAS_TAVILY:
            results = search_tool.invoke(query)
        else:
            results = simulated_search(query)

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for r in results:
            url = r.get("url", "unknown")
            content = r.get("content", "")
            formatted.append(f"Source: {url}\n{content}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error for '{query}': {e}"


# ── Pydantic models ────────────────────────────────────────────────────────────
class ResearchPlan(BaseModel):
    """Decomposed research plan with focused sub-queries."""
    sub_queries: list[str] = Field(
        description="3-5 focused search queries to cover the research topic",
        min_length=2,
        max_length=6,
    )
    research_angle: str = Field(
        description="Brief description of the research approach (1 sentence)"
    )


# ── Graph state ────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    topic:          str
    sub_queries:    list[str]
    research_angle: str
    search_results: Annotated[list[str], operator.add]  # accumulates results
    current_step:   int
    report:         str


# ── Nodes ─────────────────────────────────────────────────────────────────────
def plan_research_node(state: ResearchState) -> dict:
    """Decompose the topic into focused sub-queries."""
    print(f"\n[Planner] Analysing topic: {state['topic']}")

    planner_llm = llm.with_structured_output(ResearchPlan)

    result = planner_llm.invoke(
        f"""You are a research planner. Given a research topic, create a structured research plan.

Topic: {state['topic']}

Create 3-5 focused sub-queries that together provide comprehensive coverage:
1. Historical context and background
2. Current state and key developments
3. Practical applications and real-world examples
4. Challenges and limitations
5. Future trends and outlook (if relevant)

Make each query specific and searchable."""
    )

    print(f"[Planner] Research angle: {result.research_angle}")
    print(f"[Planner] Sub-queries ({len(result.sub_queries)}):")
    for i, q in enumerate(result.sub_queries, 1):
        print(f"  {i}. {q}")

    return {
        "sub_queries":    result.sub_queries,
        "research_angle": result.research_angle,
        "current_step":  0,
    }


def execute_search_node(state: ResearchState) -> dict:
    """Execute the next search query."""
    step = state["current_step"]
    query = state["sub_queries"][step]

    print(f"\n[Searcher] Step {step + 1}/{len(state['sub_queries'])}: {query}")

    results = do_search(query)
    result_entry = f"## Sub-query {step + 1}: {query}\n\n{results}"

    print(f"[Searcher] Retrieved {len(results.split())} words of results")

    return {
        "search_results": [result_entry],
        "current_step":   step + 1,
    }


def should_continue_searching(state: ResearchState) -> str:
    """Continue searching or move to synthesis."""
    if state["current_step"] < len(state["sub_queries"]):
        return "search"
    return "synthesise"


def synthesise_report_node(state: ResearchState) -> dict:
    """Synthesise all search results into a structured markdown report."""
    print(f"\n[Synthesiser] Compiling report from {len(state['search_results'])} search sets...")

    all_findings = "\n\n---\n\n".join(state["search_results"])

    report = llm.invoke(
        f"""You are an expert research analyst. Synthesise the research findings into a comprehensive report.

Original Topic: {state['topic']}
Research Angle: {state['research_angle']}

Raw Research Findings:
{all_findings}

Write a structured markdown research report with these sections:
1. **Executive Summary** (2-3 sentences)
2. **Key Findings** (3-5 bullet points)
3. **Detailed Analysis** (3-4 paragraphs covering the sub-topics)
4. **Challenges & Limitations**
5. **Future Outlook**
6. **Conclusion** (2-3 sentences)

Be analytical, accurate, and cite key points from the findings."""
    )

    return {"report": report.content}


# ── Build the graph ────────────────────────────────────────────────────────────
def build_research_graph():
    memory = MemorySaver()
    graph = StateGraph(ResearchState)

    graph.add_node("plan",       plan_research_node)
    graph.add_node("search",     execute_search_node)
    graph.add_node("synthesise", synthesise_report_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_conditional_edges(
        "search",
        should_continue_searching,
        {"search": "search", "synthesise": "synthesise"},
    )
    graph.add_edge("synthesise", END)

    return graph.compile(checkpointer=memory)


@traceable(name="research_assistant", tags=["research", "production"])
def research_topic(topic: str, thread_id: str = "default") -> str:
    """Run a full research pipeline on a topic."""
    app = build_research_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke(
        {
            "topic":          topic,
            "sub_queries":    [],
            "research_angle": "",
            "search_results": [],
            "current_step":   0,
            "report":         "",
        },
        config=config,
    )
    return result["report"]


# ── Demo ───────────────────────────────────────────────────────────────────────
def demo_single_research():
    print("=" * 60)
    print("Research Assistant — Demo 1: Single Topic")
    print("=" * 60)

    topic = "The impact of Large Language Models on software engineering productivity"
    print(f"\nResearch Topic: {topic}\n")

    report = research_topic(topic, thread_id="demo-llm-se")

    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(report)


def demo_multiple_topics():
    print("\n" + "=" * 60)
    print("Research Assistant — Demo 2: Multiple Topics")
    print("=" * 60)

    topics = [
        ("Quantum computing applications in cryptography", "quantum-crypto"),
        ("Sustainable energy transition challenges", "energy-transition"),
    ]

    for topic, thread_id in topics:
        print(f"\n{'─'*50}")
        print(f"Topic: {topic}")
        print(f"{'─'*50}")

        report = research_topic(topic, thread_id=thread_id)

        # Print just the executive summary and key findings
        lines = report.split("\n")
        in_section = False
        printed = 0
        for line in lines:
            if "Executive Summary" in line or "Key Findings" in line:
                in_section = True
            if in_section:
                print(line)
                if line.strip() == "" and printed > 3:
                    break
                printed += 1
            if "Detailed Analysis" in line:
                print("  [... full report truncated for demo ...]")
                break


def demo_checkpoint_resume():
    print("\n" + "=" * 60)
    print("Research Assistant — Demo 3: Checkpoint Resume")
    print("=" * 60)

    memory = MemorySaver()
    app = build_research_graph()
    topic = "The future of autonomous vehicles"
    thread_id = "av-research"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nRunning research on: {topic}")
    result = app.invoke(
        {
            "topic":          topic,
            "sub_queries":    [],
            "research_angle": "",
            "search_results": [],
            "current_step":   0,
            "report":         "",
        },
        config=config,
    )

    # Show checkpoint info
    state = app.get_state(config)
    print(f"\nCheckpoint info:")
    print(f"  Sub-queries executed: {state.values['current_step']}")
    print(f"  Search results collected: {len(state.values['search_results'])}")
    print(f"  Report length: {len(state.values['report'])} chars")


if __name__ == "__main__":
    demo_single_research()
    demo_multiple_topics()
    demo_checkpoint_resume()
