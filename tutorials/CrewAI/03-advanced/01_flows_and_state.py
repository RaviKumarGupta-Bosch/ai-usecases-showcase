"""
CrewAI 03-Advanced — Flows and State Management
=================================================
Topics covered:
  1. CrewAI Flows basics — @start, @listen, structured state
  2. Conditional routing with @router
  3. Parallel execution with and_() / or_() listeners
  4. Flow composition — calling a Crew inside a Flow
  5. Error handling and retries in flows
  6. Flow persistence and resumption
  7. Practical: content publishing flow with approval gate

Prerequisites:
  pip install crewai openai python-dotenv pydantic

Run:
  python 01_flows_and_state.py
"""

import os
import random
from typing import Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.flow.flow import Flow, listen, start, router, and_, or_
from pydantic import BaseModel, Field

load_dotenv()

llm = LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
          api_key=os.getenv("OPENAI_API_KEY"), temperature=0)


# ── Shared state models ───────────────────────────────────────────────────────

class ResearchState(BaseModel):
    topic: str = ""
    research_notes: str = ""
    draft_article: str = ""
    review_feedback: str = ""
    approved: bool = False
    publish_url: Optional[str] = None
    iteration: int = 0


class ContentState(BaseModel):
    topic: str = ""
    target_audience: str = "developers"
    tone: str = "professional"
    draft: str = ""
    seo_keywords: list[str] = Field(default_factory=list)
    approved: bool = False
    published: bool = False


# ── 1. Flow basics: @start and @listen ───────────────────────────────────────
def demo_flow_basics():
    print("\n=== 1. Flow Basics — @start and @listen ===")

    class SimpleFlow(Flow[ResearchState]):
        """
        Linear flow: gather_topic → research → write_draft
        Each method listens to the return value of the previous.
        """

        @start()
        def gather_topic(self):
            print("    [start] gathering topic")
            self.state.topic = "Python async programming best practices"
            return self.state.topic

        @listen(gather_topic)
        def research(self, topic: str):
            print(f"    [listen] researching: {topic!r}")
            # In a real flow, this would call a Crew or tool
            self.state.research_notes = f"Notes on {topic}: use asyncio, avoid blocking I/O."
            return self.state.research_notes

        @listen(research)
        def write_draft(self, notes: str):
            print(f"    [listen] writing draft from {len(notes)} chars of notes")
            self.state.draft_article = f"# Article\n\n{notes}\n\nWritten by AI."
            self.state.iteration += 1
            return self.state.draft_article

    flow = SimpleFlow()
    print("  Flow structure: gather_topic → research → write_draft")
    print("  State type: ResearchState (pydantic BaseModel)")
    print("  Run: result = flow.kickoff()")
    print("  Access state after: flow.state.draft_article")
    print()
    # Run the flow
    result = flow.kickoff()
    print(f"  State after run: topic={flow.state.topic!r}, iteration={flow.state.iteration}")


# ── 2. Conditional routing with @router ──────────────────────────────────────
def demo_router():
    print("\n=== 2. Conditional Routing with @router ===")

    class ReviewFlow(Flow[ResearchState]):

        @start()
        def write_draft(self):
            self.state.draft_article = "Draft article content here."
            self.state.iteration += 1
            print(f"    [start] draft written (iteration {self.state.iteration})")
            return "draft_ready"

        @router(write_draft)
        def review_gate(self, event: str):
            """Route to approval or revision based on simulated review."""
            score = random.uniform(0.4, 1.0)   # simulate review score
            print(f"    [router] review score: {score:.2f}")
            if score >= 0.7 or self.state.iteration >= 3:
                self.state.approved = True
                return "approved"
            else:
                self.state.review_feedback = "Add more examples and improve intro."
                return "needs_revision"

        @listen("approved")
        def publish(self):
            self.state.publish_url = "https://blog.example.com/article-123"
            print(f"    [listen:approved] published at {self.state.publish_url!r}")

        @listen("needs_revision")
        def revise(self):
            print(f"    [listen:needs_revision] revising — {self.state.review_feedback!r}")
            self.state.draft_article += "\n\n[Revised: added examples]"
            return self.write_draft()   # loop back

    flow = ReviewFlow()
    print("  @router: routes to different listeners based on returned string")
    print("  Routes: 'approved' → publish | 'needs_revision' → revise → write_draft")
    result = flow.kickoff()
    print(f"  Final state: approved={flow.state.approved}, url={flow.state.publish_url!r}")


# ── 3. Parallel listeners with and_() / or_() ────────────────────────────────
def demo_parallel_listeners():
    print("\n=== 3. Parallel Listeners with and_() / or_() ===")

    class ParallelFlow(Flow[ContentState]):

        @start()
        def init_topic(self):
            self.state.topic = "AI Developer Tools in 2025"
            return self.state.topic

        @listen(init_topic)
        def generate_draft(self, topic: str):
            self.state.draft = f"Draft content about {topic}."
            print(f"    [generate_draft] done")
            return "draft_done"

        @listen(init_topic)
        def gather_seo_keywords(self, topic: str):
            self.state.seo_keywords = ["AI tools", "developer productivity", "2025 trends"]
            print(f"    [gather_seo_keywords] done: {self.state.seo_keywords}")
            return "keywords_done"

        # and_() waits for BOTH generate_draft AND gather_seo_keywords to complete
        @listen(and_(generate_draft, gather_seo_keywords))
        def combine_and_publish(self):
            print("    [combine_and_publish] both tasks finished — combining")
            self.state.published = True
            full = f"{self.state.draft}\n\nKeywords: {', '.join(self.state.seo_keywords)}"
            return full

    flow = ParallelFlow()
    print("  and_(task_a, task_b): waits for both before firing")
    print("  or_(task_a, task_b):  fires when either completes first")
    result = flow.kickoff()
    print(f"  published={flow.state.published}, keywords={flow.state.seo_keywords}")


# ── 4. Crew inside a Flow ─────────────────────────────────────────────────────
def demo_crew_in_flow():
    print("\n=== 4. Crew Inside a Flow ===")

    # Build a Crew to use inside the Flow
    researcher_agent = Agent(
        role="Researcher", goal="Research topics concisely",
        backstory="Fast, accurate researcher.", llm=llm, verbose=False,
    )

    class CrewFlow(Flow[ContentState]):

        @start()
        def set_topic(self):
            self.state.topic = "Benefits of type hints in Python"
            print(f"    [start] topic: {self.state.topic!r}")
            return self.state.topic

        @listen(set_topic)
        def run_research_crew(self, topic: str):
            task = Task(
                description=f"List 3 key benefits of: {topic}",
                expected_output="Numbered list of 3 benefits.",
                agent=researcher_agent,
            )
            crew = Crew(
                agents=[researcher_agent], tasks=[task],
                process=Process.sequential, verbose=False,
            )
            print("    [listen] spawning Crew for research...")
            print("    (Run: crew.kickoff() — needs OPENAI_API_KEY)")
            # result = crew.kickoff()
            # self.state.draft = result.raw
            self.state.draft = "[crew output would appear here]"
            return self.state.draft

    flow = CrewFlow()
    print("  Pattern: Flow orchestrates multiple Crews as steps")
    print("  Each @listen method can kickoff a different Crew")
    print("  Flow state is shared across all steps")
    result = flow.kickoff()
    print(f"  draft[:60]: {flow.state.draft[:60]!r}")


# ── 5. Error handling in flows ────────────────────────────────────────────────
def demo_error_handling():
    print("\n=== 5. Error Handling in Flows ===")

    class ResilientFlow(Flow[ResearchState]):

        @start()
        def fetch_data(self):
            # Simulate intermittent failure
            if random.random() < 0.4:
                raise ConnectionError("Data source unavailable")
            self.state.research_notes = "Data fetched successfully."
            print("    [fetch_data] success")
            return "data_ready"

        @listen("data_ready")
        def process_data(self, event: str):
            print(f"    [process_data] processing...")
            self.state.draft_article = f"Processed: {self.state.research_notes}"
            return "done"

    # Retry wrapper around flow.kickoff()
    def run_with_retry(flow_cls, max_attempts: int = 3):
        for attempt in range(1, max_attempts + 1):
            try:
                f = flow_cls()
                result = f.kickoff()
                print(f"  Succeeded on attempt {attempt}")
                return result, f.state
            except (ConnectionError, Exception) as e:
                print(f"  Attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    raise
        return None, None

    print("  Error handling patterns:")
    print("  1. try/except around kickoff() with retry loop")
    print("  2. @router to route to error-recovery listener")
    print("  3. State.error_message field to track failures")
    print()
    try:
        result, state = run_with_retry(ResilientFlow, max_attempts=5)
    except Exception as e:
        print(f"  All retries exhausted: {e}")


# ── 6. Flow persistence (state save/load) ────────────────────────────────────
def demo_persistence():
    print("\n=== 6. Flow Persistence and Resumption ===")

    import json
    from pathlib import Path

    class PersistentFlow(Flow[ContentState]):

        STATE_FILE = Path("/tmp/flow_state.json")

        def save_state(self):
            self.STATE_FILE.write_text(self.state.model_dump_json(), encoding="utf-8")
            print(f"    state saved to {self.STATE_FILE}")

        def load_state(self) -> bool:
            if self.STATE_FILE.exists():
                data = json.loads(self.STATE_FILE.read_text(encoding="utf-8"))
                self.state = ContentState(**data)
                print(f"    state loaded: topic={self.state.topic!r}")
                return True
            return False

        @start()
        def begin(self):
            if not self.load_state():
                self.state.topic = "Resumable AI workflows"
                print("    [begin] fresh start")
            else:
                print("    [begin] resuming from saved state")
            return "ready"

        @listen("ready")
        def work(self, event: str):
            self.state.draft = f"Article about: {self.state.topic}"
            self.state.published = True
            self.save_state()
            print("    [work] draft complete and state persisted")

    print("  Persistence pattern: save state.model_dump_json() to file/DB")
    print("  Resumption: load state at @start before processing")
    print("  Use cases: long-running workflows, crash recovery, human checkpoints")
    flow = PersistentFlow()
    flow.kickoff()


# ── 7. Practical: content publishing flow ────────────────────────────────────
def demo_publishing_flow():
    print("\n=== 7. Practical: Content Publishing Flow ===")

    class PublishingState(BaseModel):
        topic: str = ""
        target_audience: str = "developers"
        draft: str = ""
        seo_keywords: list[str] = Field(default_factory=list)
        review_passed: bool = False
        published_url: Optional[str] = None
        retry_count: int = 0

    class PublishingFlow(Flow[PublishingState]):

        @start()
        def receive_brief(self):
            self.state.topic = "Top 10 Python Libraries for AI in 2025"
            self.state.target_audience = "Python developers"
            print(f"    [brief] topic={self.state.topic!r}")
            return "brief_received"

        @listen("brief_received")
        def research_and_draft(self, event: str):
            print("    [draft] generating article draft...")
            self.state.draft = (
                f"# {self.state.topic}\n\n"
                f"Audience: {self.state.target_audience}\n\n"
                "1. LangChain  2. AutoGen  3. CrewAI  4. LlamaIndex  5. Haystack ..."
            )
            self.state.seo_keywords = ["Python AI", "LLM libraries", "2025 AI tools"]
            return "draft_ready"

        @router("draft_ready")
        def editorial_review(self, event: str):
            score = 0.85   # simulate review
            print(f"    [review] score={score:.2f}")
            if score >= 0.75:
                self.state.review_passed = True
                return "approved"
            else:
                self.state.retry_count += 1
                return "rejected" if self.state.retry_count < 3 else "force_publish"

        @listen("approved")
        @listen("force_publish")
        def publish_content(self):
            slug = self.state.topic.lower().replace(" ", "-")[:40]
            self.state.published_url = f"https://blog.example.com/{slug}"
            print(f"    [publish] live at: {self.state.published_url!r}")

        @listen("rejected")
        def revise_draft(self):
            print(f"    [revise] improving draft (attempt {self.state.retry_count})")
            self.state.draft += "\n\n[Revised with more detail]"
            return "draft_ready"

    flow = PublishingFlow()
    print("  Stages: brief → draft → review → publish/revise")
    result = flow.kickoff()
    print(f"\n  Final: review_passed={flow.state.review_passed}, url={flow.state.published_url!r}")


if __name__ == "__main__":
    print("CrewAI 03-Advanced — Flows and State Management")
    print("=" * 50)
    demo_flow_basics()
    demo_router()
    demo_parallel_listeners()
    demo_crew_in_flow()
    demo_error_handling()
    demo_persistence()
    demo_publishing_flow()
