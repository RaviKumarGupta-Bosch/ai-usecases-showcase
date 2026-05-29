# CrewAI Tutorial — Basic to Advanced

CrewAI is a framework for orchestrating role-playing AI agents in collaborative teams ("crews").
Each agent has a role, goal, backstory, and assigned tasks. Crews run tasks sequentially or
hierarchically, with agents using tools and passing outputs between each other.

## Curriculum

```
01-basics/
  01_agents_and_tasks.py      — Agent, Task, Crew fundamentals
  02_sequential_crew.py       — Sequential process: tasks in order
  03_tools.py                 — Built-in and custom tools

02-intermediate/
  01_hierarchical_crew.py     — Manager LLM delegates to specialist agents
  02_custom_tools.py          — @tool decorator, BaseTool subclass
  03_crew_output.py           — Structured output, Pydantic models, callbacks

03-advanced/
  01_flows.py                 — CrewAI Flows: event-driven pipelines
  02_memory.py                — Short-term, long-term, entity memory
  03_async_crews.py           — Async task execution, kickoff_async

04-UseCases/
  01_content_creation_crew.py — Blog post: researcher + writer + editor + SEO
  02_research_analyst_crew.py — Deep research: planner + searcher + analyst + reporter
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python 01-basics/01_agents_and_tasks.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `Agent` | Has role, goal, backstory, LLM config, and tools |
| `Task` | Has description, expected_output, agent assignment |
| `Crew` | Orchestrates agents and tasks with a Process |
| `Process.sequential` | Tasks run one after another |
| `Process.hierarchical` | Manager LLM delegates and reviews |
| `@tool` | Decorator to create a CrewAI tool from a function |
| `BaseTool` | Base class for complex tools |

## Prerequisites
- Python 3.10+
- `OPENAI_API_KEY`
