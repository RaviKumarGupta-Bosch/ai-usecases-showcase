# AutoGen Tutorial — Basic to Advanced

Microsoft AutoGen is a framework for building multi-agent AI applications where
agents can converse with each other, write and execute code, and collaborate to
solve complex tasks.

## Curriculum

```
01-basics/
  01_intro_agents.py          — ConversableAgent, AssistantAgent, UserProxyAgent
  02_two_agent_chat.py        — Two-agent conversation, termination, cost tracking
  03_group_chat.py            — GroupChat, GroupChatManager, speaker selection

02-intermediate/
  01_tool_use.py              — @register_for_llm / @register_for_execution decorators
  02_code_execution.py        — LocalCommandLineCodeExecutor, Docker executor
  03_custom_agents.py         — Subclassing ConversableAgent, custom reply functions

03-advanced/
  01_nested_chats.py          — Nested chat patterns, swarm agents
  02_structured_output.py     — Structured responses with Pydantic models
  03_stateful_agents.py       — Agent memory, context management

04-UseCases/
  01_coding_team.py           — Full coding team: planner + coder + reviewer + tester
  02_research_team.py         — Research team with web search and synthesis
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python 01-basics/01_intro_agents.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `AssistantAgent` | LLM-powered agent that can chat and suggest code |
| `UserProxyAgent` | Executes code, mediates human input |
| `ConversableAgent` | Base class for all agents |
| `GroupChat` | Multi-agent round-robin or selector conversation |
| `GroupChatManager` | Moderates group chat, selects next speaker |
| `register_for_llm` | Tells LLM about available tools |
| `register_for_execution` | Binds tool implementation to the executor agent |

## Prerequisites
- Python 3.10+
- `OPENAI_API_KEY` (or Azure OpenAI / Anthropic / Gemini)
- Docker (optional, for sandboxed code execution)
