# LangChain · LangGraph · LangSmith — Complete AI Engineer Tutorial

A progressive, hands-on tutorial covering **LangChain**, **LangGraph**, and **LangSmith**
from first principles to advanced AI engineering patterns — with rich real-world use cases.

---

## Learning Path

```
Basic → Intermediate → Advanced
   │           │            │
LangChain   LangChain   LangChain
 Basics    Intermediate  Advanced
   │           │            │
   └─────── LangGraph ──────┘
             │    │    │
          Basics  Mid  Advanced
                  │
             LangSmith
          (tracing & evals)
                  │
            Use Cases
     (real end-to-end apps)
```

---

## Folder Structure

```
tutorials/LangChain-LangGraph-LangSmith/
│
├── 01-LangChain/
│   ├── 01_basics/
│   │   ├── 01_llm_and_chat_models.py      ← LLMs, streaming, async, batch
│   │   ├── 02_prompt_templates.py          ← PromptTemplate, ChatPromptTemplate, FewShot
│   │   ├── 03_lcel_chains.py               ← LCEL pipe |, Parallel, Passthrough, Branch
│   │   ├── 04_output_parsers.py            ← Str, JSON, Pydantic, List parsers
│   │   └── 05_memory.py                    ← Chat history, window memory, summary
│   │
│   ├── 02_intermediate/
│   │   ├── 01_document_loaders.py          ← Text, Web, Directory, custom loaders
│   │   ├── 02_text_splitters.py            ← Recursive, Token, Code, Markdown splitters
│   │   ├── 03_embeddings_vectorstores.py   ← OpenAI embeddings, FAISS, Chroma
│   │   ├── 04_retrieval.py                 ← MultiQuery, Compression, Ensemble retrievers
│   │   └── 05_agents_tools.py              ← @tool, create_react_agent, AgentExecutor
│   │
│   └── 03_advanced/
│       ├── 01_rag_pipeline.py              ← Full RAG: load→split→embed→retrieve→generate
│       ├── 02_conversational_rag.py        ← History-aware retriever, chat RAG
│       ├── 03_structured_output.py         ← with_structured_output, Pydantic extraction
│       ├── 04_streaming_callbacks.py       ← Token streaming, custom callback handlers
│       └── 05_routing_chains.py            ← RunnableBranch, semantic routing
│
├── 02-LangGraph/
│   ├── 01_basics/
│   │   ├── 01_simple_graph.py              ← StateGraph, nodes, edges, compile, invoke
│   │   ├── 02_state_and_nodes.py           ← TypedDict state, reducers, add_messages
│   │   └── 03_conditional_edges.py         ← Router functions, END, loops
│   │
│   ├── 02_intermediate/
│   │   ├── 01_react_agent.py               ← ReAct loop built from scratch with LangGraph
│   │   ├── 02_human_in_the_loop.py         ← interrupt_before, Command, approval workflow
│   │   └── 03_persistence_checkpoints.py   ← MemorySaver, threads, resume conversations
│   │
│   └── 03_advanced/
│       ├── 01_multi_agent_supervisor.py    ← Supervisor + worker agents team
│       ├── 02_parallel_nodes.py            ← Send API, fan-out/fan-in, map-reduce
│       └── 03_plan_and_execute.py          ← Planner → Executor → Replanner loop
│
├── 03-LangSmith/
│   ├── 01_tracing.py                       ← @traceable, metadata, nested runs
│   ├── 02_evaluation.py                    ← LLM-as-judge, custom evaluators, evaluate()
│   └── 03_datasets_experiments.py          ← Datasets, examples, experiments, comparison
│
└── 04-UseCases/
    ├── 01_customer_support_bot/            ← RAG + memory + escalation + tracing
    ├── 02_rag_document_qa/                 ← Multi-doc QA with citations + history
    ├── 03_research_assistant/              ← Plan→Search→Synthesize with LangGraph
    └── 04_code_review_agent/               ← Multi-aspect code review with structured output
```

---

## Prerequisites

- Python 3.10+
- OpenAI API key (required for all LangChain/LangGraph examples)
- LangSmith API key (required for `03-LangSmith/` and tracing in use cases)
- Tavily API key (required for `03_research_assistant/` — free tier available)

---

## Setup

```bash
# 1. Clone / navigate to this folder
cd tutorials/LangChain-LangGraph-LangSmith

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API keys
cp .env.example .env
# Edit .env and fill in your keys

# 5. Run any example
python 01-LangChain/01_basics/01_llm_and_chat_models.py
```

---

## Concepts Covered

### LangChain
| Level | Concepts |
|---|---|
| **Basics** | ChatOpenAI, PromptTemplate, LCEL (`\|`), OutputParsers, Memory |
| **Intermediate** | Document loaders, Text splitters, Embeddings, Vector stores, Retrieval, Agents |
| **Advanced** | Full RAG, Conversational RAG, Structured output, Streaming, Routing chains |

### LangGraph
| Level | Concepts |
|---|---|
| **Basics** | StateGraph, Nodes, Edges, TypedDict state, END |
| **Intermediate** | ReAct agent, Human-in-the-loop, Persistence & checkpoints |
| **Advanced** | Multi-agent supervisor, Parallel execution (Send), Plan-and-execute |

### LangSmith
| Concept | Description |
|---|---|
| **Tracing** | `@traceable`, run metadata, nested traces, dashboards |
| **Evaluation** | LLM-as-judge, custom evaluators, scoring |
| **Datasets** | Create datasets, run experiments, compare prompt versions |

---

## Key Patterns Quick Reference

```python
# LCEL chain
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "..."})

# RAG retrieval chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# LangGraph minimal agent
graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()

# LangSmith tracing
from langsmith import traceable
@traceable(name="my_chain")
def run_chain(question: str) -> str:
    return chain.invoke({"question": question})
```
