"""
02 - Parallel Nodes & Map-Reduce (LangGraph Advanced)
======================================================
LangGraph's Send API enables fan-out: one node spawns multiple parallel
subgraph executions, then a reducer aggregates all results (fan-in).

Topics covered:
  1. Send API — fan-out to parallel nodes
  2. Reducer aggregating parallel results (operator.add)
  3. Map-reduce: process a list in parallel, combine results
  4. Parallel LLM calls (concurrent document analysis)
  5. Weighted score aggregation
  6. Error handling in parallel branches
"""

import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. Basic Send — fan-out to parallel nodes ─────────────────────────────────
class ItemState(TypedDict):
    item: str
    processed_item: str


class CollectionState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]
    summary: str


def fan_out_node(state: CollectionState) -> list[Send]:
    """
    Instead of returning a dict, return a list of Send objects.
    Each Send spawns a parallel execution of the target node with its own state.
    """
    return [Send("process_item", {"item": item, "processed_item": ""}) for item in state["items"]]


def process_item_node(state: ItemState) -> dict:
    """Process a single item. Runs in parallel for each Send."""
    item = state["item"]
    result = f"[PROCESSED] {item.upper()} (length: {len(item)})"
    return {"processed_item": result}


def aggregate_node(state: CollectionState) -> dict:
    """Fan-in: all parallel results have been reduced into state['results']."""
    summary = f"Processed {len(state['results'])} items successfully."
    return {"summary": summary}


def demo_basic_fanout():
    # We need a combined state that includes both CollectionState and ItemState fields
    class FanOutState(TypedDict):
        items: list[str]
        results: Annotated[list[str], operator.add]
        item: str
        processed_item: str
        summary: str

    def fan_out(state: FanOutState) -> list[Send]:
        return [Send("process", {"item": it, "processed_item": "", "items": [], "results": [], "summary": ""})
                for it in state["items"]]

    def process(state: FanOutState) -> dict:
        result = f"[DONE] {state['item'].upper()}"
        return {"results": [result]}

    def aggregate(state: FanOutState) -> dict:
        return {"summary": f"Completed {len(state['results'])} items."}

    graph = StateGraph(FanOutState)
    graph.add_node("fan_out",   fan_out)
    graph.add_node("process",   process)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("fan_out")
    graph.add_conditional_edges("fan_out", lambda _: "process", {"process": "process"})
    # Override with Send-based fan-out
    graph = StateGraph(FanOutState)
    graph.add_node("process",   process)
    graph.add_node("aggregate", aggregate)
    graph.set_entry_point("aggregate")  # placeholder — will be replaced

    # Proper Send-based graph
    graph2 = StateGraph(FanOutState)
    graph2.add_node("fan_out",   fan_out)
    graph2.add_node("process",   process)
    graph2.add_node("aggregate", aggregate)
    graph2.set_entry_point("fan_out")
    graph2.add_conditional_edges("fan_out", fan_out)  # returns list[Send]
    graph2.add_edge("process",   "aggregate")
    graph2.add_edge("aggregate", END)

    app = graph2.compile()

    items = ["apple", "banana", "cherry", "date", "elderberry"]
    result = app.invoke({
        "items": items, "results": [], "item": "", "processed_item": "", "summary": ""
    })

    print("=== 1. Basic Fan-Out with Send ===")
    print(f"  Input items  : {items}")
    print(f"  Results      : {result['results']}")
    print(f"  Summary      : {result['summary']}")


# ── 2. Map-Reduce: parallel document analysis ─────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning enables computers to learn from data without explicit programming. "
                   "Key algorithms include linear regression, decision trees, and neural networks. "
                   "Applications span image recognition, NLP, and recommendation systems.",
    },
    {
        "id": "doc2",
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning uses multi-layered neural networks to model complex patterns. "
                   "Convolutional networks excel at images. Transformers revolutionised NLP. "
                   "Training requires large datasets and significant compute resources.",
    },
    {
        "id": "doc3",
        "title": "Reinforcement Learning",
        "content": "Reinforcement learning trains agents through reward signals. "
                   "Agents learn optimal policies by interacting with environments. "
                   "Famous successes include AlphaGo and OpenAI Five.",
    },
    {
        "id": "doc4",
        "title": "Natural Language Processing",
        "content": "NLP enables computers to understand and generate human language. "
                   "Modern NLP relies on transformer models like BERT and GPT. "
                   "Tasks include translation, sentiment analysis, and question answering.",
    },
]


class DocAnalysisInput(TypedDict):
    doc_id: str
    title: str
    content: str
    # Carry-through fields for the outer graph
    documents: list
    analyses: Annotated[list[dict], operator.add]
    final_report: str


class DocAnalysis(BaseModel):
    key_topics: list[str] = Field(max_length=3)
    sentiment: str
    difficulty: str = Field(description="beginner/intermediate/advanced")
    one_line_summary: str


def fan_out_docs(state: DocAnalysisInput) -> list[Send]:
    """Create one parallel Send per document."""
    sends = []
    for doc in state["documents"]:
        sends.append(Send("analyse_doc", {
            "doc_id":    doc["id"],
            "title":     doc["title"],
            "content":   doc["content"],
            "documents": state["documents"],
            "analyses":  [],
            "final_report": "",
        }))
    return sends


def analyse_doc_node(state: DocAnalysisInput) -> dict:
    """LLM-powered analysis of a single document. Runs in parallel."""
    analysis_llm = llm.with_structured_output(DocAnalysis)
    result = analysis_llm.invoke(
        f"Analyse this document:\nTitle: {state['title']}\nContent: {state['content']}"
    )
    return {
        "analyses": [{
            "doc_id":   state["doc_id"],
            "title":    state["title"],
            "topics":   result.key_topics,
            "sentiment": result.sentiment,
            "difficulty": result.difficulty,
            "summary":  result.one_line_summary,
        }]
    }


def synthesise_node(state: DocAnalysisInput) -> dict:
    """Combine all parallel analyses into a final report."""
    analyses = state["analyses"]
    lines = ["# Document Analysis Report\n"]
    for a in sorted(analyses, key=lambda x: x["doc_id"]):
        lines.append(f"## {a['title']}")
        lines.append(f"- **Topics**: {', '.join(a['topics'])}")
        lines.append(f"- **Difficulty**: {a['difficulty']}")
        lines.append(f"- **Sentiment**: {a['sentiment']}")
        lines.append(f"- **Summary**: {a['summary']}\n")
    return {"final_report": "\n".join(lines)}


def demo_map_reduce():
    graph = StateGraph(DocAnalysisInput)
    graph.add_node("fan_out",    fan_out_docs)
    graph.add_node("analyse_doc", analyse_doc_node)
    graph.add_node("synthesise", synthesise_node)

    graph.set_entry_point("fan_out")
    graph.add_conditional_edges("fan_out", fan_out_docs)   # returns list[Send]
    graph.add_edge("analyse_doc", "synthesise")
    graph.add_edge("synthesise",  END)

    app = graph.compile()

    print("\n=== 2. Map-Reduce: Parallel Document Analysis ===")
    print(f"  Analysing {len(DOCUMENTS)} documents in parallel...\n")

    result = app.invoke({
        "doc_id": "",
        "title": "",
        "content": "",
        "documents": DOCUMENTS,
        "analyses": [],
        "final_report": "",
    })

    print(result["final_report"])
    print(f"  Analyses collected: {len(result['analyses'])}")


# ── 3. Parallel scoring — aggregate with weighted average ─────────────────────
class ScoreState(TypedDict):
    text: str
    criterion: str
    score: float
    # Outer graph fields
    all_scores: Annotated[list[float], operator.add]
    final_score: float
    feedback: str


CRITERIA = [
    ("clarity",      "How clear and easy to understand is this text? Score 0-10."),
    ("completeness", "How complete and comprehensive is this text? Score 0-10."),
    ("accuracy",     "How factually accurate does this text appear? Score 0-10."),
]


class CriterionScore(BaseModel):
    score: float = Field(ge=0, le=10)
    feedback: str


def fan_out_scoring(state: ScoreState) -> list[Send]:
    return [
        Send("score_criterion", {
            "text": state["text"],
            "criterion": criterion,
            "score": 0.0,
            "all_scores": [],
            "final_score": 0.0,
            "feedback": description,
        })
        for criterion, description in CRITERIA
    ]


def score_criterion_node(state: ScoreState) -> dict:
    score_llm = llm.with_structured_output(CriterionScore)
    result = score_llm.invoke(
        f"Criterion: {state['feedback']}\n\nText to evaluate:\n{state['text']}\n\n"
        "Give a score 0-10 and brief feedback."
    )
    return {"all_scores": [result.score]}


def compute_final_score(state: ScoreState) -> dict:
    scores = state["all_scores"]
    avg = sum(scores) / len(scores) if scores else 0
    return {
        "final_score": round(avg, 2),
        "feedback": f"Scores per criterion: {[round(s, 1) for s in scores]}. Average: {avg:.1f}/10",
    }


def demo_parallel_scoring():
    graph = StateGraph(ScoreState)
    graph.add_node("fan_out",         fan_out_scoring)
    graph.add_node("score_criterion", score_criterion_node)
    graph.add_node("final_score",     compute_final_score)

    graph.set_entry_point("fan_out")
    graph.add_conditional_edges("fan_out", fan_out_scoring)
    graph.add_edge("score_criterion", "final_score")
    graph.add_edge("final_score",     END)

    app = graph.compile()

    text = (
        "Machine learning is a powerful technique that allows computers to learn "
        "from data. It is used in many applications such as image recognition, "
        "natural language processing, and recommendation systems."
    )

    print("\n=== 3. Parallel Scoring (Map-Reduce) ===")
    result = app.invoke({
        "text": text, "criterion": "", "score": 0.0,
        "all_scores": [], "final_score": 0.0, "feedback": "",
    })
    print(f"  Text         : {text[:60]}...")
    print(f"  Final Score  : {result['final_score']}/10")
    print(f"  Feedback     : {result['feedback']}")


if __name__ == "__main__":
    demo_basic_fanout()
    demo_map_reduce()
    demo_parallel_scoring()
