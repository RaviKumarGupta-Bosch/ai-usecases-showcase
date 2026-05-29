"""
03 - Datasets & Experiments (LangSmith)
==========================================
LangSmith datasets store input/output pairs for:
- Regression testing (did the model get worse?)
- A/B experiments (which prompt is better?)
- Fine-tuning data curation
- Systematic quality tracking over time

Topics covered:
  1. Creating and managing datasets
  2. Adding examples from code and from production traces
  3. Running experiments (evaluate against a dataset)
  4. Comparing multiple experiments
  5. Versioning and updating datasets
  6. Curating datasets from LangSmith traces
  7. Generating a performance trend report
"""

import os
import json
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_client() -> Optional[Client]:
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("  [Skip] LANGCHAIN_API_KEY not set")
        return None
    try:
        return Client()
    except Exception as e:
        print(f"  [Error] {e}")
        return None


# ── Dataset definitions ───────────────────────────────────────────────────────
SUMMARISATION_DATASET = [
    {
        "input": {
            "text": "Machine learning is a subset of artificial intelligence that enables computers "
                    "to learn and improve from experience without being explicitly programmed. "
                    "It focuses on developing computer programs that can access data and use it to learn for themselves."
        },
        "output": {"summary": "Machine learning lets computers learn from data without explicit programming."},
    },
    {
        "input": {
            "text": "Neural networks are computing systems inspired by the biological neural networks "
                    "that constitute animal brains. They consist of layers of interconnected nodes "
                    "that process information and learn patterns from data."
        },
        "output": {"summary": "Neural networks are brain-inspired computing systems that learn patterns through interconnected layers."},
    },
    {
        "input": {
            "text": "Natural language processing (NLP) is a branch of artificial intelligence that "
                    "helps computers understand, interpret, and manipulate human language. "
                    "NLP draws from many disciplines, including computer science and computational linguistics."
        },
        "output": {"summary": "NLP enables computers to understand and process human language using AI and linguistics."},
    },
    {
        "input": {
            "text": "Reinforcement learning is a type of machine learning where an agent learns to "
                    "make decisions by performing actions in an environment to maximise cumulative reward. "
                    "The agent learns from consequences of its actions, not from being told what to do."
        },
        "output": {"summary": "Reinforcement learning trains agents to maximise rewards through trial-and-error in an environment."},
    },
    {
        "input": {
            "text": "Transfer learning is a machine learning technique where a model developed for "
                    "a task is reused as the starting point for a model on a different task. "
                    "It dramatically reduces training time and data requirements."
        },
        "output": {"summary": "Transfer learning reuses a pre-trained model for a new task, saving time and data."},
    },
]

TRANSLATION_DATASET = [
    {"input": {"text": "Hello, how are you?",     "target_lang": "Spanish"}, "output": {"translation": "Hola, ¿cómo estás?"}},
    {"input": {"text": "The weather is nice today.", "target_lang": "French"},  "output": {"translation": "Le temps est beau aujourd'hui."}},
    {"input": {"text": "I love programming.",      "target_lang": "German"},   "output": {"translation": "Ich liebe das Programmieren."}},
]


# ── 1. Create & populate datasets ────────────────────────────────────────────
def demo_create_datasets():
    client = get_client()
    if not client:
        return None, None

    print("=== 1. Creating Datasets ===")
    datasets = {}

    for name, data, description in [
        ("ai-summarisation-v1", SUMMARISATION_DATASET, "AI concept summarisation benchmark"),
        ("ai-translation-v1",   TRANSLATION_DATASET,   "Basic translation evaluation set"),
    ]:
        try:
            ds = client.read_dataset(dataset_name=name)
            print(f"  Dataset '{name}' exists (id={str(ds.id)[:8]}...)")
            datasets[name] = ds
        except Exception:
            ds = client.create_dataset(dataset_name=name, description=description)
            client.create_examples(
                inputs=[ex["input"] for ex in data],
                outputs=[ex["output"] for ex in data],
                dataset_id=ds.id,
            )
            print(f"  Created '{name}' with {len(data)} examples")
            datasets[name] = ds

    return "ai-summarisation-v1", "ai-translation-v1"


# ── 2. Summarisation chains (v1 and v2 for comparison) ───────────────────────
def summarise_v1(inputs: dict) -> dict:
    """Version 1: Simple summarisation prompt."""
    chain = (
        ChatPromptTemplate.from_template("Summarise in one sentence: {text}")
        | llm
        | StrOutputParser()
    )
    return {"summary": chain.invoke({"text": inputs["text"]})}


def summarise_v2(inputs: dict) -> dict:
    """Version 2: Chain-of-thought summarisation prompt."""
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating concise, accurate summaries. "
                       "Identify the core concept and state it in a single clear sentence."),
            ("human", "Summarise this text:\n\n{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    return {"summary": chain.invoke({"text": inputs["text"]})}


# ── 3. Custom evaluator for summarisation ────────────────────────────────────
class SummaryScore(BaseModel):
    faithfulness: float = Field(ge=0, le=1, description="Is the summary factually faithful to the source?")
    conciseness:  float = Field(ge=0, le=1, description="Is the summary appropriately brief?")
    coverage:     float = Field(ge=0, le=1, description="Does the summary capture the main idea?")
    overall:      float = Field(ge=0, le=1, description="Overall quality score")


def summary_quality_evaluator(run: Run, example: Example) -> dict:
    """Multi-dimensional summary quality evaluator."""
    prediction = (run.outputs or {}).get("summary", "")
    source_text = (example.inputs or {}).get("text", "")
    reference   = (example.outputs or {}).get("summary", "")

    if not prediction:
        return {"key": "summary_quality", "score": 0}

    result = llm_judge.with_structured_output(SummaryScore).invoke(
        f"""Evaluate this summary against the source text.

Source: {source_text}
Reference summary: {reference}
Predicted summary: {prediction}

Score each dimension 0-1."""
    )

    overall_score = (result.faithfulness + result.conciseness + result.coverage + result.overall) / 4
    return {
        "key":   "summary_quality",
        "score": overall_score,
        "comment": f"F={result.faithfulness:.2f} C={result.conciseness:.2f} Cov={result.coverage:.2f}",
    }


def length_check_evaluator(run: Run, example: Example) -> dict:
    """Check that the summary is shorter than the source."""
    source   = (example.inputs  or {}).get("text",    "")
    summary  = (run.outputs or {}).get("summary", "")
    is_shorter = len(summary.split()) < len(source.split())
    return {
        "key":   "is_shorter",
        "score": 1.0 if is_shorter else 0.0,
        "comment": f"Source={len(source.split())}w Summary={len(summary.split())}w",
    }


# ── 4. Run experiments ────────────────────────────────────────────────────────
def demo_run_experiments(dataset_name: str):
    client = get_client()
    if not client:
        return

    print(f"\n=== 3. Running Experiments on '{dataset_name}' ===")
    ts = datetime.now().strftime("%H%M%S")

    print("  Running V1 experiment...")
    results_v1 = evaluate(
        summarise_v1,
        data=dataset_name,
        evaluators=[summary_quality_evaluator, length_check_evaluator],
        experiment_prefix=f"summ-v1-{ts}",
        metadata={"version": "1", "prompt": "simple"},
    )

    print("  Running V2 experiment...")
    results_v2 = evaluate(
        summarise_v2,
        data=dataset_name,
        evaluators=[summary_quality_evaluator, length_check_evaluator],
        experiment_prefix=f"summ-v2-{ts}",
        metadata={"version": "2", "prompt": "cot"},
    )

    print(f"  V1 URL: {getattr(results_v1, 'url', 'See LangSmith UI')}")
    print(f"  V2 URL: {getattr(results_v2, 'url', 'See LangSmith UI')}")
    print("  Tip: Open LangSmith → Datasets → ai-summarisation-v1 → Compare experiments")


# ── 5. Add examples from production traces ────────────────────────────────────
@traceable(name="production_summariser", tags=["production"])
def production_summariser(text: str) -> str:
    """Simulates a production function that we later add to a dataset."""
    chain = (
        ChatPromptTemplate.from_template("Summarise in one sentence: {text}")
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"text": text})


def demo_curate_from_traces(dataset_name: str):
    client = get_client()
    if not client:
        return

    print(f"\n=== 4. Curating Dataset from Production Traces ===")

    # Run some "production" requests (these get traced)
    new_texts = [
        "Gradient descent is an optimisation algorithm that iteratively adjusts parameters "
        "to minimise a loss function by moving in the direction of the steepest descent.",
        "Overfitting occurs when a model learns the training data too well, including noise, "
        "resulting in poor performance on unseen data.",
    ]

    for text in new_texts:
        summary = production_summariser(text)
        print(f"  Generated summary: {summary[:60]}...")

    # Fetch recent traces for "production_summariser"
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    try:
        runs = list(client.list_runs(
            project_name=project,
            run_type="chain",
            filter='has(tags, "production")',
            limit=5,
        ))
        print(f"  Found {len(runs)} production traces")

        if runs:
            dataset = client.read_dataset(dataset_name=dataset_name)
            added = 0
            for run in runs[:2]:  # add up to 2 new examples
                if run.inputs and run.outputs:
                    client.create_examples(
                        inputs=[run.inputs],
                        outputs=[run.outputs],
                        dataset_id=dataset.id,
                    )
                    added += 1
            print(f"  Added {added} examples from traces to '{dataset_name}'")
    except Exception as e:
        print(f"  [Note] Trace curation: {e}")


# ── 6. List experiments and show trend ────────────────────────────────────────
def demo_experiment_trend(dataset_name: str):
    client = get_client()
    if not client:
        return

    print(f"\n=== 5. Experiment History ===")
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        tests = list(client.list_tests(dataset_id=dataset.id))
        print(f"  Total experiments for '{dataset_name}': {len(tests)}")
        for test in tests[-5:]:  # last 5
            print(f"  - {getattr(test, 'name', str(test)[:40])} | {getattr(test, 'created_at', '')}")
    except Exception as e:
        print(f"  [Error] {e}")


# ── 7. Offline experiment (no LangSmith API) ─────────────────────────────────
def demo_offline_experiment():
    """Run a local experiment and save results to JSON."""
    print("\n=== 6. Offline Experiment (Local JSON) ===")

    class LocalScore(BaseModel):
        quality: float = Field(ge=0, le=1)
        comment: str

    results = []
    for ex in SUMMARISATION_DATASET:
        text = ex["input"]["text"]
        reference = ex["output"]["summary"]

        # V1
        summary_v1 = summarise_v1({"text": text})["summary"]
        score_v1   = llm_judge.with_structured_output(LocalScore).invoke(
            f"Rate this summary quality (0-1).\nSource: {text}\nSummary: {summary_v1}"
        )

        # V2
        summary_v2 = summarise_v2({"text": text})["summary"]
        score_v2   = llm_judge.with_structured_output(LocalScore).invoke(
            f"Rate this summary quality (0-1).\nSource: {text}\nSummary: {summary_v2}"
        )

        results.append({
            "text":       text[:60] + "...",
            "v1_score":   score_v1.quality,
            "v2_score":   score_v2.quality,
            "v1_summary": summary_v1[:60],
            "v2_summary": summary_v2[:60],
        })

        print(f"  {text[:50]}...")
        print(f"    V1 ({score_v1.quality:.2f}): {summary_v1[:60]}")
        print(f"    V2 ({score_v2.quality:.2f}): {summary_v2[:60]}")

    avg_v1 = sum(r["v1_score"] for r in results) / len(results)
    avg_v2 = sum(r["v2_score"] for r in results) / len(results)
    print(f"\n  Average V1: {avg_v1:.2f}  |  Average V2: {avg_v2:.2f}")
    winner = "V2" if avg_v2 > avg_v1 else "V1"
    print(f"  Winner: {winner}")


if __name__ == "__main__":
    # Always run offline demo first (no API key needed)
    demo_offline_experiment()

    # LangSmith demos (require API key)
    summ_ds, trans_ds = demo_create_datasets()
    if summ_ds:
        demo_run_experiments(summ_ds)
        demo_curate_from_traces(summ_ds)
        demo_experiment_trend(summ_ds)
