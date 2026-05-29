"""
02 - Evaluation with LangSmith
================================
LangSmith evaluation lets you measure LLM app quality with:
- Built-in evaluators (correctness, conciseness, criteria)
- Custom evaluators (domain-specific metrics)
- Pairwise comparison (A vs B prompt versions)

Topics covered:
  1. Creating a dataset for evaluation
  2. evaluate() with LangChainStringEvaluator
  3. Custom evaluator functions
  4. Evaluating a RAG pipeline
  5. Pairwise (A/B) comparison
  6. Interpreting results and scores
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_client() -> Optional[Client]:
    """Return a LangSmith client, or None if not configured."""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("  [Skip] LANGCHAIN_API_KEY not set — LangSmith evaluation unavailable")
        return None
    try:
        return Client()
    except Exception as e:
        print(f"  [Error] Cannot connect to LangSmith: {e}")
        return None


# ── QA dataset ────────────────────────────────────────────────────────────────
QA_EXAMPLES = [
    {
        "question": "What is LangChain?",
        "answer":   "LangChain is an open-source framework for building applications powered by large language models.",
    },
    {
        "question": "What does LangGraph add to LangChain?",
        "answer":   "LangGraph adds support for stateful, cyclic workflows with checkpointing, enabling complex multi-step agents.",
    },
    {
        "question": "What is LangSmith used for?",
        "answer":   "LangSmith is used for debugging, testing, evaluating, and monitoring LLM applications.",
    },
    {
        "question": "What is RAG?",
        "answer":   "RAG (Retrieval-Augmented Generation) combines a retrieval system with an LLM to answer questions using relevant documents.",
    },
    {
        "question": "What is a vector embedding?",
        "answer":   "A vector embedding is a numerical representation of text in a high-dimensional space where semantically similar texts are close together.",
    },
]


# ── 1. Creating an evaluation dataset ────────────────────────────────────────
def demo_create_dataset():
    client = get_client()
    if not client:
        return None

    dataset_name = "langchain-qa-tutorial"

    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        print(f"\n=== 1. Dataset: '{dataset_name}' already exists ({existing.id}) ===")
        return dataset_name
    except Exception:
        pass

    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="QA pairs about LangChain ecosystem for tutorial evaluation",
    )

    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in QA_EXAMPLES],
        outputs=[{"answer": ex["answer"]} for ex in QA_EXAMPLES],
        dataset_id=dataset.id,
    )

    print(f"\n=== 1. Dataset Created ===")
    print(f"  Name   : {dataset_name}")
    print(f"  ID     : {dataset.id}")
    print(f"  Examples: {len(QA_EXAMPLES)}")
    return dataset_name


# ── 2. Target function (the LLM app to evaluate) ─────────────────────────────
def qa_chain(inputs: dict) -> dict:
    """The LLM application we want to evaluate."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant specializing in LangChain ecosystem. "
                   "Answer concisely in 1-2 sentences."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": inputs["question"]})
    return {"answer": answer}


# ── 3. Built-in evaluators ────────────────────────────────────────────────────
def demo_builtin_evaluators(dataset_name: str):
    """
    LangChainStringEvaluator wraps OpenAI-based evaluators:
    - "qa"           — checks if answer is correct based on reference
    - "criteria"     — checks specific criteria (conciseness, harmfulness, etc.)
    - "embedding_distance" — semantic similarity to reference
    """
    client = get_client()
    if not client:
        return

    print(f"\n=== 2. Built-in Evaluators ===")
    print(f"  Running evaluation on '{dataset_name}'...")

    # QA correctness evaluator (compares answer to reference)
    qa_evaluator = LangChainStringEvaluator(
        "qa",
        config={"llm": llm},
        prepare_data=lambda run, example: {
            "prediction": run.outputs.get("answer", ""),
            "reference":  example.outputs.get("answer", ""),
            "input":      example.inputs.get("question", ""),
        },
    )

    # Conciseness criteria evaluator
    conciseness_evaluator = LangChainStringEvaluator(
        "criteria",
        config={
            "criteria": "conciseness",
            "llm": llm,
        },
        prepare_data=lambda run, example: {
            "prediction": run.outputs.get("answer", ""),
            "input":      example.inputs.get("question", ""),
        },
    )

    results = evaluate(
        qa_chain,
        data=dataset_name,
        evaluators=[qa_evaluator, conciseness_evaluator],
        experiment_prefix="baseline-gpt4o-mini",
        metadata={"model": "gpt-4o-mini", "temperature": 0},
    )

    print(f"  Evaluation complete!")
    print(f"  Results URL: {results.url if hasattr(results, 'url') else 'See LangSmith UI'}")


# ── 4. Custom evaluator function ─────────────────────────────────────────────
class EvaluationScore(BaseModel):
    score: float = Field(ge=0, le=1)
    reasoning: str


def accuracy_evaluator(run: Run, example: Example) -> dict:
    """
    Custom evaluator that uses an LLM judge.
    Must return a dict with 'key' and 'score'.
    """
    prediction = (run.outputs or {}).get("answer", "")
    reference  = (example.outputs or {}).get("answer", "")
    question   = (example.inputs or {}).get("question", "")

    if not prediction or not reference:
        return {"key": "accuracy", "score": 0}

    judge_llm = llm.with_structured_output(EvaluationScore)

    result = judge_llm.invoke(
        f"""You are an expert evaluator. Score the predicted answer vs the reference answer.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Score 0-1 where:
  1.0 = Fully correct and complete
  0.7 = Mostly correct, minor gaps
  0.4 = Partially correct
  0.0 = Incorrect or missing"""
    )

    return {
        "key":       "accuracy",
        "score":     result.score,
        "comment":   result.reasoning,
    }


def length_evaluator(run: Run, example: Example) -> dict:
    """Custom evaluator: penalise very long answers."""
    answer = (run.outputs or {}).get("answer", "")
    word_count = len(answer.split())

    if word_count <= 30:
        score = 1.0
    elif word_count <= 60:
        score = 0.7
    elif word_count <= 100:
        score = 0.4
    else:
        score = 0.2

    return {
        "key":     "brevity",
        "score":   score,
        "comment": f"Answer has {word_count} words",
    }


def demo_custom_evaluators(dataset_name: str):
    client = get_client()
    if not client:
        return

    print(f"\n=== 3. Custom Evaluators ===")
    print(f"  Running custom evaluation on '{dataset_name}'...")

    results = evaluate(
        qa_chain,
        data=dataset_name,
        evaluators=[accuracy_evaluator, length_evaluator],
        experiment_prefix="custom-eval-gpt4o-mini",
        metadata={"evaluator": "custom-llm-judge"},
    )

    print(f"  Custom evaluation complete!")
    print(f"  Results: {results.url if hasattr(results, 'url') else 'See LangSmith UI'}")


# ── 5. A/B comparison — two prompt versions ───────────────────────────────────
def qa_chain_v2(inputs: dict) -> dict:
    """Version 2: more detailed system prompt."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior AI engineer specializing in the LangChain ecosystem. "
         "Provide accurate, technical answers with concrete examples when possible. "
         "Be specific and avoid vague statements."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"answer": chain.invoke({"question": inputs["question"]})}


def demo_ab_comparison(dataset_name: str):
    client = get_client()
    if not client:
        return

    print(f"\n=== 4. A/B Comparison ===")

    # Evaluate both versions
    results_v1 = evaluate(
        qa_chain,
        data=dataset_name,
        evaluators=[accuracy_evaluator, length_evaluator],
        experiment_prefix="v1-minimal-prompt",
    )

    results_v2 = evaluate(
        qa_chain_v2,
        data=dataset_name,
        evaluators=[accuracy_evaluator, length_evaluator],
        experiment_prefix="v2-detailed-prompt",
    )

    print(f"  V1 results: {results_v1.url if hasattr(results_v1, 'url') else 'See LangSmith UI'}")
    print(f"  V2 results: {results_v2.url if hasattr(results_v2, 'url') else 'See LangSmith UI'}")
    print("  Compare both experiments in LangSmith UI → Experiments → Compare")


# ── 6. Offline evaluation (no LangSmith API required) ────────────────────────
def demo_offline_evaluation():
    """
    You can run evaluations locally without LangSmith.
    Useful for CI/CD or air-gapped environments.
    """
    print("\n=== 5. Offline / Local Evaluation ===")

    judge_llm = llm.with_structured_output(EvaluationScore)

    scores = []
    for ex in QA_EXAMPLES:
        # Generate answer
        result = qa_chain({"question": ex["question"]})
        prediction = result["answer"]

        # Score locally
        score_result = judge_llm.invoke(
            f"Score this answer (0-1) for correctness.\n"
            f"Question: {ex['question']}\n"
            f"Reference: {ex['answer']}\n"
            f"Prediction: {prediction}"
        )

        scores.append(score_result.score)
        print(f"  Q: {ex['question'][:50]}...")
        print(f"  A: {prediction[:80]}...")
        print(f"  Score: {score_result.score:.2f} | {score_result.reasoning[:60]}")
        print()

    avg_score = sum(scores) / len(scores)
    print(f"  Average accuracy score: {avg_score:.2f}")


if __name__ == "__main__":
    # Offline evaluation always works
    demo_offline_evaluation()

    # LangSmith-connected demos (require LANGCHAIN_API_KEY)
    dataset_name = demo_create_dataset()
    if dataset_name:
        demo_builtin_evaluators(dataset_name)
        demo_custom_evaluators(dataset_name)
        demo_ab_comparison(dataset_name)
