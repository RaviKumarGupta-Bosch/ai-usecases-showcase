"""
03 - LCEL — LangChain Expression Language
==========================================
LCEL is the declarative way to compose chains using the pipe `|` operator.
It gives you streaming, async, batching, and parallelism for free.

Topics covered:
  1. The pipe operator |
  2. RunnablePassthrough  — pass input unchanged
  3. RunnableParallel     — run multiple branches in parallel
  4. RunnableLambda       — wrap any Python function
  5. itemgetter           — extract dict fields
  6. .bind()              — fix parameters
  7. .with_fallbacks()    — graceful error handling
  8. .with_retry()        — automatic retries
  9. Branching / routing with RunnableBranch
 10. inspect chain schema (input / output)
"""

from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
)
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# ── 1. Minimal chain ─────────────────────────────────────────────────────────
def demo_minimal_chain():
    prompt = ChatPromptTemplate.from_template("Translate '{text}' to {language}.")
    chain = prompt | llm | parser

    result = chain.invoke({"text": "Hello, world!", "language": "French"})
    print("=== 1. Minimal Chain ===")
    print(result)


# ── 2. RunnablePassthrough ───────────────────────────────────────────────────
def demo_passthrough():
    prompt = ChatPromptTemplate.from_template(
        "Answer in one sentence.\n\nContext: {context}\n\nQuestion: {question}"
    )

    # Pass question through unchanged while retrieving context separately
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: f"The capital of France is Paris. Population: 2.2M."
        )
        | prompt
        | llm
        | parser
    )

    result = chain.invoke({"question": "What is the capital of France?"})
    print("\n=== 2. RunnablePassthrough ===")
    print(result)


# ── 3. RunnableParallel ──────────────────────────────────────────────────────
def demo_parallel():
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarise in one sentence: {text}"
    )
    tone_prompt = ChatPromptTemplate.from_template(
        "Detect the tone (positive/negative/neutral) of: {text}"
    )
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 3 keywords from: {text}"
    )

    # All three branches run in parallel
    parallel_chain = RunnableParallel(
        summary=summary_prompt | llm | parser,
        tone=tone_prompt | llm | parser,
        keywords=keywords_prompt | llm | parser,
    )

    text = (
        "The new AI model exceeded expectations on all benchmarks, "
        "delivering stunning performance with significantly lower energy consumption."
    )
    results = parallel_chain.invoke({"text": text})

    print("\n=== 3. RunnableParallel ===")
    print(f"Summary  : {results['summary']}")
    print(f"Tone     : {results['tone']}")
    print(f"Keywords : {results['keywords']}")


# ── 4. RunnableLambda ─────────────────────────────────────────────────────────
def demo_lambda():
    def word_count(text: str) -> str:
        n = len(text.split())
        return f"{text}\n\n[Word count: {n}]"

    prompt = ChatPromptTemplate.from_template(
        "Write a 2-sentence description of {topic}."
    )

    chain = (
        prompt
        | llm
        | parser
        | RunnableLambda(word_count)   # post-process the LLM output
    )

    result = chain.invoke({"topic": "quantum computing"})
    print("\n=== 4. RunnableLambda ===")
    print(result)


# ── 5. itemgetter — extract dict keys ────────────────────────────────────────
def demo_itemgetter():
    prompt = ChatPromptTemplate.from_template(
        "In {style}, explain: {concept}"
    )

    # itemgetter pulls named fields from a dict input
    chain = (
        {"style": itemgetter("style"), "concept": itemgetter("concept")}
        | prompt
        | llm
        | parser
    )

    result = chain.invoke({
        "style": "simple analogies for a 10-year-old",
        "concept": "neural networks",
    })
    print("\n=== 5. itemgetter ===")
    print(result)


# ── 6. .bind() — fix parameters at chain build time ──────────────────────────
def demo_bind():
    prompt = ChatPromptTemplate.from_template("Classify the sentiment of: {text}")

    # Bind stop sequences and max_tokens directly on the llm
    constrained_llm = llm.bind(stop=["\n"], max_tokens=10)
    chain = prompt | constrained_llm | parser

    result = chain.invoke({"text": "This product is absolutely amazing!"})
    print("\n=== 6. .bind() ===")
    print(f"Sentiment: {result}")


# ── 7. .with_fallbacks() — handle failures gracefully ────────────────────────
def demo_fallbacks():
    # Simulate a bad LLM (wrong model name) that should fall back to a working one
    bad_llm = ChatOpenAI(model="gpt-99-turbo", temperature=0)  # doesn't exist
    good_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("What is 1 + 1?")
    chain_with_fallback = (prompt | bad_llm | parser).with_fallbacks(
        [prompt | good_llm | parser]
    )

    result = chain_with_fallback.invoke({})
    print("\n=== 7. .with_fallbacks() ===")
    print(f"Answer (via fallback): {result}")


# ── 8. RunnableBranch — conditional routing ──────────────────────────────────
def demo_branch():
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Answer this programming question:\n{question}"
    )
    math_prompt = ChatPromptTemplate.from_template(
        "You are a math tutor. Solve step by step:\n{question}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer:\n{question}"
    )

    def classify(x: dict) -> str:
        q = x["question"].lower()
        if any(k in q for k in ["code", "python", "function", "algorithm", "class"]):
            return "code"
        if any(k in q for k in ["calculate", "solve", "equation", "integral", "derivative"]):
            return "math"
        return "general"

    branch = RunnableBranch(
        (lambda x: classify(x) == "code",    code_prompt    | llm | parser),
        (lambda x: classify(x) == "math",    math_prompt    | llm | parser),
        general_prompt | llm | parser,  # default
    )

    questions = [
        {"question": "Write a Python function to reverse a string."},
        {"question": "Calculate the derivative of x^3 + 2x."},
        {"question": "What is the speed of light?"},
    ]

    print("\n=== 8. RunnableBranch ===")
    for q in questions:
        result = branch.invoke(q)
        print(f"\nQ: {q['question']}")
        print(f"A: {result[:120]}...")


# ── 9. Inspect chain input/output schema ─────────────────────────────────────
def demo_schema():
    prompt = ChatPromptTemplate.from_template("Describe {topic} in {n} words.")
    chain = prompt | llm | parser

    print("\n=== 9. Chain Schema ===")
    print("Input schema :", chain.input_schema.schema())
    print("Output schema:", chain.output_schema.schema())


if __name__ == "__main__":
    demo_minimal_chain()
    demo_passthrough()
    demo_parallel()
    demo_lambda()
    demo_itemgetter()
    demo_bind()
    demo_fallbacks()
    demo_branch()
    demo_schema()
