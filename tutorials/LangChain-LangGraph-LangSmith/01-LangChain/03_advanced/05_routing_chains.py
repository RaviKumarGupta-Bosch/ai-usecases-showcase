"""
05 - Routing Chains
====================
Route user input to specialised sub-chains based on content.
Two main approaches: rule-based (RunnableBranch) and semantic (embedding similarity).

Topics covered:
  1. RunnableBranch — explicit condition-based routing
  2. Classification-then-route pattern (LLM classifies → route)
  3. Semantic routing — embed query, compare to topic vectors
  4. Per-domain prompt templates
  5. Fallback chain for unmatched routes
  6. Multi-level routing (nested branches)
"""

import numpy as np
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ── Domain-specific prompts ───────────────────────────────────────────────────
def make_domain_chain(domain: str, expertise: str, style: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert {expertise}. {style}. Be concise."),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


science_chain = make_domain_chain(
    "science", "scientist with deep knowledge of physics, chemistry, and biology",
    "Explain with scientific accuracy, use examples from nature"
)
math_chain = make_domain_chain(
    "math", "mathematics professor",
    "Show reasoning step by step. Use equations where helpful"
)
history_chain = make_domain_chain(
    "history", "historian",
    "Provide historical context, dates, and cause-and-effect relationships"
)
technology_chain = make_domain_chain(
    "technology", "senior software engineer and tech expert",
    "Be precise and practical, give code examples when relevant"
)
general_chain = make_domain_chain(
    "general", "helpful and knowledgeable assistant",
    "Answer clearly and concisely for a general audience"
)


# ── 1. RunnableBranch — rule-based routing ────────────────────────────────────
def demo_runnable_branch():
    """
    RunnableBranch takes a list of (condition, runnable) pairs.
    Evaluates conditions in order; first True condition wins.
    Last entry is the default (fallback).
    """
    def classify_simple(x: dict) -> str:
        """Simple keyword-based classifier."""
        q = x["question"].lower()
        if any(w in q for w in ["physics", "chemistry", "biology", "atom", "cell", "molecule"]):
            return "science"
        if any(w in q for w in ["calculate", "equation", "integral", "derivative", "matrix", "probability"]):
            return "math"
        if any(w in q for w in ["history", "war", "empire", "century", "ancient", "revolution"]):
            return "history"
        if any(w in q for w in ["python", "code", "algorithm", "software", "computer", "api", "database"]):
            return "technology"
        return "general"

    router = RunnableBranch(
        (lambda x: classify_simple(x) == "science",    science_chain),
        (lambda x: classify_simple(x) == "math",       math_chain),
        (lambda x: classify_simple(x) == "history",    history_chain),
        (lambda x: classify_simple(x) == "technology", technology_chain),
        general_chain,  # default fallback
    )

    questions = [
        "Explain how photosynthesis works in plant cells",
        "What is the derivative of x² * sin(x)?",
        "What caused the fall of the Roman Empire?",
        "How does a Python decorator work?",
        "What is the meaning of life?",
    ]

    print("=== 1. RunnableBranch (keyword routing) ===")
    for q in questions:
        domain = classify_simple({"question": q})
        answer = router.invoke({"question": q})
        print(f"\n[{domain.upper()}] {q}")
        print(f"  → {answer[:120]}...")


# ── 2. LLM-based classification then route ───────────────────────────────────
class RouteQuery(BaseModel):
    domain: Literal["science", "math", "history", "technology", "general"] = Field(
        description="The domain that best matches the user's question"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")


def demo_llm_routing():
    """
    Use the LLM to classify the query into a domain, then route to specialist.
    More accurate than keyword matching for ambiguous queries.
    """
    classifier = llm.with_structured_output(RouteQuery)

    classify_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Classify the user's question into exactly one domain: "
         "science, math, history, technology, or general. "
         "Choose based on the primary expertise required to answer it."),
        ("human", "{question}"),
    ])

    classify_chain = classify_prompt | classifier

    chains = {
        "science":    science_chain,
        "math":       math_chain,
        "history":    history_chain,
        "technology": technology_chain,
        "general":    general_chain,
    }

    def route_and_answer(question: str) -> dict:
        route: RouteQuery = classify_chain.invoke({"question": question})
        chain = chains[route.domain]
        answer = chain.invoke({"question": question})
        return {
            "domain": route.domain,
            "confidence": route.confidence,
            "answer": answer,
        }

    questions = [
        "What is quantum entanglement?",
        "How do I implement a binary search tree in Python?",
        "Who was Napoleon Bonaparte?",
    ]

    print("\n=== 2. LLM Classification + Routing ===")
    for q in questions:
        result = route_and_answer(q)
        print(f"\nQ: {q}")
        print(f"  Domain ({result['confidence']:.0%}): {result['domain']}")
        print(f"  Answer: {result['answer'][:120]}...")


# ── 3. Semantic routing — embedding similarity ────────────────────────────────
def demo_semantic_routing():
    """
    Embed the query and compare cosine similarity to pre-embedded domain descriptions.
    No LLM call needed for routing — very fast.
    """
    domain_descriptions = {
        "science":    "physics chemistry biology atoms molecules cells organisms natural science",
        "math":       "mathematics equations calculus algebra statistics probability matrix geometry",
        "history":    "historical events ancient civilizations wars empires dates centuries revolutions",
        "technology": "software programming code algorithms computer science APIs databases systems",
        "general":    "general knowledge everyday questions lifestyle opinions advice",
    }

    # Pre-compute domain embeddings
    domain_names = list(domain_descriptions.keys())
    domain_texts = list(domain_descriptions.values())
    domain_vectors = np.array(embeddings.embed_documents(domain_texts))

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return b_norm @ a_norm

    def semantic_route(question: str) -> str:
        q_vec = np.array(embeddings.embed_query(question))
        sims = cosine_similarity(q_vec, domain_vectors)
        return domain_names[int(np.argmax(sims))]

    chains = {
        "science":    science_chain,
        "math":       math_chain,
        "history":    history_chain,
        "technology": technology_chain,
        "general":    general_chain,
    }

    questions = [
        "How does CRISPR gene editing work?",
        "What is the Pythagorean theorem?",
        "When did World War II end?",
        "Explain REST vs GraphQL",
        "What are some good productivity tips?",
    ]

    print("\n=== 3. Semantic Routing (Embedding Similarity) ===")
    for q in questions:
        domain = semantic_route(q)
        answer = chains[domain].invoke({"question": q})
        print(f"\n[→ {domain.upper()}] {q}")
        print(f"  {answer[:120]}...")


# ── 4. Multi-level routing ────────────────────────────────────────────────────
def demo_multi_level_routing():
    """
    First route: is it technical or non-technical?
    Second route (within technical): frontend, backend, or ML?
    """
    frontend_chain = make_domain_chain("frontend", "frontend developer", "Focus on HTML, CSS, JavaScript, React, UX")
    backend_chain  = make_domain_chain("backend",  "backend developer",  "Focus on APIs, databases, performance, scalability")
    ml_chain       = make_domain_chain("ML",       "ML engineer",        "Focus on models, training, data pipelines, evaluation")

    class TechSubCategory(BaseModel):
        sub_domain: Literal["frontend", "backend", "machine_learning", "general_tech"]

    sub_classifier = llm.with_structured_output(TechSubCategory)
    sub_classify_prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify this tech question as: frontend, backend, machine_learning, or general_tech."),
        ("human", "{question}"),
    ])
    sub_chain = sub_classify_prompt | sub_classifier

    tech_sub_chains = {
        "frontend":         frontend_chain,
        "backend":          backend_chain,
        "machine_learning": ml_chain,
        "general_tech":     technology_chain,
    }

    class Category(BaseModel):
        is_technical: bool

    level1_classifier = llm.with_structured_output(Category)
    level1_prompt = ChatPromptTemplate.from_messages([
        ("system", "Determine if this question is primarily a technical/programming question (true) or not (false)."),
        ("human", "{question}"),
    ])
    level1_chain = level1_prompt | level1_classifier

    def route_two_level(question: str) -> dict:
        l1 = level1_chain.invoke({"question": question})
        if not l1.is_technical:
            answer = general_chain.invoke({"question": question})
            return {"path": "general", "answer": answer}

        sub = sub_chain.invoke({"question": question})
        answer = tech_sub_chains[sub.sub_domain].invoke({"question": question})
        return {"path": f"tech → {sub.sub_domain}", "answer": answer}

    questions = [
        "How do I use React hooks?",
        "What is database connection pooling?",
        "Explain batch normalisation in neural networks",
        "What's the best way to stay focused while working from home?",
    ]

    print("\n=== 4. Multi-Level Routing ===")
    for q in questions:
        result = route_two_level(q)
        print(f"\nQ: {q}")
        print(f"  Route : {result['path']}")
        print(f"  Answer: {result['answer'][:120]}...")


if __name__ == "__main__":
    demo_runnable_branch()
    demo_llm_routing()
    demo_semantic_routing()
    demo_multi_level_routing()
