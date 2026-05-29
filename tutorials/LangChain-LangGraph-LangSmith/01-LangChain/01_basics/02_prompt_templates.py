"""
02 - Prompt Templates
=====================
Prompt templates let you parameterise and reuse prompts cleanly.

Topics covered:
  1. PromptTemplate  — string-based, with named variables
  2. ChatPromptTemplate — message-based (system + human + AI turns)
  3. FewShotPromptTemplate — inject examples automatically
  4. MessagesPlaceholder — insert dynamic chat history
  5. Partial prompt templates — pre-fill some variables
  6. Piping templates directly into LLMs (LCEL preview)
"""

from dotenv import load_dotenv
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. PromptTemplate ────────────────────────────────────────────────────────
def demo_prompt_template():
    # Define a reusable template with {variables}
    template = PromptTemplate.from_template(
        "Summarise the following {topic} in exactly {num_sentences} sentence(s):\n\n{text}"
    )

    # Inspect the prompt string before sending to the LLM
    prompt_text = template.format(
        topic="machine learning",
        num_sentences=1,
        text=(
            "Machine learning is a branch of AI that gives computers the ability "
            "to learn from data without being explicitly programmed."
        ),
    )
    print("=== 1. PromptTemplate ===")
    print("Formatted prompt:\n", prompt_text)

    # Pipe template → llm (LCEL style)
    chain = template | llm
    response = chain.invoke({
        "topic": "Python",
        "num_sentences": 2,
        "text": "Python is a high-level, interpreted programming language known for its readability.",
    })
    print("LLM answer:", response.content)


# ── 2. ChatPromptTemplate ────────────────────────────────────────────────────
def demo_chat_prompt_template():
    # Tuples: ("role", "content with {variables}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}. Always respond in formal {language}."),
        ("human", "{question}"),
    ])

    print("\n=== 2. ChatPromptTemplate ===")
    messages = prompt.format_messages(
        role="senior data scientist",
        language="English",
        question="What is overfitting in machine learning?",
    )
    print("Formatted messages:")
    for m in messages:
        print(f"  [{m.type}] {m.content[:80]}")

    chain = prompt | llm
    response = chain.invoke({
        "role": "machine learning expert",
        "language": "English",
        "question": "Explain regularisation in one paragraph.",
    })
    print("\nLLM answer:", response.content)


# ── 3. FewShotPromptTemplate ─────────────────────────────────────────────────
def demo_few_shot():
    # Examples to inject into the prompt
    examples = [
        {"word": "happy",  "antonym": "sad"},
        {"word": "fast",   "antonym": "slow"},
        {"word": "bright", "antonym": "dark"},
    ]

    example_prompt = PromptTemplate.from_template("Word: {word}\nAntonym: {antonym}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of each word.\n",
        suffix="Word: {word}\nAntonym:",
        input_variables=["word"],
    )

    print("\n=== 3. FewShotPromptTemplate ===")
    print(few_shot_prompt.format(word="strong"))

    chain = few_shot_prompt | llm
    response = chain.invoke({"word": "noisy"})
    print(f"\nAntonym of 'noisy': {response.content.strip()}")


# ── 4. MessagesPlaceholder (dynamic chat history) ────────────────────────────
def demo_messages_placeholder():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on the conversation so far."),
        MessagesPlaceholder(variable_name="history"),   # ← injected at runtime
        ("human", "{input}"),
    ])

    # Simulated history from earlier turns
    chat_history = [
        HumanMessage(content="I'm building a recommendation system."),
        AIMessage(content="Exciting! Are you using collaborative or content-based filtering?"),
    ]

    chain = prompt | llm
    response = chain.invoke({
        "history": chat_history,
        "input": "What algorithm should I start with?",
    })

    print("\n=== 4. MessagesPlaceholder ===")
    print("Response:", response.content)


# ── 5. Partial prompt templates ──────────────────────────────────────────────
def demo_partial_prompt():
    from datetime import datetime

    # Some variables known at definition time, others provided later
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Today is {date}. You are a helpful assistant."),
        ("human", "{question}"),
    ])

    # Pre-fill {date} at construction time
    dated_prompt = prompt.partial(date=datetime.now().strftime("%B %d, %Y"))

    chain = dated_prompt | llm
    response = chain.invoke({"question": "What is today's date?"})

    print("\n=== 5. Partial Prompt Templates ===")
    print("Response:", response.content)


# ── 6. Prompt composition ─────────────────────────────────────────────────────
def demo_prompt_composition():
    # Build complex prompts by combining smaller pieces
    persona = "You are a {persona}."
    instruction = "Your task: {task}"
    format_hint = "Respond as a {format}."

    combined = ChatPromptTemplate.from_messages([
        ("system", f"{persona} {instruction} {format_hint}"),
        ("human", "{query}"),
    ])

    chain = combined | llm
    response = chain.invoke({
        "persona": "financial analyst",
        "task": "answer questions about investment strategies",
        "format": "concise bullet list (max 4 bullets)",
        "query": "What are the key risks in emerging market equities?",
    })

    print("\n=== 6. Prompt Composition ===")
    print(response.content)


if __name__ == "__main__":
    demo_prompt_template()
    demo_chat_prompt_template()
    demo_few_shot()
    demo_messages_placeholder()
    demo_partial_prompt()
    demo_prompt_composition()
