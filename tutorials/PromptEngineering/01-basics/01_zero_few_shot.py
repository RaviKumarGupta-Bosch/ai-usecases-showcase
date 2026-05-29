"""
Prompt Engineering 01-basics — Zero-shot, One-shot, Few-shot
=============================================================
Topics covered:
  1. Zero-shot prompting
  2. One-shot prompting (single example)
  3. Few-shot prompting (multiple examples)
  4. Comparing output quality across shot counts
  5. Format steering with few-shot examples

Run:
  python 01_zero_few_shot.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def complete(system: str, user: str, temperature: float = 0.0) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content.strip()


# ── 1. Zero-shot ──────────────────────────────────────────────────────────────
def demo_zero_shot():
    print("\n=== 1. Zero-Shot Prompting ===")

    # Zero-shot: no examples — model relies on pretrained knowledge
    tasks = [
        ("sentiment", "Classify the sentiment: 'I absolutely love this product!'"),
        ("translation", "Translate to French: 'The weather is beautiful today.'"),
        ("summary", "Summarise in one sentence: 'Quantum computing uses qubits that can be in superposition of 0 and 1, allowing exponentially more computation than classical bits.'"),
    ]

    for task_name, prompt in tasks:
        result = complete("You are a helpful assistant.", prompt)
        print(f"\n[{task_name}]")
        print(f"Prompt: {prompt}")
        print(f"Output: {result}")


# ── 2. One-shot ───────────────────────────────────────────────────────────────
def demo_one_shot():
    print("\n=== 2. One-Shot Prompting ===")

    # One example in the prompt guides format and style
    system = "You are a product reviewer."

    one_shot_prompt = """Write a product review in this format:
PRODUCT: Wireless headphones
PROS: Great sound quality, comfortable fit, long battery life
CONS: Slightly expensive, no carry case included
VERDICT: ★★★★☆ Excellent choice for music lovers

Now write a review for:
PRODUCT: Mechanical keyboard"""

    result = complete(system, one_shot_prompt)
    print(f"Output:\n{result}")


# ── 3. Few-shot classification ────────────────────────────────────────────────
def demo_few_shot_classification():
    print("\n=== 3. Few-Shot Classification ===")

    few_shot_prompt = """Classify each customer message as: COMPLAINT, QUESTION, COMPLIMENT, or REFUND_REQUEST.

Message: "My order arrived broken and the packaging was damaged."
Category: COMPLAINT

Message: "Do you offer bulk discounts for orders over 100 units?"
Category: QUESTION

Message: "Your customer service team was incredibly helpful and resolved my issue instantly!"
Category: COMPLIMENT

Message: "I'd like to return this item and get my money back."
Category: REFUND_REQUEST

Now classify these:
Message: "The app keeps crashing whenever I try to checkout."
Category:"""

    result = complete("You are a customer service classifier.", few_shot_prompt)
    print(f"Output: {result}")

    # Test more messages
    messages = [
        "Can you tell me the estimated delivery time for international orders?",
        "This is the worst experience I've ever had with an online shop.",
        "I want a full refund for order #12345.",
    ]

    for msg in messages:
        prompt = few_shot_prompt.replace(
            "Message: \"The app keeps crashing whenever I try to checkout.\"\nCategory:",
            f'Message: "{msg}"\nCategory:'
        )
        result = complete("You are a customer service classifier.", prompt)
        print(f"  \"{msg[:60]}...\" → {result.split()[0]}")


# ── 4. Few-shot format steering ───────────────────────────────────────────────
def demo_few_shot_format():
    print("\n=== 4. Few-Shot Format Steering ===")

    # Few-shot examples that enforce a structured output format
    few_shot_prompt = """Extract person information from text into structured format.

Text: "Dr. Sarah Chen, a 34-year-old cardiologist, works at Boston Medical Center and lives in Cambridge, MA."
Extraction:
  Name: Dr. Sarah Chen
  Age: 34
  Occupation: Cardiologist
  Workplace: Boston Medical Center
  Location: Cambridge, MA

Text: "James O'Brien is a 28-year-old software engineer at Google's London office."
Extraction:
  Name: James O'Brien
  Age: 28
  Occupation: Software Engineer
  Workplace: Google
  Location: London

Text: "Professor Maria Santos, aged 52, teaches economics at the University of São Paulo and recently published a book on trade theory."
Extraction:"""

    result = complete("You are an information extraction assistant.", few_shot_prompt)
    print(f"Output:\n{result}")


# ── 5. Shot count comparison ──────────────────────────────────────────────────
def demo_shot_count_comparison():
    print("\n=== 5. Shot Count Comparison ===")

    test_input = "The battery drains faster than expected after the latest update."

    # Zero-shot
    zero = complete(
        "You are a customer service classifier.",
        f'Classify as COMPLAINT, QUESTION, COMPLIMENT, or REFUND_REQUEST:\n"{test_input}"'
    )

    # One-shot
    one = complete(
        "You are a customer service classifier.",
        f'''Classify as COMPLAINT, QUESTION, COMPLIMENT, or REFUND_REQUEST.

Message: "My order arrived broken."
Category: COMPLAINT

Classify: "{test_input}"
Category:'''
    )

    # Three-shot
    three = complete(
        "You are a customer service classifier.",
        f'''Classify as COMPLAINT, QUESTION, COMPLIMENT, or REFUND_REQUEST.

Message: "My order arrived broken."
Category: COMPLAINT

Message: "What are your store hours?"
Category: QUESTION

Message: "The product is fantastic!"
Category: COMPLIMENT

Classify: "{test_input}"
Category:'''
    )

    print(f"Input: \"{test_input}\"")
    print(f"  Zero-shot: {zero}")
    print(f"  One-shot:  {one}")
    print(f"  Three-shot: {three}")


if __name__ == "__main__":
    demo_zero_shot()
    demo_one_shot()
    demo_few_shot_classification()
    demo_few_shot_format()
    demo_shot_count_comparison()
