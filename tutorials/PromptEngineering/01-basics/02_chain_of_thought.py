"""
Prompt Engineering 01-basics — Chain-of-Thought (CoT)
======================================================
Topics covered:
  1. Standard vs CoT prompting
  2. Zero-shot CoT ("think step by step")
  3. Few-shot CoT with worked examples
  4. Self-consistency: majority vote over multiple reasoning chains
  5. CoT for complex multi-step problems

Run:
  python 02_chain_of_thought.py
"""

import os
from collections import Counter
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


# ── 1. Standard vs CoT ────────────────────────────────────────────────────────
def demo_standard_vs_cot():
    print("\n=== 1. Standard vs Chain-of-Thought ===")

    problem = "A store had 120 apples. They sold 30% on Monday and 25% of the remainder on Tuesday. How many apples remain?"

    # Standard: direct answer
    standard = complete("You are a maths tutor.", f"Answer this: {problem}")

    # CoT: reason then answer
    cot_prompt = f"""Solve this step by step:
{problem}

Show your working:"""
    cot = complete("You are a maths tutor who shows all working.", cot_prompt)

    print(f"Problem: {problem}\n")
    print(f"Standard response:\n{standard}\n")
    print(f"CoT response:\n{cot}")


# ── 2. Zero-shot CoT ──────────────────────────────────────────────────────────
def demo_zero_shot_cot():
    print("\n=== 2. Zero-Shot CoT ('Think step by step') ===")

    problems = [
        "If John has twice as many marbles as Mary, and together they have 90 marbles, how many does John have?",
        "A train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours. What is the total distance?",
        "Is the statement 'All birds can fly, penguins are birds, therefore penguins can fly' logically valid?",
    ]

    for p in problems:
        # Without CoT trigger
        standard = complete("Answer briefly.", p)
        # With CoT trigger
        cot = complete("Answer the question.", f"{p}\n\nLet's think step by step:")

        print(f"\nProblem: {p}")
        print(f"  Standard: {standard}")
        print(f"  Zero-shot CoT: {cot[:200]}...")


# ── 3. Few-shot CoT ───────────────────────────────────────────────────────────
def demo_few_shot_cot():
    print("\n=== 3. Few-Shot CoT (worked examples) ===")

    few_shot_cot = """Solve each maths problem with step-by-step reasoning.

Problem: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many does he have now?
Reasoning: Roger starts with 5 balls. He buys 2 cans × 3 balls = 6 new balls. Total = 5 + 6 = 11.
Answer: 11

Problem: A farmer has 17 sheep. All but 9 run away. How many remain?
Reasoning: "All but 9" means 9 sheep stay. The rest ran away. So 9 sheep remain.
Answer: 9

Problem: Three friends split a bill of $84. They leave a 20% tip. How much does each person pay?
Reasoning: Tip = 20% × $84 = $16.80. Total with tip = $84 + $16.80 = $100.80. Each person pays $100.80 ÷ 3 = $33.60.
Answer: $33.60

Problem: A tank is 40% full. Adding 30 litres makes it 70% full. What is the tank's capacity?
Reasoning:"""

    result = complete("You are a maths tutor.", few_shot_cot)
    print(f"Output:\n{result}")


# ── 4. Self-consistency ───────────────────────────────────────────────────────
def demo_self_consistency():
    print("\n=== 4. Self-Consistency (majority vote) ===")

    problem = "In a class of 30 students, 60% are girls. Of the girls, 1/3 play football. How many girls play football?"

    prompt = f"""Solve step by step:
{problem}

Final answer (just the number):"""

    # Sample multiple reasoning chains at higher temperature
    answers = []
    for i in range(5):
        response = complete("Solve maths problems.", prompt, temperature=0.7)
        # Extract the last number from the response
        import re
        nums = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if nums:
            answers.append(nums[-1])
            print(f"  Sample {i+1}: ...{response[-60:].strip()} → {nums[-1]}")

    if answers:
        vote = Counter(answers).most_common(1)[0]
        print(f"\nSelf-consistency answer: {vote[0]} (voted by {vote[1]}/{len(answers)} chains)")


# ── 5. CoT for logical/ethical reasoning ─────────────────────────────────────
def demo_cot_reasoning():
    print("\n=== 5. CoT for Complex Reasoning ===")

    problems = [
        {
            "type": "logic",
            "problem": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        },
        {
            "type": "ethical",
            "problem": "A self-driving car must choose between swerving (harming 1 pedestrian) or braking (potentially harming its passenger). What should it do and why?",
        },
    ]

    for item in problems:
        print(f"\n[{item['type'].upper()}] {item['problem']}")
        response = complete(
            "You are a careful reasoner. Think through problems methodically before concluding.",
            f"Problem: {item['problem']}\n\nThink through this carefully:",
        )
        print(f"Response:\n{response[:400]}...")


if __name__ == "__main__":
    demo_standard_vs_cot()
    demo_zero_shot_cot()
    demo_few_shot_cot()
    demo_self_consistency()
    demo_cot_reasoning()
