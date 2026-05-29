"""
Prompt Engineering 01-basics — Role Prompting & Personas
=========================================================
Topics covered:
  1. System message as role assignment
  2. Expert persona framing
  3. Tone and style control
  4. Audience adaptation
  5. Comparing outputs across personas

Run:
  python 03_role_prompting.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def complete(system: str, user: str) -> str:
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content.strip()


# ── 1. Basic role assignment ──────────────────────────────────────────────────
def demo_basic_roles():
    print("\n=== 1. Basic Role Assignment ===")

    topic = "What causes inflation?"

    roles = {
        "default": "You are a helpful assistant.",
        "economist": "You are a senior economist with 20 years of experience in macroeconomic policy. Explain concepts rigorously with examples from economic history.",
        "teacher": "You are a middle-school economics teacher. Use simple language, relatable analogies, and keep explanations to 3-4 sentences.",
        "journalist": "You are an investigative financial journalist writing for a general audience. Be concise, engaging, and highlight real-world impacts.",
    }

    for role_name, system in roles.items():
        answer = complete(system, topic)
        print(f"\n[{role_name.upper()}]")
        print(f"{answer[:250]}...")


# ── 2. Expert persona for technical tasks ─────────────────────────────────────
def demo_expert_persona():
    print("\n=== 2. Expert Persona for Technical Code Review ===")

    code = '''
def process_users(db_conn, user_ids):
    results = []
    for uid in user_ids:
        query = f"SELECT * FROM users WHERE id = {uid}"
        result = db_conn.execute(query)
        results.append(result.fetchone())
    return results
'''

    # Generic assistant
    generic = complete(
        "You are a helpful assistant.",
        f"Review this code:\n{code}"
    )

    # Expert security engineer
    expert = complete(
        "You are a senior Python security engineer specialising in secure coding practices, SQL injection prevention, and performance optimisation. Be specific about CVEs and OWASP categories where relevant.",
        f"Review this code for security vulnerabilities and performance issues:\n{code}"
    )

    print(f"\n[GENERIC ASSISTANT]\n{generic[:300]}...")
    print(f"\n[SECURITY EXPERT]\n{expert[:400]}...")


# ── 3. Audience adaptation ────────────────────────────────────────────────────
def demo_audience_adaptation():
    print("\n=== 3. Audience Adaptation ===")

    concept = "Explain how neural networks learn using backpropagation."

    audiences = {
        "5-year-old": "Explain as if talking to a 5-year-old child using toys as analogies.",
        "high school student": "Explain to a high school student who knows algebra but not calculus.",
        "ML practitioner": "Explain to an experienced ML practitioner. Use precise technical language.",
        "business executive": "Explain to a non-technical CEO focused on business value and risk.",
    }

    for audience, instruction in audiences.items():
        system = f"You are an AI educator. {instruction}"
        result = complete(system, concept)
        print(f"\n[For: {audience.upper()}]")
        print(f"{result[:200]}...")


# ── 4. Tone and style control ─────────────────────────────────────────────────
def demo_tone_control():
    print("\n=== 4. Tone and Style Control ===")

    message = "Write a short announcement that our product launch is delayed by 2 weeks."

    tones = {
        "formal/corporate":   "Write in a formal, corporate tone suitable for a press release.",
        "empathetic/honest":  "Write in an empathetic, transparent tone acknowledging customer impact.",
        "casual/startup":     "Write in a casual, friendly startup tone as if messaging your community on Discord.",
        "confident/positive": "Write in a confident, positive tone that reframes the delay as quality commitment.",
    }

    for tone, instruction in tones.items():
        result = complete(f"You are a communications writer. {instruction}", message)
        print(f"\n[{tone.upper()}]\n{result}")


# ── 5. Persona consistency across turns ──────────────────────────────────────
def demo_persona_consistency():
    print("\n=== 5. Persona Consistency in a Dialogue ===")

    system = """You are Dr. Alex Morgan, a marine biologist specialising in deep-sea ecosystems.
You are enthusiastic about your field, use scientific terminology naturally (with brief explanations),
and often relate answers to current climate research. You speak in first person about your work."""

    conversation = [
        "What's the most fascinating creature you've studied?",
        "How does deep-sea research help us understand climate change?",
        "What equipment do you use on a typical dive expedition?",
    ]

    messages = [{"role": "system", "content": system}]

    for user_msg in conversation:
        messages.append({"role": "user", "content": user_msg})
        r = client.chat.completions.create(model=MODEL, temperature=0, messages=messages)
        reply = r.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": reply})
        print(f"\nUser: {user_msg}")
        print(f"Dr. Morgan: {reply[:200]}...")


if __name__ == "__main__":
    demo_basic_roles()
    demo_expert_persona()
    demo_audience_adaptation()
    demo_tone_control()
    demo_persona_consistency()
