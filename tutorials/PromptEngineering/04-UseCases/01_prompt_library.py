"""
Prompt Engineering 04-UseCases — Reusable Prompt Library
=========================================================
Topics covered:
  1. Summarisation templates (email, article, meeting notes)
  2. Classification templates
  3. Extraction templates
  4. Code-related templates (review, explain, document)
  5. Writing assistance templates
  6. Evaluation/scoring templates

Run:
  python 01_prompt_library.py
"""

import os
import json
from textwrap import dedent
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def complete(system: str, user: str, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs
    )
    return r.choices[0].message.content.strip()


# ── Prompt templates ───────────────────────────────────────────────────────────

class PromptLibrary:
    """Collection of reusable, parametrised prompt templates."""

    # ── Summarisation ─────────────────────────────────────────────────────────
    @staticmethod
    def email_summary(email_text: str, max_bullets: int = 4) -> str:
        system = dedent(f"""
            You summarise emails into action-oriented bullet points.
            Format:
            **From/Context:** [brief context]
            **Key Points:**
            • [point 1]
            ... (up to {max_bullets} bullets)
            **Action Required:** [what the recipient must do, or 'None']
            **Deadline:** [deadline if mentioned, or 'Not specified']
        """).strip()
        return complete(system, f"Summarise this email:\n\n{email_text}")

    @staticmethod
    def article_summary(article: str, audience: str = "general", length: str = "brief") -> str:
        lengths = {"brief": "2-3 sentences", "medium": "1 paragraph", "detailed": "3-4 paragraphs"}
        system = f"""Summarise articles for a {audience} audience.
Length: {lengths.get(length, '2-3 sentences')}.
Focus on the most important findings or arguments. Use plain language."""
        return complete(system, article)

    @staticmethod
    def meeting_notes(transcript: str) -> str:
        system = dedent("""
            Convert meeting transcripts into structured notes:
            **Meeting Summary:** [1-2 sentence overview]
            **Decisions Made:** [list of decisions]
            **Action Items:** [owner: task, deadline]
            **Open Questions:** [unresolved items]
        """).strip()
        return complete(system, transcript)

    # ── Classification ────────────────────────────────────────────────────────
    @staticmethod
    def classify_support_ticket(text: str) -> dict:
        system = """Classify support tickets. Return JSON:
{
  "category": "billing|technical|account|shipping|general",
  "priority": "low|medium|high|critical",
  "sentiment": "positive|neutral|negative|angry",
  "requires_human": true_or_false,
  "suggested_response_template": "billing_faq|escalate|technical_guide|acknowledgement"
}"""
        return json.loads(complete(system, text, json_mode=True))

    @staticmethod
    def classify_content_safety(text: str) -> dict:
        system = """Evaluate content for safety. Return JSON:
{
  "safe": true_or_false,
  "categories_flagged": ["list of: hate_speech|violence|adult|spam|misinformation"],
  "confidence": "high|medium|low",
  "recommendation": "allow|review|block"
}"""
        return json.loads(complete(system, text, json_mode=True))

    # ── Extraction ────────────────────────────────────────────────────────────
    @staticmethod
    def extract_action_items(text: str) -> list:
        system = """Extract all action items from text. Return JSON:
{"action_items": [{"task": "...", "owner": "... or Unknown", "deadline": "... or Not specified"}]}"""
        result = json.loads(complete(system, text, json_mode=True))
        return result.get("action_items", [])

    @staticmethod
    def extract_key_entities(text: str) -> dict:
        system = """Extract named entities. Return JSON:
{
  "people": [],
  "organisations": [],
  "locations": [],
  "dates": [],
  "products": [],
  "amounts": []
}"""
        return json.loads(complete(system, text, json_mode=True))

    # ── Code assistance ───────────────────────────────────────────────────────
    @staticmethod
    def review_code(code: str, language: str = "Python") -> str:
        system = f"""You are an expert {language} code reviewer.
Review for: correctness, security vulnerabilities, performance, readability, best practices.
Structure your review:
**Issues Found:** [list with severity: critical/major/minor]
**Suggestions:** [improvement recommendations]
**Positive Aspects:** [what's done well]
**Overall Rating:** X/10"""
        return complete(system, f"Review this {language} code:\n\n```{language.lower()}\n{code}\n```")

    @staticmethod
    def explain_code(code: str, level: str = "intermediate") -> str:
        system = f"""Explain code to a {level} developer.
Structure: purpose → how it works → key concepts used → potential gotchas."""
        return complete(system, f"Explain this code:\n\n{code}")

    @staticmethod
    def generate_docstring(code: str, style: str = "Google") -> str:
        system = f"Generate a {style}-style docstring for this function. Include Args, Returns, Raises, and an Example."
        return complete(system, code)

    # ── Writing assistance ─────────────────────────────────────────────────────
    @staticmethod
    def improve_writing(text: str, goal: str = "clarity") -> str:
        goals = {
            "clarity":     "Make it clearer and more concise. Cut unnecessary words.",
            "formality":   "Make it more professional and formal in tone.",
            "engagement":  "Make it more engaging, vivid, and compelling.",
            "simplicity":  "Simplify vocabulary and sentence structure for a general audience.",
        }
        system = f"You improve writing. Goal: {goals.get(goal, goal)}. Return the improved version only."
        return complete(system, text)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    @staticmethod
    def evaluate_answer(question: str, answer: str, criteria: list = None) -> dict:
        if criteria is None:
            criteria = ["accuracy", "completeness", "clarity"]
        system = f"""Evaluate this answer to the question. Score each criterion 1-5. Return JSON:
{{
  "criteria": {json.dumps({c: {{"score": 0, "justification": ""}} for c in criteria})},
  "overall_score": 0,
  "strengths": [],
  "improvements": []
}}"""
        user = f"Question: {question}\n\nAnswer: {answer}"
        return json.loads(complete(system, user, json_mode=True))


# ── Demo each template ─────────────────────────────────────────────────────────
def demo_summarisation():
    print("\n=== 1. Summarisation Templates ===")

    email = """From: Sarah Johnson <sarah@partner.com>
Subject: Urgent: Contract review deadline

Hi team,
Following our call last Thursday, I need the signed contract by end of day Friday (14th).
Legal has reviewed and approved all clauses. The main change from v1 is section 4.2 
(liability cap increased to £500k). Please have your CEO sign pages 8 and 15.
Send to contracts@partner.com. Please confirm receipt.
Sarah"""

    print("\n[Email Summary]")
    print(PromptLibrary.email_summary(email))

    meeting = """
Alex: OK let's start. We need to decide on the database migration timeline.
Ben: I think we can do it in 3 weeks if Sarah handles the data validation.
Sarah: I can do that, but I'll need the schema docs by next Monday.
Alex: Agreed. Ben, can you write the migration scripts by the 20th?
Ben: Yes. Should we use blue-green deployment?
Alex: Good point. Let's confirm that in the next meeting. I'll book a slot.
Sarah: One thing — do we need to notify customers about downtime?
Alex: Still unclear. I'll check with legal and get back to you by Thursday.
"""
    print("\n[Meeting Notes]")
    print(PromptLibrary.meeting_notes(meeting))


def demo_classification():
    print("\n=== 2. Classification Templates ===")

    tickets = [
        "My account was charged twice for the same order! This is unacceptable. I want an immediate refund!",
        "Hi, I was wondering if you offer student discounts?",
        "The app crashes every time I try to upload a profile photo on iOS 17.",
    ]

    for ticket in tickets:
        result = PromptLibrary.classify_support_ticket(ticket)
        print(f"\nTicket: {ticket[:70]}...")
        print(f"  Category: {result.get('category')} | Priority: {result.get('priority')} | Human needed: {result.get('requires_human')}")


def demo_extraction():
    print("\n=== 3. Extraction Templates ===")

    text = """
    In our Q3 planning meeting on November 15th, CEO James Park confirmed the £2M budget for the London office expansion.
    Maria Chen (Engineering) will deliver the infrastructure plan by December 1st.
    Tom Brown must complete the vendor evaluations before the board meeting on December 10th.
    Pending: Sarah needs to confirm headcount numbers — deadline is end of November.
    """

    print("\n[Action Items]")
    items = PromptLibrary.extract_action_items(text)
    for item in items:
        print(f"  • {item['owner']}: {item['task']} (by {item['deadline']})")

    print("\n[Key Entities]")
    entities = PromptLibrary.extract_key_entities(text)
    for entity_type, values in entities.items():
        if values:
            print(f"  {entity_type}: {', '.join(str(v) for v in values)}")


def demo_code_templates():
    print("\n=== 4. Code Templates ===")

    code = '''
def get_user_data(user_id):
    import sqlite3
    conn = sqlite3.connect("users.db")
    cursor = conn.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchall()
'''

    print("\n[Code Review]")
    review = PromptLibrary.review_code(code)
    print(review[:400])

    print("\n[Docstring Generation]")
    fn = "def calculate_compound_interest(principal: float, rate: float, n: int, t: int) -> float:\n    return principal * (1 + rate/n) ** (n*t)"
    docstring = PromptLibrary.generate_docstring(fn)
    print(docstring)


def demo_evaluation():
    print("\n=== 5. Evaluation Template ===")

    question = "What is the difference between a list and a tuple in Python?"
    answer = "Lists are mutable and use square brackets. Tuples are immutable and use parentheses. Tuples are slightly faster."

    result = PromptLibrary.evaluate_answer(
        question, answer,
        criteria=["accuracy", "completeness", "clarity"]
    )

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"Overall Score: {result.get('overall_score')}/5")
    for criterion, data in result.get("criteria", {}).items():
        print(f"  {criterion}: {data.get('score')}/5 — {data.get('justification', '')[:80]}")
    print(f"Improvements: {result.get('improvements', [])}")


if __name__ == "__main__":
    demo_summarisation()
    demo_classification()
    demo_extraction()
    demo_code_templates()
    demo_evaluation()
