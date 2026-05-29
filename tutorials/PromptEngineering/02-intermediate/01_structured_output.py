"""
Prompt Engineering 02-intermediate — Structured Output
=======================================================
Topics covered:
  1. JSON mode (response_format)
  2. Pydantic schema enforcement via JSON mode
  3. Output parsing patterns
  4. Extracting structured data from unstructured text
  5. Schema design best practices

Run:
  python 01_structured_output.py
"""

import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def complete_json(system: str, user: str) -> dict:
    """Call the API in JSON mode and parse the response."""
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return json.loads(r.choices[0].message.content)


# ── 1. Basic JSON mode ────────────────────────────────────────────────────────
def demo_json_mode():
    print("\n=== 1. Basic JSON Mode ===")

    texts = [
        "Apple's iPhone 15 Pro costs $999 and was released in September 2023.",
        "The Tesla Model 3 starts at $40,240 and has a range of 358 miles.",
        "Python 3.12 was released on October 2, 2023, featuring improved error messages.",
    ]

    system = """Extract key information and return as JSON with this schema:
{
  "name": "product/item name",
  "price_usd": "price as a number or null",
  "date": "release date as YYYY-MM-DD or null",
  "key_fact": "the most important single fact"
}"""

    for text in texts:
        result = complete_json(system, f"Extract from: {text}")
        print(f"\nInput: {text}")
        print(f"Output: {json.dumps(result, indent=2)}")


# ── 2. Nested schema extraction ───────────────────────────────────────────────
def demo_nested_schema():
    print("\n=== 2. Nested Schema Extraction ===")

    job_posting = """
    We are looking for a Senior Backend Engineer to join our fintech startup in London.
    The role offers £90,000 - £120,000 base salary plus equity.

    Requirements:
    - 5+ years Python experience
    - Expertise in FastAPI or Django
    - PostgreSQL and Redis proficiency
    - Experience with AWS or GCP
    - Nice to have: Kubernetes, Kafka

    The team is 10 engineers, fully remote-friendly with optional office in Canary Wharf.
    """

    system = """Parse this job posting and return JSON matching this schema exactly:
{
  "title": "job title",
  "company_type": "type of company",
  "location": "city",
  "salary": {"min": number_or_null, "max": number_or_null, "currency": "GBP/USD/EUR"},
  "required_skills": ["list", "of", "required", "skills"],
  "nice_to_have": ["list", "of", "optional", "skills"],
  "remote": true_or_false,
  "team_size": number_or_null
}"""

    result = complete_json(system, job_posting)
    print(f"Extracted:\n{json.dumps(result, indent=2)}")


# ── 3. Pydantic schema enforcement ────────────────────────────────────────────
def demo_pydantic_extraction():
    print("\n=== 3. Pydantic Schema Enforcement ===")

    class Address(BaseModel):
        street: Optional[str] = None
        city: str
        country: str
        postcode: Optional[str] = None

    class Person(BaseModel):
        full_name: str
        age: Optional[int] = None
        occupation: str
        employer: Optional[str] = None
        address: Optional[Address] = None
        email: Optional[str] = None

    schema_json = Person.model_json_schema()

    texts = [
        "Dr. Emma Wilson, 41, is a neuroscientist at the Wellcome Trust in London, EC1A 2BE. She can be reached at e.wilson@wellcome.org.",
        "Carlos Mendoza (age 27) works as a data engineer at Spotify's Stockholm office. Lives in Södermalm.",
    ]

    for text in texts:
        result = complete_json(
            f"Extract person information into this JSON schema: {json.dumps(schema_json)}",
            text
        )
        try:
            person = Person(**result)
            print(f"\nInput: {text}")
            print(f"Parsed: {person.model_dump_json(indent=2, exclude_none=True)}")
        except Exception as e:
            print(f"Validation error: {e}")
            print(f"Raw output: {result}")


# ── 4. List extraction ────────────────────────────────────────────────────────
def demo_list_extraction():
    print("\n=== 4. List Extraction ===")

    article = """
    This year's top programming languages include Python, widely used in data science and ML.
    JavaScript continues to dominate web development. Rust is growing rapidly for systems programming.
    TypeScript has become the preferred choice for large-scale Node.js projects.
    Go is popular for microservices due to its simplicity and performance.
    Kotlin is the recommended language for Android development, replacing Java.
    """

    result = complete_json(
        """Extract all programming languages mentioned and return JSON:
{
  "languages": [
    {"name": "...", "primary_use": "...", "trend": "growing/stable/declining"}
  ]
}""",
        article
    )

    print(f"Extracted {len(result.get('languages', []))} languages:")
    for lang in result.get("languages", []):
        print(f"  • {lang['name']} — {lang['primary_use']} [{lang['trend']}]")


# ── 5. Structured sentiment analysis ─────────────────────────────────────────
def demo_structured_sentiment():
    print("\n=== 5. Structured Sentiment Analysis ===")

    reviews = [
        "The camera is absolutely stunning! Best phone camera ever. But the battery dies in 6 hours — very frustrating.",
        "Delivery was fast and packaging was great. However the product itself broke after just 2 days. Customer support never responded.",
        "Excellent build quality. The new UI is intuitive. Price is a bit high but worth it for professionals.",
    ]

    system = """Analyse this review and return JSON:
{
  "overall_sentiment": "positive/negative/mixed",
  "score": -5_to_5,
  "positives": ["list of positive aspects"],
  "negatives": ["list of negative aspects"],
  "main_concern": "the biggest complaint or null",
  "would_recommend": true_or_false
}"""

    for review in reviews:
        result = complete_json(system, review)
        print(f"\nReview: {review[:80]}...")
        print(f"  Sentiment: {result.get('overall_sentiment')} (score: {result.get('score')})")
        print(f"  Positives: {result.get('positives', [])}")
        print(f"  Negatives: {result.get('negatives', [])}")
        print(f"  Would recommend: {result.get('would_recommend')}")


if __name__ == "__main__":
    demo_json_mode()
    demo_nested_schema()
    demo_pydantic_extraction()
    demo_list_extraction()
    demo_structured_sentiment()
