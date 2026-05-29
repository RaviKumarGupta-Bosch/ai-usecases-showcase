"""
Ollama 02-Intermediate — Structured Output
============================================
Topics covered:
  1. format="json" in ollama.generate() — forces valid JSON
  2. format="json" in ollama.chat()
  3. Providing a Pydantic schema as format specification
  4. Comparing output with and without format parameter
  5. Parsing and validating the returned JSON

Prerequisites:
  - Ollama running: `ollama serve`
  - Model pulled: `ollama pull llama3.2`

Run:
  python 02_structured_output.py
"""

import os
import sys
import json
from typing import Optional
from dotenv import load_dotenv
import ollama
from pydantic import BaseModel

load_dotenv()

MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class BookReview(BaseModel):
    title: str
    author: str
    genre: str
    rating: int          # 1-5
    summary: str
    recommended: bool


class TechJobPosting(BaseModel):
    job_title: str
    company: str
    location: str
    seniority: str       # junior / mid / senior / lead
    skills_required: list[str]
    salary_range: Optional[str] = None
    remote: bool


class SentimentResult(BaseModel):
    text: str
    sentiment: str       # positive / neutral / negative
    confidence: float    # 0.0 – 1.0
    key_phrases: list[str]


# ── 1. Without format parameter (may hallucinate non-JSON) ────────────────────
def demo_without_format():
    print("\n=== 1. Without format='json' ===")
    prompt = (
        "Extract the following info from this book description as JSON: "
        "title, author, genre, rating (1-5), summary (one sentence).\n\n"
        "Book: 'The Pragmatic Programmer' by Andrew Hunt and David Thomas. "
        "A classic software engineering book. Essential reading. Rating: 5/5."
    )
    response = ollama.generate(model=MODEL, prompt=prompt)
    raw = response.response if hasattr(response, "response") else response.get("response", "")
    print(f"  Raw output:\n{raw[:400]}")
    print("\n  NOTE: May include markdown fences, preamble, or trailing text.")


# ── 2. With format="json" (generate) ─────────────────────────────────────────
def demo_generate_json():
    print("\n=== 2. With format='json' — ollama.generate() ===")
    prompt = (
        "Extract the following as JSON with keys: title, author, genre, rating (integer 1-5), "
        "summary (one sentence), recommended (boolean).\n\n"
        "Book: 'Clean Code' by Robert C. Martin. A guide to writing readable, maintainable code. "
        "Highly recommended for professional developers. Rating: 5."
    )
    response = ollama.generate(model=MODEL, prompt=prompt, format="json")
    raw = response.response if hasattr(response, "response") else response.get("response", "")

    try:
        data = json.loads(raw)
        print("  Parsed JSON:")
        for k, v in data.items():
            print(f"    {k}: {v}")
    except json.JSONDecodeError as e:
        print(f"  Parse error: {e}\n  Raw: {raw[:200]}")


# ── 3. With format="json" (chat) ─────────────────────────────────────────────
def demo_chat_json():
    print("\n=== 3. With format='json' — ollama.chat() ===")
    messages = [
        {
            "role": "system",
            "content": (
                "You extract structured data and return ONLY valid JSON. "
                "Use these exact keys: job_title, company, location, seniority, "
                "skills_required (array), salary_range (string or null), remote (boolean)."
            ),
        },
        {
            "role": "user",
            "content": (
                "Extract from this posting:\n"
                "Senior Python Developer at DataCo, Berlin (remote ok). "
                "Requirements: Python 5+ yrs, FastAPI, PostgreSQL, Docker, Kubernetes. "
                "Salary: €75,000 – €95,000."
            ),
        },
    ]
    response = ollama.chat(model=MODEL, messages=messages, format="json")
    raw = response.message.content if hasattr(response, "message") else response["message"]["content"]

    try:
        data = json.loads(raw)
        validated = TechJobPosting(**data)
        print("  Validated TechJobPosting:")
        print(f"    title:    {validated.job_title}")
        print(f"    company:  {validated.company}")
        print(f"    location: {validated.location}")
        print(f"    remote:   {validated.remote}")
        print(f"    skills:   {', '.join(validated.skills_required)}")
        print(f"    salary:   {validated.salary_range}")
    except Exception as e:
        print(f"  Validation error: {e}\n  Raw: {raw[:300]}")


# ── 4. Pydantic schema as format ──────────────────────────────────────────────
def demo_pydantic_format():
    print("\n=== 4. Pydantic Schema as format= Parameter ===")
    # Newer Ollama versions accept a Pydantic model directly
    reviews_to_extract = [
        "'Dune' by Frank Herbert — epic sci-fi spanning politics, religion, ecology. Rating: 5. Highly recommended.",
        "'The Hobbit' by J.R.R. Tolkien — a charming adventure story, great for all ages. Rating: 4. Recommended.",
    ]

    for text in reviews_to_extract:
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Extract book review data as structured JSON."},
                    {"role": "user",   "content": f"Extract from: {text}"},
                ],
                format=BookReview.model_json_schema(),
            )
            raw = response.message.content if hasattr(response, "message") else response["message"]["content"]
            review = BookReview.model_validate_json(raw)
            print(f"\n  [{review.title}] by {review.author}")
            print(f"    Genre: {review.genre} | Rating: {review.rating}/5 | Recommended: {review.recommended}")
            print(f"    {review.summary}")
        except Exception as e:
            print(f"  Error: {e}")


# ── 5. Batch sentiment analysis ───────────────────────────────────────────────
def demo_batch_sentiment():
    print("\n=== 5. Batch Structured Sentiment Analysis ===")
    texts = [
        "The product exceeded all my expectations. Absolutely love it!",
        "Delivery was okay, nothing special about the item.",
        "Terrible experience. Broke after one day and support was useless.",
        "Good value for money, minor packaging issues but overall fine.",
    ]

    print(f"  Analysing {len(texts)} texts...\n")
    for text in texts:
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You analyse customer review sentiment. "
                            "Return JSON with keys: text (exact input), sentiment "
                            "(positive/neutral/negative), confidence (0.0-1.0), "
                            "key_phrases (array of 2-3 phrases)."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                format="json",
            )
            raw = response.message.content if hasattr(response, "message") else response["message"]["content"]
            data = json.loads(raw)
            sentiment = data.get("sentiment", "?")
            confidence = data.get("confidence", 0)
            phrases    = data.get("key_phrases", [])
            bar = "█" * round(confidence * 10)
            print(f"  [{sentiment.upper():<8}] {bar:<10} {confidence:.0%}  |  {text[:55]}")
            if phrases:
                print(f"             Key phrases: {', '.join(str(p) for p in phrases[:3])}")
        except Exception as e:
            print(f"  Error on '{text[:40]}': {e}")


if __name__ == "__main__":
    print("Ollama 02-Intermediate — Structured Output")
    print("=" * 45)

    try:
        import ollama as _test
        _test.list()
    except Exception:
        print("  ERROR: Cannot connect to Ollama. Start with: ollama serve")
        sys.exit(1)

    demo_without_format()
    demo_generate_json()
    demo_chat_json()
    demo_pydantic_format()
    demo_batch_sentiment()
