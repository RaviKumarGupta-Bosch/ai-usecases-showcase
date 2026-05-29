"""
03 - Structured Output (Advanced)
===================================
The modern way to extract typed, validated data from LLM responses.
.with_structured_output() is the recommended approach over manual parsers.

Topics covered:
  1. Basic structured output with Pydantic
  2. Nested Pydantic models (complex schemas)
  3. json_mode — raw dict output (no Pydantic)
  4. Multi-entity extraction in one call
  5. Optional fields and default values
  6. Tool-calling extraction (the underlying mechanism)
  7. Chaining structured output with downstream logic
  8. Structured output with validation and retry
"""

from typing import List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. Basic structured output ───────────────────────────────────────────────
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0–1")
    key_phrases: List[str] = Field(description="Key phrases that drove the sentiment")
    reasoning: str = Field(description="Brief explanation of the classification")


def demo_basic_structured():
    structured_llm = llm.with_structured_output(SentimentAnalysis)

    texts = [
        "The product is fantastic! Best purchase I've made all year. Fast delivery too.",
        "Terrible experience. The item arrived broken and support never responded.",
        "It's okay. Does what it says but nothing special. Average quality for the price.",
    ]

    print("=== 1. Structured Sentiment Analysis ===")
    for text in texts:
        result = structured_llm.invoke(f"Analyse the sentiment of this review:\n'{text}'")
        print(f"\nText      : {text[:60]}...")
        print(f"Sentiment : {result.sentiment} ({result.confidence:.0%})")
        print(f"Phrases   : {result.key_phrases}")
        print(f"Reasoning : {result.reasoning}")


# ── 2. Nested Pydantic models ────────────────────────────────────────────────
class Address(BaseModel):
    street: Optional[str] = None
    city: str
    country: str
    postal_code: Optional[str] = None


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None


class CompanyProfile(BaseModel):
    name: str
    founded_year: Optional[int] = None
    industry: str
    headquarters: Address
    contact: ContactInfo
    employee_count: Optional[str] = Field(None, description="Employee count range, e.g. '10-50'")
    products: List[str] = Field(default_factory=list, description="Main products or services")
    description: str = Field(description="One-sentence company description")


def demo_nested_models():
    structured_llm = llm.with_structured_output(CompanyProfile)

    text = """
    OpenAI is an American AI research company headquartered at 3180 18th Street, San Francisco, 
    California 94110. Founded in 2015 by Sam Altman, Greg Brockman, Ilya Sutskever, and others,
    it employs over 1,500 people. OpenAI's main products include GPT-4, ChatGPT, DALL-E, Sora,
    and the OpenAI API. You can reach them at safety@openai.com or visit openai.com.
    """

    result: CompanyProfile = structured_llm.invoke(
        f"Extract company information from this text:\n{text}"
    )

    print("\n=== 2. Nested Models — Company Profile ===")
    print(f"Company    : {result.name} (founded {result.founded_year})")
    print(f"Industry   : {result.industry}")
    print(f"Location   : {result.headquarters.city}, {result.headquarters.country}")
    print(f"Contact    : {result.contact.email} | {result.contact.website}")
    print(f"Products   : {result.products}")
    print(f"Description: {result.description}")


# ── 3. json_mode — raw dict output ───────────────────────────────────────────
def demo_json_mode():
    """json_mode forces JSON output without a Pydantic schema. Returns a dict."""
    json_llm = llm.with_structured_output(None, method="json_mode")

    result = json_llm.invoke(
        "Return a JSON object with: recipe_name, cuisine, prep_time_minutes (int), "
        "ingredients (list of strings), difficulty (easy|medium|hard) "
        "for a classic Italian pasta dish."
    )

    print("\n=== 3. json_mode (dict output) ===")
    print(type(result).__name__, "→", result)
    print(f"Recipe: {result.get('recipe_name')}, Difficulty: {result.get('difficulty')}")


# ── 4. Multi-entity extraction ───────────────────────────────────────────────
class Skill(BaseModel):
    name: str
    level: Literal["beginner", "intermediate", "advanced", "expert"]
    years_experience: Optional[int] = None


class WorkExperience(BaseModel):
    company: str
    role: str
    start_year: int
    end_year: Optional[int] = Field(None, description="None if current job")
    responsibilities: List[str] = Field(max_length=3)


class ResumeExtraction(BaseModel):
    candidate_name: str
    total_years_experience: int
    skills: List[Skill]
    work_history: List[WorkExperience]
    education_level: Literal["high_school", "bachelor", "master", "phd", "other"]
    is_remote_friendly: Optional[bool] = None


def demo_multi_entity_extraction():
    structured_llm = llm.with_structured_output(ResumeExtraction)

    resume = """
    Jane Smith
    Senior Software Engineer | 8 years experience

    Skills: Python (expert, 8 years), JavaScript (advanced, 5 years), 
    Kubernetes (intermediate, 3 years), Rust (beginner, 1 year)

    Work History:
    - TechCorp Inc. (2019–present): Senior Engineer. Built ML pipeline, led backend team, 
      reduced latency by 40%.
    - StartupXYZ (2016–2019): Full-stack Developer. Developed React frontend, 
      Python APIs, deployed on AWS.

    Education: MSc Computer Science, MIT
    Open to remote work: Yes
    """

    result: ResumeExtraction = structured_llm.invoke(
        f"Extract structured information from this resume:\n{resume}"
    )

    print("\n=== 4. Multi-Entity Resume Extraction ===")
    print(f"Candidate : {result.candidate_name} ({result.total_years_experience} years exp)")
    print(f"Education : {result.education_level}")
    print(f"Remote    : {result.is_remote_friendly}")
    print("Skills:")
    for skill in result.skills:
        print(f"  • {skill.name} — {skill.level} ({skill.years_experience}y)")
    print("Work History:")
    for job in result.work_history:
        end = job.end_year or "present"
        print(f"  • {job.company} ({job.start_year}–{end}): {job.role}")


# ── 5. Structured output in an LCEL chain ────────────────────────────────────
class EmailClassification(BaseModel):
    category: Literal["sales", "support", "spam", "hr", "billing", "other"]
    priority: Literal["urgent", "high", "normal", "low"]
    suggested_action: str
    auto_reply_suitable: bool
    tags: List[str] = Field(max_length=5)


def demo_structured_in_chain():
    """Combine a prompt + structured output + downstream routing logic."""
    classify_llm = llm.with_structured_output(EmailClassification)

    prompt = ChatPromptTemplate.from_template(
        "You are an email triage system. Classify this email and suggest an action.\n\n"
        "Email:\nSubject: {subject}\nBody: {body}"
    )

    classify_chain = prompt | classify_llm

    emails = [
        {
            "subject": "URGENT: Production database is down!",
            "body": "Our main DB stopped responding 10 minutes ago. All services are failing. Need immediate help.",
        },
        {
            "subject": "Congratulations! You've won $1,000,000!",
            "body": "Click here to claim your prize. Limited time offer. Send your bank details.",
        },
        {
            "subject": "Interested in upgrading to Enterprise plan",
            "body": "Hi, we're a team of 200 and considering moving to your Enterprise tier. Can we schedule a demo?",
        },
    ]

    print("\n=== 5. Structured Output in LCEL Chain ===")
    for email in emails:
        result: EmailClassification = classify_chain.invoke(email)
        action = "🚨 AUTO-ESCALATE" if result.priority == "urgent" else "📋 QUEUE"
        print(f"\n[{action}] {email['subject'][:50]}")
        print(f"  Category : {result.category} | Priority: {result.priority}")
        print(f"  Action   : {result.suggested_action}")
        print(f"  Auto-reply: {result.auto_reply_suitable} | Tags: {result.tags}")


# ── 6. include_raw — get both raw and parsed output ──────────────────────────
def demo_include_raw():
    """
    include_raw=True returns both the parsed model and the raw LLM message.
    Useful for debugging or when you need the raw text alongside the structured data.
    """
    structured_llm = llm.with_structured_output(SentimentAnalysis, include_raw=True)

    result = structured_llm.invoke("Analyse: 'Absolutely love this product! 10/10 would buy again.'")

    print("\n=== 6. include_raw ===")
    print("Raw message type:", type(result["raw"]).__name__)
    print("Parsed type     :", type(result["parsed"]).__name__)
    if result["parsed"]:
        print("Sentiment       :", result["parsed"].sentiment)
    print("Parse error     :", result.get("parsing_error"))


if __name__ == "__main__":
    demo_basic_structured()
    demo_nested_models()
    demo_json_mode()
    demo_multi_entity_extraction()
    demo_structured_in_chain()
    demo_include_raw()
