"""
04 - Output Parsers
===================
Output parsers transform raw LLM text into structured Python objects.

Topics covered:
  1. StrOutputParser          — raw string (most common)
  2. JsonOutputParser         — dict / list from JSON
  3. PydanticOutputParser     — validated Pydantic model
  4. CommaSeparatedListOutputParser
  5. .with_structured_output() — the modern recommended approach
  6. Structured extraction from unstructured text
  7. Handling parse errors with OutputFixingParser
"""

from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
)
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. StrOutputParser ───────────────────────────────────────────────────────
def demo_str_parser():
    prompt = ChatPromptTemplate.from_template("Explain {topic} in one paragraph.")
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "transformers in NLP"})
    print("=== 1. StrOutputParser ===")
    print(type(result).__name__, "→", result[:100], "...")


# ── 2. JsonOutputParser ──────────────────────────────────────────────────────
def demo_json_parser():
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You always respond with valid JSON only. No extra text."),
        ("human", (
            "Return a JSON object with fields: name, founded_year, headquarters "
            "for the company: {company}"
        )),
    ])

    chain = prompt | llm | parser
    result = chain.invoke({"company": "OpenAI"})

    print("\n=== 2. JsonOutputParser ===")
    print(type(result).__name__, "→", result)
    print("Founded:", result.get("founded_year"))


# ── 3. CommaSeparatedListOutputParser ────────────────────────────────────────
def demo_list_parser():
    parser = CommaSeparatedListOutputParser()
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(
        "List 5 popular Python libraries for data science.\n{format_instructions}"
    )

    chain = prompt | llm | parser
    result = chain.invoke({"format_instructions": format_instructions})

    print("\n=== 3. CommaSeparatedListOutputParser ===")
    print("Type:", type(result).__name__)
    for i, lib in enumerate(result, 1):
        print(f"  {i}. {lib.strip()}")


# ── 4. PydanticOutputParser ──────────────────────────────────────────────────
class MovieReview(BaseModel):
    title: str = Field(description="Name of the movie")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    genres: List[str] = Field(description="List of genre tags")
    summary: str = Field(description="One-sentence summary of the movie")
    recommended: bool = Field(description="Whether you recommend watching it")


def demo_pydantic_parser():
    parser = PydanticOutputParser(pydantic_object=MovieReview)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a film critic. Respond using the requested format."),
        ("human", "Review the movie '{movie}'.\n\n{format_instructions}"),
    ])

    chain = prompt | llm | parser
    result: MovieReview = chain.invoke({
        "movie": "Inception",
        "format_instructions": parser.get_format_instructions(),
    })

    print("\n=== 4. PydanticOutputParser ===")
    print(f"Title      : {result.title}")
    print(f"Rating     : {result.rating}/10")
    print(f"Genres     : {result.genres}")
    print(f"Summary    : {result.summary}")
    print(f"Recommended: {result.recommended}")


# ── 5. with_structured_output (modern recommended approach) ──────────────────
class JobPosting(BaseModel):
    company: str = Field(description="Company name")
    role: str = Field(description="Job title")
    required_skills: List[str] = Field(description="Required technical skills")
    experience_years: int = Field(description="Minimum years of experience required")
    remote: Optional[bool] = Field(description="True if remote-friendly, None if unknown")


def demo_with_structured_output():
    # No parser needed — the LLM itself returns a structured object
    structured_llm = llm.with_structured_output(JobPosting)

    result: JobPosting = structured_llm.invoke(
        "Parse this job description: "
        "Senior ML Engineer at TechCorp. 5+ years experience. "
        "Skills: Python, TensorFlow, PyTorch, Kubernetes. "
        "Hybrid work from Berlin."
    )

    print("\n=== 5. with_structured_output ===")
    print(f"Company   : {result.company}")
    print(f"Role      : {result.role}")
    print(f"Skills    : {result.required_skills}")
    print(f"Experience: {result.experience_years}+ years")
    print(f"Remote    : {result.remote}")


# ── 6. Structured extraction from unstructured text ──────────────────────────
class PersonInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(description="Age if mentioned, else None")
    occupation: str = Field(description="Their job or role")
    key_achievements: List[str] = Field(description="Notable achievements (up to 3)")


def demo_extraction():
    structured_llm = llm.with_structured_output(PersonInfo)

    bio = """
    Marie Curie (born 1867) was a Polish-French physicist and chemist who 
    conducted pioneering research on radioactivity. She was the first woman 
    to win a Nobel Prize, the only person to win Nobel Prizes in two sciences 
    (Physics 1903, Chemistry 1911), and the first woman to become a professor 
    at the University of Paris.
    """

    result: PersonInfo = structured_llm.invoke(f"Extract information from this bio:\n{bio}")

    print("\n=== 6. Structured Extraction ===")
    print(f"Name        : {result.name}")
    print(f"Age at birth: {result.age}")
    print(f"Occupation  : {result.occupation}")
    print("Achievements:")
    for a in result.key_achievements:
        print(f"  • {a}")


# ── 7. Multiple extractions in one call ──────────────────────────────────────
class ExtractedEntities(BaseModel):
    people: List[str] = Field(description="Names of people mentioned")
    organizations: List[str] = Field(description="Organisation or company names")
    locations: List[str] = Field(description="Cities, countries, or places")
    dates: List[str] = Field(description="Dates or time references")


def demo_ner_extraction():
    structured_llm = llm.with_structured_output(ExtractedEntities)

    text = (
        "On March 15, 2024, Elon Musk met with Angela Merkel in Berlin "
        "to discuss Tesla's new Gigafactory expansion plans. The meeting, "
        "also attended by BMW executives, took place at the German Chancellery."
    )

    result: ExtractedEntities = structured_llm.invoke(
        f"Extract all named entities from this text:\n{text}"
    )

    print("\n=== 7. Named Entity Extraction ===")
    print(f"People       : {result.people}")
    print(f"Organisations: {result.organizations}")
    print(f"Locations    : {result.locations}")
    print(f"Dates        : {result.dates}")


# ── 8. OutputFixingParser — handle malformed output ──────────────────────────
def demo_output_fixing_parser():
    base_parser = PydanticOutputParser(pydantic_object=MovieReview)
    # Wraps the parser: if the LLM output is malformed, asks the LLM to fix it
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    # Simulate malformed JSON (missing quotes, wrong types)
    malformed = """{
        title: Interstellar,
        rating: "nine",
        genres: [sci-fi, drama],
        summary: "A visually stunning space epic.",
        recommended: yes
    }"""

    result: MovieReview = fixing_parser.parse(malformed)
    print("\n=== 8. OutputFixingParser ===")
    print(f"Fixed  → title={result.title}, rating={result.rating}, recommended={result.recommended}")


if __name__ == "__main__":
    demo_str_parser()
    demo_json_parser()
    demo_list_parser()
    demo_pydantic_parser()
    demo_with_structured_output()
    demo_extraction()
    demo_ner_extraction()
    demo_output_fixing_parser()
