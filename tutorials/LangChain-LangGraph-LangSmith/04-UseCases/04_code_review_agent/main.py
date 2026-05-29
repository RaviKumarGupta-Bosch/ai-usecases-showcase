"""
Use Case 04 — Code Review Agent
==================================
An automated code review system that performs parallel multi-dimensional analysis:
- Security review  : OWASP Top 10, injection vulnerabilities, auth issues
- Performance review: algorithmic complexity, memory usage, anti-patterns
- Style review     : PEP 8, naming conventions, documentation, readability

Architecture (LangGraph with parallel Send API):
  route_review → [security_check ‖ performance_check ‖ style_check] → synthesise → END

Output: Structured CodeReviewReport with severity scoring and actionable suggestions

Run:
  python main.py
"""

import os
import operator
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.types import Send

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Pydantic models ────────────────────────────────────────────────────────────
class CodeIssue(BaseModel):
    """A single code issue identified during review."""
    category:    str = Field(description="Issue category: security | performance | style")
    severity:    Literal["critical", "high", "medium", "low", "info"] = Field(
        description="Issue severity level"
    )
    description: str = Field(description="Clear description of the issue")
    line_hint:   str = Field(description="Code reference or line hint (e.g., 'line 12: cursor.execute(query)')")
    suggestion:  str = Field(description="Specific actionable fix recommendation")


class ReviewResult(BaseModel):
    """Result from a single review dimension."""
    category:    Literal["security", "performance", "style"]
    issues:      list[CodeIssue] = Field(description="Issues found in this dimension")
    score:       float = Field(description="Score 0-10 (10=perfect)", ge=0, le=10)
    summary:     str   = Field(description="One-paragraph summary of findings")


class CodeReviewReport(BaseModel):
    """Complete code review report across all dimensions."""
    overall_score:   float       = Field(description="Weighted average score 0-10", ge=0, le=10)
    overall_verdict: Literal["approve", "approve_with_comments", "request_changes", "reject"]
    all_issues:      list[CodeIssue] = Field(description="All issues sorted by severity")
    security_score:  float       = Field(ge=0, le=10)
    performance_score: float     = Field(ge=0, le=10)
    style_score:     float       = Field(ge=0, le=10)
    executive_summary: str       = Field(description="2-3 sentence overall assessment")
    top_priorities:  list[str]   = Field(description="3-5 most important fixes needed")


# ── Graph state ────────────────────────────────────────────────────────────────
class ReviewState(TypedDict):
    code:            str
    language:        str
    review_results:  Annotated[list[ReviewResult], operator.add]  # parallel accumulator
    report:          CodeReviewReport


class ReviewTaskState(TypedDict):
    code:     str
    language: str
    category: Literal["security", "performance", "style"]


# ── Review prompts ─────────────────────────────────────────────────────────────
REVIEW_PROMPTS = {
    "security": """You are a senior application security engineer performing a security code review.

Analyse the following {language} code for security vulnerabilities:
- SQL/Command/LDAP/XSS injection vulnerabilities
- Hardcoded secrets, credentials, or API keys
- Insecure authentication or authorisation
- Sensitive data exposure (logging, error messages)
- Insecure cryptography or hashing (MD5, SHA1 for passwords)
- Missing input validation or sanitisation
- Path traversal vulnerabilities
- Insecure deserialization
- Missing rate limiting or DoS protection

Code to review:
```{language}
{code}
```

Rate severity accurately: critical (immediate exploit risk), high (serious), medium (moderate), low (minor), info (best practice).""",

    "performance": """You are a performance engineering expert performing a performance code review.

Analyse the following {language} code for performance issues:
- Algorithm complexity (O(n²) or worse where O(n) is possible)
- Database: N+1 queries, missing indexes, SELECT *, unoptimised queries
- Memory: memory leaks, large objects in memory, missing cleanup
- I/O: blocking calls, missing async/await, unnecessary network calls
- Caching: missing caches for expensive operations
- Resource management: unclosed connections/files/handles
- String concatenation in loops (use join/StringBuilder)
- Redundant computations that should be memoised

Code to review:
```{language}
{code}
```

Focus on measurable performance impacts, not micro-optimisations.""",

    "style": """You are a senior developer performing a code style and maintainability review.

Analyse the following {language} code for style and quality issues:
- Naming conventions (variables, functions, classes)
- Code documentation (missing docstrings, unclear comments, outdated comments)
- Function length and single responsibility principle
- Magic numbers and strings (should be named constants)
- Error handling completeness and specificity
- Code duplication (DRY principle)
- Readability (complex logic that needs simplification)
- Type annotations/hints usage
- Dead code or unused variables/imports

Code to review:
```{language}
{code}
```

Focus on maintainability and readability for a team environment.""",
}


# ── Nodes ─────────────────────────────────────────────────────────────────────
def route_to_reviewers(state: ReviewState) -> list[Send]:
    """Fan-out: send the code to all three reviewers in parallel."""
    categories = ["security", "performance", "style"]
    return [
        Send(
            "run_review",
            {
                "code":     state["code"],
                "language": state["language"],
                "category": cat,
            },
        )
        for cat in categories
    ]


def run_review_node(state: ReviewTaskState) -> dict:
    """Execute a single dimension review."""
    category = state["category"]
    print(f"  [Reviewer] Running {category} review...")

    review_llm = llm.with_structured_output(ReviewResult)

    prompt = REVIEW_PROMPTS[category].format(
        language=state["language"],
        code=state["code"],
    )

    result = review_llm.invoke(prompt)

    # Ensure category is set correctly
    result = ReviewResult(
        category=category,
        issues=result.issues,
        score=result.score,
        summary=result.summary,
    )

    return {"review_results": [result]}


def synthesise_report_node(state: ReviewState) -> dict:
    """Synthesise all review dimensions into a final CodeReviewReport."""
    print("  [Synthesiser] Generating final report...")

    results = state["review_results"]

    # Sort results to ensure consistent ordering
    result_by_cat = {r.category: r for r in results}

    # Collect all issues and sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    all_issues = []
    for r in results:
        all_issues.extend(r.issues)
    all_issues.sort(key=lambda i: severity_order.get(i.severity, 99))

    # Compute scores
    sec_score  = result_by_cat.get("security",    ReviewResult(category="security",    issues=[], score=10.0, summary="")).score
    perf_score = result_by_cat.get("performance", ReviewResult(category="performance", issues=[], score=10.0, summary="")).score
    style_score = result_by_cat.get("style",      ReviewResult(category="style",       issues=[], score=10.0, summary="")).score

    # Weighted: security 40%, performance 35%, style 25%
    overall = round(sec_score * 0.40 + perf_score * 0.35 + style_score * 0.25, 1)

    # Determine verdict
    critical_count = sum(1 for i in all_issues if i.severity == "critical")
    high_count     = sum(1 for i in all_issues if i.severity == "high")

    if critical_count > 0:
        verdict = "reject"
    elif high_count > 2 or overall < 5:
        verdict = "request_changes"
    elif high_count > 0 or overall < 7.5:
        verdict = "approve_with_comments"
    else:
        verdict = "approve"

    # Generate executive summary
    summary_prompt = f"""Write a 2-3 sentence executive summary for a code review.

Security score: {sec_score}/10 — {result_by_cat.get('security', ReviewResult(category='security', issues=[], score=10, summary='')).summary[:100]}
Performance score: {perf_score}/10 — {result_by_cat.get('performance', ReviewResult(category='performance', issues=[], score=10, summary='')).summary[:100]}
Style score: {style_score}/10 — {result_by_cat.get('style', ReviewResult(category='style', issues=[], score=10, summary='')).summary[:100]}
Total issues: {len(all_issues)} ({critical_count} critical, {high_count} high)
Verdict: {verdict}

Be concise and professional."""

    summary_response = llm.invoke(summary_prompt)

    # Extract top 3-5 priorities
    top_issues = [
        f"[{i.severity.upper()}] {i.category}: {i.description[:80]}"
        for i in all_issues[:5]
    ]

    report = CodeReviewReport(
        overall_score=overall,
        overall_verdict=verdict,
        all_issues=all_issues,
        security_score=sec_score,
        performance_score=perf_score,
        style_score=style_score,
        executive_summary=summary_response.content,
        top_priorities=top_issues or ["No critical issues found"],
    )

    return {"report": report}


# ── Build the graph ────────────────────────────────────────────────────────────
def build_code_review_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("run_review", run_review_node)
    graph.add_node("synthesise", synthesise_report_node)

    graph.set_conditional_entry_point(route_to_reviewers)
    graph.add_edge("run_review", "synthesise")
    graph.add_edge("synthesise", END)

    return graph.compile()


@traceable(name="code_review_agent", tags=["code-quality", "production"])
def review_code(code: str, language: str = "python") -> CodeReviewReport:
    """Run the parallel code review pipeline."""
    app = build_code_review_graph()
    result = app.invoke({
        "code":           code,
        "language":       language,
        "review_results": [],
        "report":         None,
    })
    return result["report"]


def print_report(report: CodeReviewReport, code_label: str):
    """Pretty-print a code review report."""
    verdict_icons = {
        "approve": "✅",
        "approve_with_comments": "⚠️",
        "request_changes": "🔄",
        "reject": "❌",
    }
    icon = verdict_icons.get(report.overall_verdict, "?")

    print(f"\n{'='*65}")
    print(f"Code Review Report: {code_label}")
    print(f"{'='*65}")
    print(f"Verdict : {icon} {report.overall_verdict.upper().replace('_', ' ')}")
    print(f"Overall : {report.overall_score:.1f}/10")
    print(f"  Security   : {report.security_score:.1f}/10")
    print(f"  Performance: {report.performance_score:.1f}/10")
    print(f"  Style      : {report.style_score:.1f}/10")

    print(f"\nExecutive Summary:")
    print(f"  {report.executive_summary}")

    print(f"\nTop Priorities ({len(report.top_priorities)}):")
    for p in report.top_priorities:
        print(f"  • {p}")

    if report.all_issues:
        print(f"\nAll Issues ({len(report.all_issues)} total):")
        for issue in report.all_issues:
            severity_colors = {
                "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"
            }
            icon2 = severity_colors.get(issue.severity, "•")
            print(f"\n  {icon2} [{issue.severity.upper()}] {issue.category}")
            print(f"    Issue : {issue.description}")
            print(f"    Where : {issue.line_hint}")
            print(f"    Fix   : {issue.suggestion}")


# ── Demo code samples ─────────────────────────────────────────────────────────
CODE_SAMPLES = {
    "vulnerable_db_code": {
        "label": "Vulnerable Database Handler",
        "language": "python",
        "code": '''
import sqlite3
import hashlib
import pickle

SECRET_KEY = "my_super_secret_key_123"
DB_PASS = "admin123"

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Build query with user input directly
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    cursor.execute(query)
    user = cursor.fetchone()
    
    return user is not None

def get_user_data(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = " + str(user_id))
    results = cursor.fetchall()
    
    all_data = []
    for row in results:
        all_data.append(row)
    return all_data

def store_password(password):
    # Hash password for storage
    hashed = hashlib.md5(password.encode()).hexdigest()
    return hashed

def load_user_session(session_data):
    # Load user session from cookie
    user = pickle.loads(session_data)
    return user

def process_report(user_input):
    import os
    # Generate report file
    os.system("generate_report.sh " + user_input)
    
    result = []
    for i in range(len(user_input)):
        for j in range(len(user_input)):
            result.append(user_input[i] + user_input[j])
    return result
'''
    },

    "performance_issues": {
        "label": "Performance Anti-Patterns",
        "language": "python",
        "code": '''
import time
import requests

def get_all_users_with_orders(db):
    """Get all users and their order counts."""
    users = db.query("SELECT * FROM users")
    
    result = []
    for user in users:
        # N+1 query problem
        orders = db.query(f"SELECT * FROM orders WHERE user_id = {user['id']}")
        user["order_count"] = len(orders)
        result.append(user)
    
    return result

def find_duplicates(items):
    """Find duplicate items in a list."""
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):  # O(n^2)
            if i != j and items[i] == items[j]:
                if items[i] not in duplicates:
                    duplicates.append(items[i])
    return duplicates

def build_report(records):
    """Build HTML report string."""
    report = ""
    for record in records:
        report = report + "<tr>"
        report = report + "<td>" + str(record["name"]) + "</td>"
        report = report + "<td>" + str(record["value"]) + "</td>"
        report = report + "</tr>"
    return report

def fetch_prices(product_ids):
    """Fetch price for each product."""
    prices = {}
    for product_id in product_ids:
        response = requests.get(f"https://api.example.com/price/{product_id}")
        prices[product_id] = response.json()["price"]
        time.sleep(0.1)
    return prices

def compute_fibonacci(n):
    if n <= 1:
        return n
    return compute_fibonacci(n-1) + compute_fibonacci(n-2)  # Exponential recursion
'''
    },

    "well_written_code": {
        "label": "Well-Written Code (Should Score High)",
        "language": "python",
        "code": '''
"""
User authentication service with secure practices.
"""
import hashlib
import hmac
import secrets
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AuthResult:
    """Result of an authentication attempt."""
    success: bool
    user_id: Optional[int] = None
    error_message: Optional[str] = None


class UserAuthService:
    """Secure user authentication service using parameterised queries and bcrypt."""

    BCRYPT_ROUNDS = 12

    def __init__(self, db_connection, secret_key: str):
        self._db = db_connection
        self._secret_key = secret_key

    def authenticate(self, username: str, password: str) -> AuthResult:
        """
        Authenticate a user by username and password.
        
        Args:
            username: The username to authenticate.
            password: The plaintext password to verify.
            
        Returns:
            AuthResult with success status and user ID if successful.
        """
        if not username or not password:
            return AuthResult(success=False, error_message="Credentials required")

        try:
            # Parameterised query prevents SQL injection
            user = self._db.execute(
                "SELECT id, password_hash FROM users WHERE username = ? AND is_active = 1",
                (username,)
            ).fetchone()

            if user is None:
                # Constant-time response to prevent user enumeration
                self._verify_dummy_password(password)
                return AuthResult(success=False, error_message="Invalid credentials")

            if not self._verify_password(password, user["password_hash"]):
                logger.warning("Failed login attempt for user: %s", username)
                return AuthResult(success=False, error_message="Invalid credentials")

            logger.info("Successful authentication for user_id: %d", user["id"])
            return AuthResult(success=True, user_id=user["id"])

        except Exception:
            logger.exception("Authentication error (details hidden from caller)")
            return AuthResult(success=False, error_message="Authentication service error")

    def hash_password(self, password: str) -> str:
        """Hash a password using PBKDF2-HMAC-SHA256 with a random salt."""
        salt = secrets.token_hex(32)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
        return f"{salt}:{hashed.hex()}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash using constant-time comparison."""
        salt, expected_hash = stored_hash.split(":", 1)
        actual_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
        return hmac.compare_digest(actual_hash.hex(), expected_hash)

    def _verify_dummy_password(self, password: str) -> None:
        """Run password hashing to prevent timing attacks even when user not found."""
        dummy_hash = "dummy_salt:" + "0" * 64
        self._verify_password(password, dummy_hash)
'''
    },
}


def run_demo():
    print("=" * 65)
    print("Code Review Agent — Parallel Multi-Dimensional Analysis")
    print("=" * 65)

    for key, sample in CODE_SAMPLES.items():
        print(f"\nReviewing: {sample['label']} ({sample['language'].upper()})")
        print("Running parallel reviews: security | performance | style...")

        report = review_code(sample["code"], sample["language"])
        print_report(report, sample["label"])


def demo_single_review():
    """Review a single code snippet."""
    print("\n" + "=" * 65)
    print("Code Review Agent — Quick Single Review")
    print("=" * 65)

    code = """
def login(user, pwd):
    q = "select * from users where user='" + user + "' and pwd='" + pwd + "'"
    r = db.execute(q)
    if r: return True
    return False
"""
    report = review_code(code, "python")
    print_report(report, "Quick Review: Inline String Query")


if __name__ == "__main__":
    run_demo()
