"""
LlamaIndex 04-UseCases — Multi-Document Q&A
============================================
Topics covered:
  1. Building indexes over distinct document collections
  2. Cross-collection question routing
  3. SubQuestionQueryEngine for multi-hop reasoning
  4. Source attribution and citation
  5. Combined answer synthesis

Run:
  python 01_multi_document_qa.py
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# ── Corpus A: Software Engineering Practices ──────────────────────────────────
SE_DOCS = [
    Document(text="SOLID principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion. These make code more maintainable and testable.", metadata={"source": "SE Guide", "category": "design"}),
    Document(text="Test-Driven Development (TDD): write failing tests first, then write code to pass them, then refactor. Red-Green-Refactor cycle. Improves design and confidence.", metadata={"source": "SE Guide", "category": "testing"}),
    Document(text="Code review best practices: focus on correctness, clarity, maintainability. Use pull requests. Review small chunks. Give constructive feedback, not personal criticism.", metadata={"source": "SE Guide", "category": "process"}),
    Document(text="Continuous Integration: automatically build and test on every commit. CI tools: GitHub Actions, Jenkins, CircleCI. Catch integration bugs early before merge.", metadata={"source": "SE Guide", "category": "devops"}),
    Document(text="Refactoring: improving code structure without changing behaviour. Extract method, rename variable, replace conditional with polymorphism. Use IDE tools to automate.", metadata={"source": "SE Guide", "category": "design"}),
    Document(text="Clean Code principles: meaningful names, small functions, DRY (Don't Repeat Yourself), YAGNI (You Aren't Gonna Need It), minimal comments — let code speak.", metadata={"source": "SE Guide", "category": "design"}),
]

# ── Corpus B: Cloud Architecture ──────────────────────────────────────────────
CLOUD_DOCS = [
    Document(text="Microservices decompose applications into small, independently deployable services. Each service owns its data. Communicate via REST APIs or message queues.", metadata={"source": "Cloud Guide", "category": "architecture"}),
    Document(text="Kubernetes orchestrates containerised workloads. Pods run containers. Deployments manage replicas. Services expose pods. Ingress routes external traffic.", metadata={"source": "Cloud Guide", "category": "orchestration"}),
    Document(text="Serverless computing: run functions without managing servers. AWS Lambda, Azure Functions, Google Cloud Run. Scales automatically, billed per execution.", metadata={"source": "Cloud Guide", "category": "compute"}),
    Document(text="Event-driven architecture: services communicate via events. Apache Kafka, RabbitMQ, AWS SQS for message brokers. Decouples producers from consumers.", metadata={"source": "Cloud Guide", "category": "architecture"}),
    Document(text="The 12-Factor App methodology: codebase in version control, dependencies explicitly declared, config in environment, stateless processes, disposable containers.", metadata={"source": "Cloud Guide", "category": "best-practices"}),
    Document(text="Infrastructure as Code (IaC): define infrastructure in code files. Terraform for multi-cloud, CloudFormation for AWS, Bicep for Azure. Version-controlled, repeatable.", metadata={"source": "Cloud Guide", "category": "devops"}),
]

# ── Corpus C: AI/ML Engineering ───────────────────────────────────────────────
AIML_DOCS = [
    Document(text="MLOps extends DevOps to ML: automate training, evaluation, and deployment pipelines. MLflow tracks experiments. DVC versions datasets. Feature stores share features across teams.", metadata={"source": "ML Guide", "category": "mlops"}),
    Document(text="Model versioning: tag models with version numbers. A/B testing new versions. Canary deployments gradually roll out. Shadow mode tests in production without serving responses.", metadata={"source": "ML Guide", "category": "deployment"}),
    Document(text="Data pipelines for ML: Airflow, Prefect, or Dagster orchestrate steps. Raw data → cleaning → feature engineering → training → evaluation → deployment.", metadata={"source": "ML Guide", "category": "data"}),
    Document(text="Model monitoring: drift detection (data drift vs concept drift). Statistical tests: KS test, PSI. Alerting on performance degradation. Retraining triggers.", metadata={"source": "ML Guide", "category": "monitoring"}),
    Document(text="LLMOps: managing large language model deployments. Prompt versioning, evaluation frameworks, latency optimisation, cost management, guardrails for safety.", metadata={"source": "ML Guide", "category": "llmops"}),
]


def build_knowledge_base():
    """Build separate indexes for each document collection."""
    print("Building domain indexes...")

    se_index = VectorStoreIndex.from_documents(SE_DOCS)
    cloud_index = VectorStoreIndex.from_documents(CLOUD_DOCS)
    aiml_index = VectorStoreIndex.from_documents(AIML_DOCS)

    tools = [
        QueryEngineTool(
            query_engine=se_index.as_query_engine(similarity_top_k=3),
            metadata=ToolMetadata(
                name="software_engineering",
                description="Covers software engineering: SOLID principles, TDD, clean code, code review, CI, refactoring.",
            ),
        ),
        QueryEngineTool(
            query_engine=cloud_index.as_query_engine(similarity_top_k=3),
            metadata=ToolMetadata(
                name="cloud_architecture",
                description="Covers cloud architecture: microservices, Kubernetes, serverless, event-driven design, IaC.",
            ),
        ),
        QueryEngineTool(
            query_engine=aiml_index.as_query_engine(similarity_top_k=3),
            metadata=ToolMetadata(
                name="ml_engineering",
                description="Covers ML engineering: MLOps, model deployment, monitoring, LLMOps, data pipelines.",
            ),
        ),
    ]

    return tools


# ── 1. Direct single-domain queries ───────────────────────────────────────────
def demo_single_domain(tools):
    print("\n=== 1. Single Domain Queries ===")

    sub_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=tools, verbose=False)

    queries = [
        "What is TDD and how does it improve code quality?",
        "How does Kubernetes manage containerised workloads?",
        "What is MLOps and what tools are used?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        r = sub_engine.query(q)
        print(f"A: {r}")


# ── 2. Multi-hop cross-domain queries ─────────────────────────────────────────
def demo_cross_domain(tools):
    print("\n=== 2. Cross-Domain Multi-Hop Queries ===")

    sub_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=tools, verbose=True)

    complex_queries = [
        "How do CI/CD principles from software engineering apply to ML model deployment and monitoring?",
        "What connections exist between microservices architecture and MLOps data pipelines?",
    ]

    for q in complex_queries:
        print(f"\n{'='*60}")
        print(f"Complex Query: {q}")
        print("="*60)
        r = sub_engine.query(q)
        print(f"\nSynthesised Answer:\n{r}")


# ── 3. Source attribution ──────────────────────────────────────────────────────
def demo_source_attribution(tools):
    print("\n=== 3. Source Attribution ===")

    # Use individual query engines for source tracking
    q = "What are the best practices for deploying and monitoring machine learning models?"
    print(f"Query: {q}\n")

    for tool in tools:
        engine = tool.query_engine
        try:
            response = engine.query(q)
            source_info = tool.metadata.name.upper()
            print(f"[{source_info}]")
            print(f"  Answer: {str(response)[:200]}...")
            if hasattr(response, "source_nodes") and response.source_nodes:
                print(f"  Sources: {[n.metadata.get('category', 'unknown') for n in response.source_nodes[:2]]}")
        except Exception:
            pass


if __name__ == "__main__":
    tool_list = build_knowledge_base()
    print(f"Knowledge base ready with {len(tool_list)} domain indexes.\n")

    demo_single_domain(tool_list)
    demo_cross_domain(tool_list)
    demo_source_attribution(tool_list)
