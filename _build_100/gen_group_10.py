"""Group 10 — Projects 91-100: Coding & Developer Agents."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 91: Local Coding Copilot ────────────────────────────────
    paths.append(write_nb(10, "91_Local_Coding_Copilot", [
        md("# Project 91 — Local Coding Copilot\n## Code Generation, Completion & Explanation\n\n**Stack:** LangChain · Ollama · Jupyter"),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Setup Code Assistant"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

class CodeOutput(BaseModel):
    code: str = Field(description="The generated code")
    language: str
    explanation: str
    complexity: str = Field(description="O(1), O(n), O(n^2), etc.")

coder = llm.with_structured_output(CodeOutput)
print("Coding copilot ready!")
"""),
        md("## Step 2 — Code Generation"),
        code("""
tasks = [
    "Write a Python function to find the longest common subsequence of two strings",
    "Implement a basic LRU cache with get and put operations",
    "Create a function that validates an email address using regex",
    "Write a decorator that retries a function N times on exception",
]

for task in tasks:
    print(f"\\nTask: {task}")
    result = coder.invoke(f"Generate Python code for: {task}")
    print(f"Language: {result.language}")
    print(f"Complexity: {result.complexity}")
    print(f"Code:\\n{result.code[:300]}")
    print(f"Explanation: {result.explanation[:200]}")
    print("-"*50)
"""),
        md("## Step 3 — Code Review"),
        code("""
review_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Review this code for:
1. Bugs and edge cases
2. Performance issues
3. Security concerns
4. Style improvements

Provide specific, actionable feedback.\"\"\"),
    ("human", "{code}")
])
review_chain = review_prompt | llm | StrOutputParser()

code_to_review = '''
def process_user_input(data):
    query = f"SELECT * FROM users WHERE name = '{data}'"
    result = db.execute(query)
    password = result[0]['password']
    return eval(data + "_handler")(password)
'''

review = review_chain.invoke({"code": code_to_review})
print("CODE REVIEW")
print("="*50)
print(review)
"""),
        md("## What You Learned\n- **Local code generation** with structured output\n- **Code review** for bugs, security, and style\n- **Complexity analysis** for generated code"),
    ]))

    # ── Project 92: Local Test Case Generator ───────────────────────────
    paths.append(write_nb(10, "92_Local_Test_Case_Generator", [
        md("# Project 92 — Local Test Case Generator\n## Analyze Code → Generate Comprehensive Tests\n\n**Stack:** LangChain · Ollama · Jupyter"),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Source Code to Test"),
        code("""
source_code = '''
class ShoppingCart:
    def __init__(self):
        self.items = {}
        self.discount_code = None

    def add_item(self, name, price, quantity=1):
        if price < 0:
            raise ValueError("Price cannot be negative")
        if quantity < 1:
            raise ValueError("Quantity must be at least 1")
        if name in self.items:
            self.items[name]["quantity"] += quantity
        else:
            self.items[name] = {"price": price, "quantity": quantity}

    def remove_item(self, name):
        if name not in self.items:
            raise KeyError(f"Item {name} not in cart")
        del self.items[name]

    def get_total(self):
        total = sum(item["price"] * item["quantity"] for item in self.items.values())
        if self.discount_code == "SAVE10":
            total *= 0.9
        return round(total, 2)

    def apply_discount(self, code):
        valid_codes = ["SAVE10", "HALF50"]
        if code not in valid_codes:
            raise ValueError(f"Invalid discount code: {code}")
        self.discount_code = code
'''
print(source_code)
"""),
        md("## Step 2 — Generate Test Suite"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

test_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Generate a comprehensive pytest test suite for the given code.
Include:
- Happy path tests
- Edge cases (empty cart, zero values, etc.)
- Error cases (invalid input, missing items)
- Boundary tests
- Integration tests (multiple operations)

Use descriptive test names. Include at least 10 test functions.
Return ONLY valid Python code.\"\"\"),
    ("human", "Generate tests for:\\n{code}")
])
test_chain = test_prompt | llm | StrOutputParser()

tests = test_chain.invoke({"code": source_code})
print("GENERATED TEST SUITE")
print("="*50)
print(tests)
"""),
        md("## Step 3 — Validate Generated Tests"),
        code("""
# Execute the generated tests in-memory
import io, contextlib

# First, execute the source code
exec(source_code)

# Try to run the test code
output = io.StringIO()
try:
    with contextlib.redirect_stdout(output):
        exec(tests)
    print("✓ Tests parsed and loaded successfully")
    print(output.getvalue())
except SyntaxError as e:
    print(f"Syntax error in generated tests: {e}")
except Exception as e:
    print(f"Runtime note: {e}")
    print("(Some tests may need pytest runner to execute fully)")

# Count test functions
test_count = tests.count("def test_")
print(f"\\nGenerated {test_count} test functions")
"""),
        md("## What You Learned\n- **Automated test generation** from source code\n- **Comprehensive test coverage** including edge/error cases\n- **Test validation** by parsing generated code"),
    ]))

    # ── Project 93: Local PR Review Assistant ───────────────────────────
    paths.append(write_nb(10, "93_Local_PR_Review_Assistant", [
        md("# Project 93 — Local PR Review Assistant\n## Parse Diffs → Detect Issues → Generate Review\n\n**Stack:** LangChain · Ollama · Jupyter"),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Sample PR Diffs"),
        code("""
pr_diffs = [
    {
        "file": "auth/login.py",
        "diff": \"\"\"
- def login(username, password):
-     user = db.query(f"SELECT * FROM users WHERE name='{username}'")
-     if user and user.password == password:
-         return create_token(user)
+ def login(username: str, password: str) -> Optional[str]:
+     user = db.query("SELECT * FROM users WHERE name=?", (username,))
+     if user and verify_hash(password, user.password_hash):
+         return create_token(user, expires_in=3600)
+     log_failed_attempt(username)
+     return None
\"\"\"
    },
    {
        "file": "api/endpoints.py",
        "diff": \"\"\"
+ @app.route("/admin/delete-all", methods=["POST"])
+ def delete_all():
+     db.execute("DELETE FROM users")
+     db.execute("DELETE FROM orders")
+     return {"status": "all data deleted"}
\"\"\"
    },
    {
        "file": "utils/helpers.py",
        "diff": \"\"\"
+ def calculate_discount(price, discount):
+     return price - (price * discount / 100)
+
+ # TODO: add input validation
+ # TODO: handle negative discounts
\"\"\"
    },
]
print(f"PR with {len(pr_diffs)} changed files")
"""),
        md("## Step 2 — AI Review"),
        code("""
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

class ReviewComment(BaseModel):
    file: str
    line_context: str
    severity: str = Field(description="critical, warning, suggestion, praise")
    category: str = Field(description="security, bug, performance, style, architecture")
    comment: str
    suggested_fix: str = Field(default="")

class PRReview(BaseModel):
    overall_verdict: str = Field(description="approve, request_changes, comment")
    comments: list[ReviewComment]
    summary: str

reviewer = llm.with_structured_output(PRReview)

diff_text = "\\n\\n".join([f"File: {d['file']}\\n{d['diff']}" for d in pr_diffs])

review = reviewer.invoke(
    f"Review this pull request. Check for security issues, bugs, "
    f"code quality, and best practices.\\n\\n{diff_text}"
)

print(f"PR REVIEW — {review.overall_verdict.upper()}")
print("="*50)
print(f"Summary: {review.summary}")
print(f"\\nComments ({len(review.comments)}):")
for c in review.comments:
    icon = {"critical":"🔴","warning":"🟡","suggestion":"🔵","praise":"🟢"}.get(c.severity,"●")
    print(f"\\n  {icon} [{c.severity}] {c.file} — {c.category}")
    print(f"    {c.comment}")
    if c.suggested_fix:
        print(f"    Fix: {c.suggested_fix}")
"""),
        md("## What You Learned\n- **Automated PR review** with structured feedback\n- **Security and bug detection** in code diffs\n- **Categorized comments** with severity levels"),
    ]))

    # ── Projects 94-99: Template-based coding agent projects ────────────
    coding_projects = [
        (94, "94_Local_Notebook_Refactor_Assistant", "Notebook Refactor Assistant",
         "Analyze Jupyter notebooks and suggest improvements",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

# Sample notebook structure to analyze
notebook_cells = [
    {"type": "code", "source": "import pandas as pd\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\nfrom sklearn.model_selection import train_test_split\\nfrom sklearn.ensemble import RandomForestClassifier\\nimport warnings\\nwarnings.filterwarnings('ignore')"},
    {"type": "code", "source": "df = pd.read_csv('data.csv')\\ndf.head()"},
    {"type": "code", "source": "print(df.shape)\\nprint(df.dtypes)\\nprint(df.isnull().sum())\\nprint(df.describe())"},
    {"type": "code", "source": "# lots of processing in one cell\\ndf = df.dropna()\\ndf['age_bin'] = pd.cut(df['age'], bins=5)\\nX = df.drop('target', axis=1)\\ny = df['target']\\nX_train, X_test, y_train, y_test = train_test_split(X, y)\\nmodel = RandomForestClassifier()\\nmodel.fit(X_train, y_train)\\nprint(model.score(X_test, y_test))"},
]

class NotebookReview(BaseModel):
    cell_count: int
    issues: list[str]
    suggested_structure: list[str]
    missing_elements: list[str]
    refactoring_tips: list[str]
    overall_quality: str = Field(description="good, needs_improvement, poor")

reviewer = llm.with_structured_output(NotebookReview)

cells_text = "\\n\\n".join([f"Cell {i+1} ({c['type']}): {c['source']}" for i, c in enumerate(notebook_cells)])

review = reviewer.invoke(f"Review this Jupyter notebook structure:\\n\\n{cells_text}")

print("NOTEBOOK REVIEW")
print("="*50)
print(f"Quality: {review.overall_quality}")
print(f"\\nIssues:")
for issue in review.issues:
    print(f"  ✗ {issue}")
print(f"\\nSuggested Structure:")
for s in review.suggested_structure:
    print(f"  → {s}")
print(f"\\nMissing Elements:")
for m in review.missing_elements:
    print(f"  ? {m}")
print(f"\\nRefactoring Tips:")
for t in review.refactoring_tips:
    print(f"  💡 {t}")
"""),

        (95, "95_Local_Debugging_Workflow_Agent", "Debugging Workflow Agent",
         "Error log analysis → root cause → fix suggestion pipeline",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

error_logs = [
    {
        "error": "TypeError: cannot unpack non-iterable NoneType object",
        "traceback": "File 'app.py', line 42, in process_data\\n    x, y = get_coordinates(point)\\n"
                     "File 'utils.py', line 15, in get_coordinates\\n    return None",
        "context": "Processing GPS data batch of 1000 points, fails on ~3% of entries",
    },
    {
        "error": "ConnectionError: Max retries exceeded with url: /api/data",
        "traceback": "requests.exceptions.ConnectionError\\n"
                     "urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='localhost', port=8080)",
        "context": "Intermittent failure during peak hours (2-4 PM), works fine otherwise",
    },
    {
        "error": "MemoryError: Unable to allocate 4.00 GiB for array",
        "traceback": "File 'model.py', line 89\\n"
                     "    features = np.zeros((n_samples, n_features))\\n"
                     "numpy.core._exceptions.MemoryError",
        "context": "Processing large dataset with 10M rows × 500 features on 8GB RAM machine",
    },
]

class DebugAnalysis(BaseModel):
    error_type: str
    root_cause: str
    severity: str = Field(description="critical, high, medium, low")
    fix_steps: list[str]
    code_fix: str = Field(description="Suggested code change")
    prevention: str = Field(description="How to prevent this in the future")

debugger = llm.with_structured_output(DebugAnalysis)

for log in error_logs:
    print(f"\\n{'='*50}")
    print(f"Error: {log['error']}")
    analysis = debugger.invoke(
        f"Error: {log['error']}\\nTraceback: {log['traceback']}\\nContext: {log['context']}"
    )
    print(f"  Type:      {analysis.error_type}")
    print(f"  Severity:  {analysis.severity}")
    print(f"  Root Cause: {analysis.root_cause}")
    print(f"  Fix Steps:")
    for step in analysis.fix_steps:
        print(f"    {step}")
    print(f"  Code Fix:  {analysis.code_fix[:200]}")
    print(f"  Prevention: {analysis.prevention}")
"""),

        (96, "96_Local_Documentation_Writer", "Local Documentation Writer",
         "Analyze code and auto-generate README, docstrings, and API docs",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

source_module = '''
class DataPipeline:
    def __init__(self, source, destination, batch_size=1000):
        self.source = source
        self.destination = destination
        self.batch_size = batch_size
        self.stats = {"processed": 0, "errors": 0}

    def extract(self, query=None):
        data = self.source.fetch(query or "SELECT * FROM raw_data")
        return data

    def transform(self, data, rules=None):
        if rules:
            for rule in rules:
                data = rule.apply(data)
        data = [row for row in data if row.get("valid", True)]
        self.stats["processed"] += len(data)
        return data

    def load(self, data):
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i+self.batch_size]
            try:
                self.destination.insert_batch(batch)
            except Exception as e:
                self.stats["errors"] += 1

    def run(self, query=None, rules=None):
        data = self.extract(query)
        data = self.transform(data, rules)
        self.load(data)
        return self.stats
'''

# Generate README
readme_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate a professional README.md for this module. Include: "
     "overview, installation, usage examples, API reference, and contributing section."),
    ("human", "Module code:\\n{code}")
])
readme_chain = readme_prompt | llm | StrOutputParser()
readme = readme_chain.invoke({"code": source_module})
print("GENERATED README.md")
print("="*50)
print(readme[:800])

# Generate docstrings
docstring_prompt = ChatPromptTemplate.from_messages([
    ("system", "Add Google-style docstrings to every class and method. "
     "Include Args, Returns, Raises, and Examples. Return the full code."),
    ("human", "{code}")
])
docstring_chain = docstring_prompt | llm | StrOutputParser()
documented = docstring_chain.invoke({"code": source_module})
print(f"\\n\\nDOCUMENTED CODE")
print("="*50)
print(documented[:800])
"""),

        (97, "97_Local_API_Spec_Explainer", "Local API Spec Explainer",
         "Parse API specifications and generate human-readable documentation",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

# Sample API spec (simplified OpenAPI)
api_spec = {
    "paths": {
        "/users": {
            "GET": {"summary": "List users", "parameters": [
                {"name": "page", "type": "int", "default": 1},
                {"name": "limit", "type": "int", "default": 20}]},
            "POST": {"summary": "Create user", "body": {"name": "str", "email": "str", "role": "str"}},
        },
        "/users/{id}": {
            "GET": {"summary": "Get user", "parameters": [{"name": "id", "type": "int"}]},
            "PUT": {"summary": "Update user", "body": {"name": "str", "email": "str"}},
            "DELETE": {"summary": "Delete user"},
        },
        "/users/{id}/orders": {
            "GET": {"summary": "List user orders", "parameters": [
                {"name": "status", "type": "str", "enum": ["pending","shipped","delivered"]}]},
        },
    }
}

class EndpointDoc(BaseModel):
    method: str
    path: str
    description: str
    curl_example: str
    python_example: str
    common_errors: list[str]

class APIDocumentation(BaseModel):
    overview: str
    endpoints: list[EndpointDoc]
    authentication_notes: str

doc_gen = llm.with_structured_output(APIDocumentation)

docs = doc_gen.invoke(f"Generate documentation for this API:\\n{json.dumps(api_spec, indent=2)}")

print("API DOCUMENTATION")
print("="*50)
print(f"Overview: {docs.overview}")
print(f"Auth: {docs.authentication_notes}")
for ep in docs.endpoints:
    print(f"\\n{ep.method} {ep.path}")
    print(f"  {ep.description}")
    print(f"  curl: {ep.curl_example[:100]}")
    print(f"  python: {ep.python_example[:100]}")
"""),

        (98, "98_Local_Data_Pipeline_Reviewer", "Local Data Pipeline Reviewer",
         "Review ETL code for robustness, error handling, and best practices",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

pipeline_code = '''
import pandas as pd
import sqlite3

def etl_pipeline():
    # Extract
    df = pd.read_csv("raw_data.csv")

    # Transform
    df = df.dropna()
    df["total"] = df["price"] * df["quantity"]
    df["date"] = pd.to_datetime(df["date_str"])
    df = df[df["total"] > 0]

    # Load
    conn = sqlite3.connect("warehouse.db")
    df.to_sql("sales", conn, if_exists="replace")
    conn.close()

    return len(df)
'''

class PipelineReview(BaseModel):
    robustness_score: float = Field(ge=0, le=1)
    issues: list[str]
    missing_features: list[str]
    suggested_improvements: list[str]
    improved_code: str

reviewer = llm.with_structured_output(PipelineReview)

review = reviewer.invoke(
    f"Review this ETL pipeline for production readiness:\\n{pipeline_code}\\n\\n"
    f"Check: error handling, logging, idempotency, data validation, "
    f"connection management, monitoring."
)

print("PIPELINE REVIEW")
print("="*50)
print(f"Robustness: {review.robustness_score:.0%}")
print(f"\\nIssues:")
for i in review.issues:
    print(f"  ✗ {i}")
print(f"\\nMissing:")
for m in review.missing_features:
    print(f"  ? {m}")
print(f"\\nImprovements:")
for s in review.suggested_improvements:
    print(f"  → {s}")
print(f"\\nImproved Code:\\n{review.improved_code[:500]}")
"""),

        (99, "99_Local_AI_Project_Critic", "Local AI Project Critic",
         "Review AI/ML project architecture and suggest improvements",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

project_description = {
    "name": "Customer Churn Predictor",
    "stack": ["Python", "scikit-learn", "Flask", "PostgreSQL"],
    "architecture": "Monolithic Flask app with sklearn model pickle",
    "data_pipeline": "Manual CSV upload, no versioning",
    "model": "Random Forest, trained on 6-month-old data",
    "deployment": "Single Flask server, no health checks",
    "monitoring": "Print statements to console",
    "testing": "No automated tests",
}

class ProjectCritique(BaseModel):
    overall_grade: str = Field(description="A, B, C, D, F")
    strengths: list[str]
    weaknesses: list[str]
    critical_risks: list[str]
    improvement_roadmap: list[str]
    architecture_suggestion: str
    estimated_tech_debt: str = Field(description="low, medium, high, critical")

critic = llm.with_structured_output(ProjectCritique)

import json
critique = critic.invoke(
    f"Critique this AI project for production readiness:\\n{json.dumps(project_description, indent=2)}"
)

print("AI PROJECT CRITIQUE")
print("="*50)
print(f"Grade: {critique.overall_grade}")
print(f"Tech Debt: {critique.estimated_tech_debt}")
print(f"\\nStrengths:")
for s in critique.strengths:
    print(f"  ✓ {s}")
print(f"\\nWeaknesses:")
for w in critique.weaknesses:
    print(f"  ✗ {w}")
print(f"\\nCritical Risks:")
for r in critique.critical_risks:
    print(f"  🔴 {r}")
print(f"\\nImprovement Roadmap:")
for i, step in enumerate(critique.improvement_roadmap, 1):
    print(f"  {i}. {step}")
print(f"\\nArchitecture: {critique.architecture_suggestion[:300]}")
"""),
    ]

    for proj_num, folder, title, desc, main_code in coding_projects:
        paths.append(write_nb(10, folder, [
            md(f"# Project {proj_num} — {title}\n## {desc}\n\n**Stack:** LangChain · Ollama · Jupyter"),
            code("# !pip install -q langchain langchain-ollama pydantic"),
            md("## Implementation"),
            code(main_code),
            md(f"## What You Learned\n- **{title}** — {desc.lower()}\n- **Developer productivity** with local AI\n- **Code intelligence** powered by local LLM"),
        ]))

    # ── Project 100: Local AI Ops Mini-Platform (Capstone) ──────────────
    paths.append(write_nb(10, "100_Local_AI_Ops_Mini_Platform", [
        md("""# Project 100 — Local AI Ops Mini-Platform (Capstone)
## The Grand Finale: Chat + RAG + Tools + Evals — All Combined

**Stack:** LangChain · LangGraph · Ollama · ChromaDB · Pydantic · Jupyter

This capstone project combines techniques from all 99 previous projects into a single
integrated platform: conversational AI, RAG retrieval, tool use, structured output,
evaluation, and observability — all running locally.
"""),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb pydantic"),
        md("## Module 1 — Core LLM & Embeddings"),
        code("""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
import json, time, shutil
from pathlib import Path
from datetime import datetime

# Core models
llm = ChatOllama(model="qwen3:8b", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("✓ Module 1: Core LLM & Embeddings initialized")
"""),
        md("## Module 2 — Knowledge Base (RAG)"),
        code("""
# Build a knowledge base from the project's own documentation
knowledge_docs = [
    Document(page_content="LangChain is a framework for building LLM-powered applications. "
        "It provides chains, agents, and retrieval components. Key abstractions include "
        "ChatModels, Prompts, OutputParsers, and Retrievers.", metadata={"topic": "langchain"}),
    Document(page_content="Ollama runs LLMs locally. Supported models include Llama 3, Qwen, "
        "Mistral, Phi-3, and Gemma. It provides an OpenAI-compatible API at localhost:11434. "
        "Models are pulled with 'ollama pull model_name'.", metadata={"topic": "ollama"}),
    Document(page_content="ChromaDB is an open-source vector database for AI applications. "
        "It stores embeddings and supports similarity search. It can run in-memory or "
        "persistently. Key operations: add, query, delete.", metadata={"topic": "chromadb"}),
    Document(page_content="LangGraph extends LangChain with stateful, multi-step workflows. "
        "It uses StateGraph with typed state dictionaries. Nodes are functions, edges define "
        "flow. Supports conditional routing and human-in-the-loop.", metadata={"topic": "langgraph"}),
    Document(page_content="RAG (Retrieval-Augmented Generation) combines search with generation. "
        "Steps: 1) Index documents as embeddings, 2) Retrieve relevant chunks for a query, "
        "3) Generate answers grounded in retrieved context.", metadata={"topic": "rag"}),
    Document(page_content="CrewAI enables multi-agent collaboration. Define Agents with roles "
        "and backstories, Tasks with descriptions, and Crews that orchestrate execution. "
        "Supports sequential and hierarchical processes.", metadata={"topic": "crewai"}),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(knowledge_docs)

shutil.rmtree("chroma_capstone", ignore_errors=True)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_capstone")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"✓ Module 2: Knowledge base built — {len(chunks)} chunks indexed")
"""),
        md("## Module 3 — Tool Registry"),
        code("""
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    \"\"\"Search the AI knowledge base for information.\"\"\"
    docs = retriever.invoke(query)
    return "\\n".join([f"[{d.metadata.get('topic','?')}] {d.page_content}" for d in docs])

@tool
def analyze_code(code: str) -> str:
    \"\"\"Analyze Python code for issues and improvements.\"\"\"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze this code briefly. List any bugs, security issues, or improvements."),
        ("human", "{code}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"code": code})

@tool
def generate_summary(text: str) -> str:
    \"\"\"Summarize text into 3 key bullet points.\"\"\"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize in exactly 3 bullet points."),
        ("human", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})

tools = [search_knowledge_base, analyze_code, generate_summary]
print(f"✓ Module 3: {len(tools)} tools registered: {[t.name for t in tools]}")
"""),
        md("## Module 4 — Evaluation Engine"),
        code("""
class EvalResult(BaseModel):
    query: str
    answer: str
    groundedness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    quality: float = Field(ge=0, le=1)
    latency_s: float

eval_judge = llm.with_structured_output(EvalResult)

class TraceLog:
    def __init__(self):
        self.entries = []

    def log(self, event, data, duration=0):
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data_size": len(str(data)),
            "duration_s": round(duration, 3),
        })

    def summary(self):
        return {
            "total_events": len(self.entries),
            "total_duration": sum(e["duration_s"] for e in self.entries),
            "events": [e["event"] for e in self.entries],
        }

trace = TraceLog()
print("✓ Module 4: Evaluation engine & trace logger ready")
"""),
        md("## Module 5 — Integrated Pipeline"),
        code("""
def ai_ops_pipeline(query):
    \"\"\"Full pipeline: retrieve → generate → evaluate → trace.\"\"\"
    results = {}

    # Step 1: Retrieve
    start = time.time()
    context_docs = retriever.invoke(query)
    context = "\\n".join([d.page_content for d in context_docs])
    retrieve_time = time.time() - start
    trace.log("retrieve", context, retrieve_time)

    # Step 2: Generate
    start = time.time()
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context. Be accurate and cite sources."),
        ("human", "Context:\\n{context}\\n\\nQuestion: {query}")
    ])
    chain = qa_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query": query})
    generate_time = time.time() - start
    trace.log("generate", answer, generate_time)

    # Step 3: Evaluate
    start = time.time()
    try:
        evaluation = eval_judge.invoke(
            f"Evaluate this Q&A:\\nQuery: {query}\\nAnswer: {answer}\\n"
            f"Context: {context[:300]}"
        )
        eval_scores = {
            "groundedness": evaluation.groundedness,
            "relevance": evaluation.relevance,
            "quality": evaluation.quality,
        }
    except Exception:
        eval_scores = {"groundedness": 0.5, "relevance": 0.5, "quality": 0.5}
    eval_time = time.time() - start
    trace.log("evaluate", eval_scores, eval_time)

    return {
        "query": query,
        "answer": answer,
        "sources": [d.metadata.get("topic") for d in context_docs],
        "scores": eval_scores,
        "timing": {
            "retrieve": round(retrieve_time, 2),
            "generate": round(generate_time, 2),
            "evaluate": round(eval_time, 2),
            "total": round(retrieve_time + generate_time + eval_time, 2),
        },
    }

print("✓ Module 5: Integrated pipeline ready")
"""),
        md("## Module 6 — Run the Platform"),
        code("""
queries = [
    "What is RAG and how does it work?",
    "How does LangGraph differ from LangChain?",
    "What models can I run with Ollama?",
    "Explain how ChromaDB stores embeddings",
    "How do CrewAI agents collaborate?",
]

all_results = []
print("LOCAL AI OPS PLATFORM — RUNNING")
print("="*60)

for q in queries:
    result = ai_ops_pipeline(q)
    all_results.append(result)
    print(f"\\nQ: {result['query']}")
    print(f"A: {result['answer'][:200]}...")
    print(f"Sources: {result['sources']}")
    print(f"Scores: G={result['scores']['groundedness']:.0%} "
          f"R={result['scores']['relevance']:.0%} "
          f"Q={result['scores']['quality']:.0%}")
    print(f"Timing: {result['timing']['total']:.1f}s "
          f"(retrieve={result['timing']['retrieve']:.1f} "
          f"generate={result['timing']['generate']:.1f} "
          f"eval={result['timing']['evaluate']:.1f})")
"""),
        md("## Module 7 — Platform Dashboard"),
        code("""
import pandas as pd

# Performance summary
metrics_df = pd.DataFrame([{
    "query": r["query"][:40],
    "groundedness": r["scores"]["groundedness"],
    "relevance": r["scores"]["relevance"],
    "quality": r["scores"]["quality"],
    "latency": r["timing"]["total"],
} for r in all_results])

print("\\n" + "="*60)
print("PLATFORM DASHBOARD")
print("="*60)
print(f"\\nQueries processed: {len(all_results)}")
print(f"\\nQuality Metrics:")
print(f"  Avg Groundedness: {metrics_df['groundedness'].mean():.0%}")
print(f"  Avg Relevance:    {metrics_df['relevance'].mean():.0%}")
print(f"  Avg Quality:      {metrics_df['quality'].mean():.0%}")
print(f"\\nPerformance:")
print(f"  Avg Latency:  {metrics_df['latency'].mean():.1f}s")
print(f"  P95 Latency:  {metrics_df['latency'].quantile(0.95):.1f}s")

print(f"\\nTrace Summary:")
trace_summary = trace.summary()
print(f"  Total events: {trace_summary['total_events']}")
print(f"  Total time:   {trace_summary['total_duration']:.1f}s")

print(f"\\nDetailed Results:")
print(metrics_df.to_string(index=False))

print(f"\\n{'='*60}")
print("🎉 CAPSTONE COMPLETE — All 100 Projects Finished!")
print("="*60)
"""),
        md("""## Capstone Summary

This project combined ALL major techniques from the 100-project series:

| Module | Technique | From Projects |
|--------|-----------|---------------|
| Core LLM | Ollama + LangChain | 1-10 |
| Knowledge Base | ChromaDB + RAG | 11-30 |
| Tools | Tool registry + execution | 51-60 |
| Evaluation | LLM-as-judge + scoring | 61-70 |
| Observability | Trace logging + metrics | 61-70 |
| Structured Output | Pydantic models | Throughout |
| Pipeline | End-to-end orchestration | 31-40 |

**Congratulations on completing all 100 Local AI Projects!**
"""),
    ]))

    print(f"Group 10 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
