"""Enrich batch A — Projects 54-60 (Tool-Using Agents) to 10+ cells each."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── 54 — Local Filesystem Agent ─────────────────────────────────────
    paths.append(write_nb(5, "54_Local_Filesystem_Agent", [
        md("# Project 54 — Local Filesystem Agent\n## Search, Summarize & Organize Files with LLM-Guided Tools\n\n**Stack:** LangChain · Ollama · pathlib · Jupyter"),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Core LLM & Tool Definitions"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import os, json

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

@tool
def list_files(directory: str) -> str:
    \"\"\"List files and directories in the given path.\"\"\"
    p = Path(directory)
    if not p.exists():
        return f"Directory not found: {directory}"
    entries = sorted(p.iterdir())[:30]
    lines = []
    for e in entries:
        kind = "[DIR] " if e.is_dir() else "[FILE]"
        size = e.stat().st_size if e.is_file() else 0
        lines.append(f"{kind} {e.name:<40} {size:>8,} bytes")
    return "\\n".join(lines) or "(empty)"

@tool
def read_file_content(filepath: str) -> str:
    \"\"\"Read the first 1000 characters of a text file.\"\"\"
    try:
        return Path(filepath).read_text(encoding="utf-8", errors="replace")[:1000]
    except Exception as e:
        return f"Error: {e}"

@tool
def file_stats(filepath: str) -> str:
    \"\"\"Return file metadata: size, extension, line count.\"\"\"
    p = Path(filepath)
    if not p.exists():
        return "File not found"
    stat = p.stat()
    try:
        lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
    except Exception:
        lines = -1
    return json.dumps({"name": p.name, "size_bytes": stat.st_size,
                        "extension": p.suffix, "lines": lines})

tools = [list_files, read_file_content, file_stats]
print(f"Tools ready: {[t.name for t in tools]}")
"""),
        md("## Step 2 — Create a Sample Workspace"),
        code("""
# Build a realistic tiny workspace to explore
ws = Path("sample_workspace")
ws.mkdir(exist_ok=True)
(ws / "src").mkdir(exist_ok=True)
(ws / "docs").mkdir(exist_ok=True)
(ws / "data").mkdir(exist_ok=True)

(ws / "README.md").write_text("# Demo Project\\nA sample project for filesystem agent demo.")
(ws / "src" / "main.py").write_text(
    'import os\\n\\ndef main():\\n    print("Hello!")\\n\\nif __name__ == "__main__":\\n    main()\\n')
(ws / "src" / "utils.py").write_text(
    'def add(a, b):\\n    return a + b\\n\\ndef multiply(a, b):\\n    return a * b\\n')
(ws / "docs" / "setup.md").write_text("## Setup\\n1. Install Python 3.10+\\n2. Run `pip install -r req.txt`")
(ws / "data" / "config.json").write_text('{"db_host": "localhost", "port": 5432, "debug": true}')
(ws / "requirements.txt").write_text("langchain>=0.2\\nollama\\nchromadb\\n")
print("Sample workspace created:")
print(list_files.invoke("sample_workspace"))
"""),
        md("## Step 3 — Tool-Calling Research Loop"),
        code("""
def agent_research(question, workspace="sample_workspace"):
    \"\"\"Multi-step: list → read relevant files → answer with context.\"\"\"
    # Step 1 — inventory
    listing = list_files.invoke(workspace)

    # Step 2 — decide which files to read
    pick_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a file listing, pick the files most relevant to the question. "
                   "Return ONLY a JSON list of relative paths."),
        ("human", "Listing:\\n{listing}\\n\\nQuestion: {question}")
    ])
    pick_chain = pick_prompt | llm | StrOutputParser()
    raw = pick_chain.invoke({"listing": listing, "question": question})
    # parse list from response
    import re
    picks = re.findall(r'[\\w/\\-\\.]+\\.[a-z]+', raw)
    print(f"  Files selected: {picks}")

    # Step 3 — gather content
    contents = {}
    for rel in picks[:5]:
        full = str(Path(workspace) / rel)
        contents[rel] = read_file_content.invoke(full)

    # Step 4 — answer
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a filesystem analyst. Use the file contents to answer."),
        ("human", "Question: {question}\\n\\nFile contents:\\n{contents}")
    ])
    answer_chain = answer_prompt | llm | StrOutputParser()
    return answer_chain.invoke({"question": question,
                                 "contents": json.dumps(contents, indent=2)[:3000]})

print("Agent research loop ready")
"""),
        md("## Step 4 — Run Agent Queries"),
        code("""
queries = [
    "What does this project do?",
    "What dependencies are required?",
    "Summarize the Python source code",
    "What is the database configuration?",
]

for q in queries:
    print(f"\\nQ: {q}")
    answer = agent_research(q)
    print(f"A: {answer[:300]}")
    print("-" * 60)
"""),
        md("## Step 5 — File Organization Suggestions"),
        code("""
from pydantic import BaseModel, Field

class OrgSuggestion(BaseModel):
    current_issues: list[str]
    recommended_structure: list[str]
    files_to_move: list[str]
    files_to_create: list[str]

organizer = llm.with_structured_output(OrgSuggestion)

listing = list_files.invoke("sample_workspace")
# also get sub-dirs
for sub in ["src", "docs", "data"]:
    listing += "\\n--- " + sub + " ---\\n"
    listing += list_files.invoke(f"sample_workspace/{sub}")

suggestion = organizer.invoke(
    f"Analyze this project structure and suggest improvements:\\n{listing}"
)
print("ORGANIZATION REPORT")
print("=" * 50)
print(f"Issues: {suggestion.current_issues}")
print(f"Recommended structure: {suggestion.recommended_structure}")
print(f"Files to move: {suggestion.files_to_move}")
print(f"Missing files: {suggestion.files_to_create}")
"""),
        md("## What You Learned\n- **Tool-based file exploration** with pathlib wrappers\n- **Multi-step agent loop**: list → pick → read → answer\n- **Structured organization suggestions** via Pydantic"),
    ]))

    # ── 55 — Local GitHub Repo Reader Agent ─────────────────────────────
    paths.append(write_nb(5, "55_Local_GitHub_Repo_Reader_Agent", [
        md("# Project 55 — Local GitHub Repo Reader Agent\n## Parse Code with AST, Build Embeddings, Answer Questions\n\n**Stack:** LangChain · Ollama · ChromaDB · ast · Jupyter"),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Create Sample Codebase"),
        code("""
from pathlib import Path
import ast, json

Path("sample_repo").mkdir(exist_ok=True)
Path("sample_repo/models").mkdir(exist_ok=True)
Path("sample_repo/utils").mkdir(exist_ok=True)

(Path("sample_repo/models/user.py")).write_text('''
class User:
    \"\"\"Represents a user in the system.\"\"\"
    def __init__(self, name: str, email: str, role: str = "viewer"):
        self.name = name
        self.email = email
        self.role = role

    def is_admin(self) -> bool:
        \"\"\"Check if user has admin privileges.\"\"\"
        return self.role == "admin"

    def to_dict(self) -> dict:
        return {"name": self.name, "email": self.email, "role": self.role}
''')

(Path("sample_repo/models/order.py")).write_text('''
from datetime import datetime

class Order:
    \"\"\"Represents a customer order.\"\"\"
    def __init__(self, user_id: int, items: list, total: float):
        self.user_id = user_id
        self.items = items
        self.total = total
        self.created_at = datetime.now()
        self.status = "pending"

    def confirm(self):
        \"\"\"Mark order as confirmed.\"\"\"
        self.status = "confirmed"

    def cancel(self):
        \"\"\"Cancel the order if still pending.\"\"\"
        if self.status == "pending":
            self.status = "cancelled"
            return True
        return False
''')

(Path("sample_repo/utils/validators.py")).write_text('''
import re

def validate_email(email: str) -> bool:
    \"\"\"Validate email format.\"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> dict:
    \"\"\"Check password strength.\"\"\"
    checks = {
        "min_length": len(password) >= 8,
        "has_upper": any(c.isupper() for c in password),
        "has_digit": any(c.isdigit() for c in password),
    }
    checks["valid"] = all(checks.values())
    return checks
''')

print("Sample repo created with 3 modules")
"""),
        md("## Step 2 — AST-Based Code Analysis"),
        code("""
def extract_code_info(filepath):
    \"\"\"Extract classes, methods, functions, docstrings using AST.\"\"\"
    source = Path(filepath).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"error": "syntax error", "file": filepath}
    info = {"file": filepath, "classes": [], "functions": [], "imports": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "name": item.name,
                        "args": [a.arg for a in item.args.args if a.arg != "self"],
                        "doc": ast.get_docstring(item) or "",
                        "line": item.lineno,
                    })
            info["classes"].append({
                "name": node.name,
                "doc": ast.get_docstring(node) or "",
                "methods": methods,
                "line": node.lineno,
            })
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            getattr(node, '_parent', None), ast.ClassDef
        ):
            info["functions"].append({
                "name": node.name,
                "doc": ast.get_docstring(node) or "",
                "args": [a.arg for a in node.args.args],
                "line": node.lineno,
            })
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            info["imports"].extend(names)
    return info

# Analyze all files
repo_info = []
for py in Path("sample_repo").rglob("*.py"):
    info = extract_code_info(str(py))
    repo_info.append(info)
    classes = [c["name"] for c in info.get("classes", [])]
    funcs = [f["name"] for f in info.get("functions", [])]
    print(f"  {py}: classes={classes}, functions={funcs}")
"""),
        md("## Step 3 — Build Code Embeddings"),
        code("""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import shutil

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# Create documents from code analysis
docs = []
for info in repo_info:
    fp = info["file"]
    source = Path(fp).read_text(encoding="utf-8")
    # One doc per class/function
    for cls in info.get("classes", []):
        text = f"Class {cls['name']} in {fp}: {cls['doc']}\\nMethods: "
        text += ", ".join([m["name"] + "(" + ",".join(m["args"]) + ")" for m in cls["methods"]])
        docs.append(Document(page_content=text, metadata={"file": fp, "type": "class", "name": cls["name"]}))
    for func in info.get("functions", []):
        text = f"Function {func['name']}({','.join(func['args'])}) in {fp}: {func['doc']}"
        docs.append(Document(page_content=text, metadata={"file": fp, "type": "function", "name": func["name"]}))
    # Full file doc
    docs.append(Document(page_content=source[:500], metadata={"file": fp, "type": "source"}))

shutil.rmtree("chroma_repo", ignore_errors=True)
store = Chroma.from_documents(docs, embeddings, persist_directory="chroma_repo")
retriever = store.as_retriever(search_kwargs={"k": 3})
print(f"Indexed {len(docs)} code documents")
"""),
        md("## Step 4 — Code Q&A"),
        code("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert code analyst. Answer questions about the codebase using the context provided."),
    ("human", "Context:\\n{context}\\n\\nQuestion: {question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

questions = [
    "How do I create a new user?",
    "Can an order be cancelled after confirmation?",
    "How is email validation implemented?",
    "What methods does the User class have?",
]

for q in questions:
    docs_found = retriever.invoke(q)
    context = "\\n---\\n".join([d.page_content for d in docs_found])
    answer = qa_chain.invoke({"context": context, "question": q})
    print(f"Q: {q}")
    print(f"A: {answer[:250]}")
    print()
"""),
        md("## Step 5 — Code Improvement Suggestions"),
        code("""
from pydantic import BaseModel, Field

class CodeReview(BaseModel):
    file: str
    issues: list[str]
    improvements: list[str]
    security_notes: list[str]
    overall_quality: str = Field(description="good, fair, poor")

reviewer = llm.with_structured_output(CodeReview)

for info in repo_info:
    source = Path(info["file"]).read_text("utf-8")
    review = reviewer.invoke(f"Review this code for quality, bugs, security:\\n{source}")
    print(f"\\n{info['file']} — {review.overall_quality}")
    for issue in review.issues:
        print(f"  ✗ {issue}")
    for imp in review.improvements:
        print(f"  → {imp}")
"""),
        md("## What You Learned\n- **AST-based code parsing** for structure extraction\n- **Code embeddings** in ChromaDB for semantic search\n- **Codebase Q&A** with retrieval-augmented generation\n- **Automated code review** with structured output"),
    ]))

    # ── 56 — CLI Command Planner Agent ──────────────────────────────────
    paths.append(write_nb(5, "56_Local_CLI_Command_Planner_Agent", [
        md("# Project 56 — Local CLI Command Planner Agent\n## Describe a Task → Get Commands with Safety Checks\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Setup Risk-Aware Command Generator"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

class CLIStep(BaseModel):
    command: str
    explanation: str
    risk_level: str = Field(description="safe, moderate, dangerous")
    reversible: bool
    platform: str = Field(description="linux, windows, macos, cross-platform")

class CLIPlan(BaseModel):
    task: str
    steps: list[CLIStep]
    warnings: list[str]
    prerequisites: list[str]
    estimated_duration: str

planner = llm.with_structured_output(CLIPlan)
print("CLI planner ready!")
"""),
        md("## Step 2 — Generate & Review Plans"),
        code("""
tasks = [
    "Find all Python files larger than 1MB in the project",
    "Set up a Python virtual environment and install requirements",
    "Find and kill a process using port 8080",
    "Create a compressed backup of the src/ directory",
    "Search for 'TODO' comments across all source files",
    "Check disk space usage per directory",
]

for task in tasks:
    print(f"\\nTask: {task}")
    print("-" * 50)
    plan = planner.invoke(f"Create a CLI plan for: {task}")
    if plan.prerequisites:
        print(f"  Prerequisites: {plan.prerequisites}")
    if plan.warnings:
        print(f"  ⚠ Warnings: {plan.warnings}")
    for i, step in enumerate(plan.steps, 1):
        risk_icon = {"safe": "🟢", "moderate": "🟡", "dangerous": "🔴"}[step.risk_level]
        rev = "↩" if step.reversible else "⚠"
        print(f"  {i}. {risk_icon} [{step.platform}] {step.command}")
        print(f"     {step.explanation} {rev}")
    print(f"  Duration: {plan.estimated_duration}")
"""),
        md("## Step 3 — Safety Gate"),
        code("""
class SafetyCheck(BaseModel):
    is_safe: bool
    risk_category: str = Field(description="data_loss, system_change, network, none")
    explanation: str
    safer_alternative: str = Field(default="")

safety = llm.with_structured_output(SafetyCheck)

dangerous_commands = [
    "rm -rf /tmp/*",
    "chmod 777 /etc/passwd",
    "DROP TABLE users;",
    "kill -9 $(pgrep python)",
    "ls -la /home",
]

print("SAFETY GATE")
print("=" * 50)
for cmd in dangerous_commands:
    check = safety.invoke(f"Evaluate the safety of this command: {cmd}")
    icon = "✓" if check.is_safe else "✗"
    print(f"  {icon} {cmd}")
    print(f"    Risk: {check.risk_category} — {check.explanation[:80]}")
    if check.safer_alternative:
        print(f"    Safe alt: {check.safer_alternative}")
    print()
"""),
        md("## Step 4 — Interactive Workflow Builder"),
        code("""
class Workflow(BaseModel):
    name: str
    description: str
    steps: list[CLIStep]
    environment_variables: list[str]
    shell_script: str = Field(description="Combined shell script")

workflow_gen = llm.with_structured_output(Workflow)

workflow = workflow_gen.invoke(
    "Create a CI/CD deployment workflow that: "
    "1) runs tests, 2) builds a Docker image, 3) pushes to registry, "
    "4) deploys to staging. Include error handling."
)

print(f"Workflow: {workflow.name}")
print(f"Description: {workflow.description}")
print(f"\\nSteps:")
for i, s in enumerate(workflow.steps, 1):
    print(f"  {i}. {s.command}")
print(f"\\nEnv vars needed: {workflow.environment_variables}")
print(f"\\nGenerated script:\\n{workflow.shell_script[:500]}")
"""),
        md("## What You Learned\n- **Risk-aware CLI planning** with safety levels\n- **Safety gate** that blocks dangerous commands\n- **Workflow generation** with shell scripts\n- **Structured output** for actionable CLI plans"),
    ]))

    # ── 57 — Expense Processing Agent ───────────────────────────────────
    paths.append(write_nb(5, "57_Local_Expense_Processing_Agent", [
        md("# Project 57 — Local Expense Processing Agent\n## Receipt Parsing → Categorization → Policy Check → Report\n\n**Stack:** LangChain · Ollama · Pydantic · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Define Expense Schemas"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

class ExpenseItem(BaseModel):
    vendor: str
    amount: float
    currency: str = Field(default="USD")
    category: str = Field(description="meals, travel, lodging, supplies, software, other")
    date: str
    description: str
    reimbursable: bool
    receipt_confidence: float = Field(ge=0, le=1, description="Parsing confidence")

class PolicyViolation(BaseModel):
    rule: str
    violation: str
    severity: str = Field(description="warning, violation, block")

class ExpenseReport(BaseModel):
    employee: str
    period: str
    items: list[ExpenseItem]
    violations: list[PolicyViolation]
    total_amount: float
    total_reimbursable: float

extractor = llm.with_structured_output(ExpenseReport)
print("Expense processor ready!")
"""),
        md("## Step 2 — Process Receipts"),
        code("""
receipts = [
    "Uber ride from SFO to downtown hotel — $48.50 — Jan 15, 2025 — business travel for client meeting",
    "Team dinner at Nobu — $385.00 — Jan 15, 2025 — 4 attendees, client entertainment",
    "Marriott Hotel — 2 nights — $520.00 — Jan 15-17, 2025 — conference accommodation",
    "AWS monthly — $1,250.00 — Jan 2025 — dev infrastructure",
    "Office Depot — $67.30 — Jan 18, 2025 — printer toner, paper, pens",
    "Starbucks — $6.75 — Jan 15, 2025 — morning coffee",
    "Delta Airlines — $890.00 — Jan 14, 2025 — roundtrip SFO-NYC economy",
    "Apple store — $1,299.00 — Jan 20, 2025 — MacBook charger and accessories",
]

receipt_block = "\\n".join([f"Receipt {i+1}: {r}" for i, r in enumerate(receipts)])

report = extractor.invoke(
    f"Employee: John Smith\\nPeriod: January 2025\\n\\n"
    f"Company policy: meals under $75/person, flights must be economy, "
    f"hotel max $300/night, purchases over $500 need pre-approval.\\n\\n"
    f"Parse these receipts into a full expense report:\\n{receipt_block}"
)

print(f"EXPENSE REPORT — {report.employee}")
print(f"Period: {report.period}")
print(f"{'='*60}")
for item in report.items:
    flag = "✓" if item.reimbursable else "✗"
    print(f"  {flag}  ${item.amount:>9,.2f}  {item.category:<10}  {item.vendor} — {item.description[:40]}")

print(f"\\nTotal:         ${report.total_amount:,.2f}")
print(f"Reimbursable:  ${report.total_reimbursable:,.2f}")
"""),
        md("## Step 3 — Policy Violation Report"),
        code("""
if report.violations:
    print("POLICY VIOLATIONS")
    print("=" * 50)
    for v in report.violations:
        icon = {"warning":"🟡", "violation":"🔴", "block":"⛔"}[v.severity]
        print(f"  {icon} [{v.severity.upper()}] {v.rule}")
        print(f"     {v.violation}")
else:
    print("No policy violations detected")

# Category summary
df = pd.DataFrame([item.model_dump() for item in report.items])
print("\\nSPENDING BY CATEGORY")
print(df.groupby("category")["amount"].agg(["sum","count"]).sort_values("sum", ascending=False).to_string())
"""),
        md("## Step 4 — Approval Recommendation"),
        code("""
approval_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a finance manager reviewing expense reports. "
     "Provide an approval decision: approve, approve_with_notes, or reject. "
     "Explain your reasoning."),
    ("human", "Total: ${total:,.2f}\\nReimbursable: ${reimb:,.2f}\\n"
     "Violations: {violations}\\nCategories: {categories}")
])
approval_chain = approval_prompt | llm | StrOutputParser()

decision = approval_chain.invoke({
    "total": report.total_amount,
    "reimb": report.total_reimbursable,
    "violations": json.dumps([v.model_dump() for v in report.violations]),
    "categories": df.groupby("category")["amount"].sum().to_dict() if len(df) > 0 else {},
})
print("APPROVAL DECISION")
print("=" * 50)
print(decision)
"""),
        md("## What You Learned\n- **Multi-stage receipt processing**: parse → categorize → policy-check → approve\n- **Policy violation detection** with structured output\n- **Financial reporting** with pandas aggregation\n- **Approval workflow** with LLM reasoning"),
    ]))

    # ── 58 — Calendar Planner Agent ─────────────────────────────────────
    paths.append(write_nb(5, "58_Local_Calendar_Planner_Agent", [
        md("# Project 58 — Local Calendar Planner Agent\n## Conflict Detection → Smart Scheduling → Daily Briefs\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Calendar Data Model"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

calendar = [
    {"id": 1, "title": "Team Standup",       "day": "Mon-Fri", "start": "09:00", "end": "09:30", "priority": "high"},
    {"id": 2, "title": "Sprint Planning",    "day": "Monday",  "start": "10:00", "end": "12:00", "priority": "high"},
    {"id": 3, "title": "Product Review",     "day": "Tuesday", "start": "10:00", "end": "11:00", "priority": "medium"},
    {"id": 4, "title": "1:1 with Manager",   "day": "Wednesday","start": "14:00", "end": "14:30", "priority": "high"},
    {"id": 5, "title": "Focus Time",         "day": "Thursday", "start": "13:00", "end": "16:00", "priority": "medium"},
    {"id": 6, "title": "Team Lunch",         "day": "Friday",   "start": "12:00", "end": "13:00", "priority": "low"},
    {"id": 7, "title": "Code Review Session","day": "Tue,Thu",  "start": "15:00", "end": "16:00", "priority": "medium"},
]

print(f"Calendar has {len(calendar)} recurring events:")
for e in calendar:
    print(f"  {e['day']:<12} {e['start']}-{e['end']}  [{e['priority']}]  {e['title']}")
"""),
        md("## Step 2 — Conflict Detector"),
        code("""
class ConflictReport(BaseModel):
    conflicts: list[str]
    overloaded_days: list[str]
    free_blocks: list[str] = Field(description="Available 1+ hour blocks per day")
    utilization: float = Field(description="% of 9-5 schedule occupied")

detector = llm.with_structured_output(ConflictReport)

report = detector.invoke(
    f"Analyze this weekly calendar for conflicts, overloaded days, and free blocks.\\n"
    f"Working hours: 9:00-17:00.\\n\\nCalendar:\\n{json.dumps(calendar, indent=2)}"
)

print("CONFLICT REPORT")
print("=" * 50)
print(f"Utilization: {report.utilization:.0%}")
if report.conflicts:
    for c in report.conflicts:
        print(f"  ⚠ {c}")
else:
    print("  No conflicts found")
print(f"\\nOverloaded days: {report.overloaded_days}")
print(f"\\nFree blocks:")
for fb in report.free_blocks:
    print(f"  ✓ {fb}")
"""),
        md("## Step 3 — Smart Meeting Scheduler"),
        code("""
class MeetingSlot(BaseModel):
    day: str
    start: str
    end: str
    score: float = Field(ge=0, le=1, description="Suitability 0-1")
    reasoning: str

class ScheduleProposal(BaseModel):
    request: str
    proposed_slots: list[MeetingSlot]
    best_slot: str
    conflicts_avoided: list[str]

scheduler = llm.with_structured_output(ScheduleProposal)

requests = [
    "Schedule a 90-minute design review with 3 people this week",
    "Find a 2-hour deep work block on Wednesday or Thursday",
    "Add a 30-minute daily check-in, preferably after standup",
    "Schedule a 1-hour all-hands meeting avoiding focus time",
]

for req in requests:
    proposal = scheduler.invoke(
        f"Calendar:\\n{json.dumps(calendar, indent=2)}\\n\\n"
        f"Working hours: 9:00-17:00\\nRequest: {req}"
    )
    print(f"\\nRequest: {req}")
    print(f"Best slot: {proposal.best_slot}")
    for slot in proposal.proposed_slots[:3]:
        print(f"  • {slot.day} {slot.start}-{slot.end} (score={slot.score:.0%}) — {slot.reasoning}")
"""),
        md("## Step 4 — Daily Brief Generator"),
        code("""
brief_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate a friendly daily brief for the given day's schedule. "
     "Include: greeting, today's events with times, preparation reminders, "
     "and suggested focus areas for free blocks."),
    ("human", "Day: {day}\\nCalendar:\\n{events}")
])
brief_chain = brief_prompt | llm | StrOutputParser()

for day in ["Monday", "Wednesday", "Friday"]:
    day_events = [e for e in calendar if day in e["day"] or "Mon-Fri" in e["day"]]
    brief = brief_chain.invoke({
        "day": day,
        "events": json.dumps(day_events, indent=2),
    })
    print(f"\\n{'='*50}")
    print(f"DAILY BRIEF — {day}")
    print("=" * 50)
    print(brief[:400])
"""),
        md("## What You Learned\n- **Calendar conflict detection** with structured analysis\n- **Smart scheduling** scored by suitability\n- **Daily brief generation** from calendar data\n- **Time-slot optimization** with LLM reasoning"),
    ]))

    # ── 59 — CRM Enrichment Agent ───────────────────────────────────────
    paths.append(write_nb(5, "59_Local_CRM_Enrichment_Agent", [
        md("# Project 59 — Local CRM Enrichment Agent\n## Account Analysis → Risk Scoring → Action Plans → Email Drafts\n\n**Stack:** LangChain · Ollama · Pydantic · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — CRM Data"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

accounts = [
    {"name": "Acme Corp", "industry": "Manufacturing", "arr": 50000,
     "renewal": "2025-06-01", "health": "at-risk", "last_contact": "45 days ago",
     "open_tickets": 3, "nps": 25,
     "notes": "Unhappy with recent downtime. Considering competitor. Champion left the company."},
    {"name": "TechStart Inc", "industry": "SaaS", "arr": 12000,
     "renewal": "2025-09-01", "health": "healthy", "last_contact": "7 days ago",
     "open_tickets": 0, "nps": 85,
     "notes": "Interested in upgrading to enterprise. Champion: CTO Sarah. Expanding team."},
    {"name": "GlobalRetail", "industry": "Retail", "arr": 95000,
     "renewal": "2025-04-15", "health": "at-risk", "last_contact": "30 days ago",
     "open_tickets": 5, "nps": 40,
     "notes": "Integration issues with their POS system. Budget review happening next month."},
    {"name": "DataDriven Co", "industry": "Analytics", "arr": 28000,
     "renewal": "2025-12-01", "health": "healthy", "last_contact": "14 days ago",
     "open_tickets": 1, "nps": 72,
     "notes": "Using 60% of features. Potential for add-on module."},
]
print(f"CRM: {len(accounts)} accounts, total ARR: ${sum(a['arr'] for a in accounts):,}")
"""),
        md("## Step 2 — Risk Scoring Engine"),
        code("""
class RiskAssessment(BaseModel):
    account: str
    risk_score: float = Field(ge=0, le=1, description="0=safe, 1=critical risk")
    risk_factors: list[str]
    churn_probability: str = Field(description="low, medium, high, critical")
    revenue_at_risk: float
    days_to_renewal: int

risk_scorer = llm.with_structured_output(RiskAssessment)

assessments = []
for acct in accounts:
    assessment = risk_scorer.invoke(
        f"Assess churn risk for this account (today is 2025-02-01):\\n{json.dumps(acct, indent=2)}"
    )
    assessments.append(assessment)
    icon = {"low":"🟢","medium":"🟡","high":"🔴","critical":"⛔"}[assessment.churn_probability]
    print(f"  {icon} {assessment.account:<20} risk={assessment.risk_score:.0%} "
          f"churn={assessment.churn_probability} ARR@risk=${assessment.revenue_at_risk:,.0f}")
"""),
        md("## Step 3 — Action Plan Generator"),
        code("""
class ActionPlan(BaseModel):
    account: str
    immediate_actions: list[str]
    thirty_day_plan: list[str]
    talking_points: list[str]
    upsell_opportunity: str
    recommended_contact: str = Field(description="Who to contact and when")

planner_llm = llm.with_structured_output(ActionPlan)

for acct, assessment in zip(accounts, assessments):
    plan = planner_llm.invoke(
        f"Create an action plan for this account:\\n"
        f"Account: {json.dumps(acct)}\\nRisk: {assessment.model_dump()}"
    )
    print(f"\\n{'='*50}")
    print(f"ACTION PLAN — {plan.account}")
    print(f"Contact: {plan.recommended_contact}")
    print(f"\\nImmediate:")
    for a in plan.immediate_actions:
        print(f"  → {a}")
    print(f"\\n30-Day Plan:")
    for a in plan.thirty_day_plan:
        print(f"  📅 {a}")
    print(f"\\nUpsell: {plan.upsell_opportunity}")
"""),
        md("## Step 4 — Generate Outreach Emails"),
        code("""
email_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a professional outreach email from a Customer Success Manager. "
     "Be warm, reference specific issues, and propose a call."),
    ("human", "Account: {account}\\nHealth: {health}\\nKey issues: {notes}\\n"
     "NPS: {nps}\\nAction: {action}")
])
email_chain = email_prompt | llm | StrOutputParser()

# Generate emails for at-risk accounts
for acct in accounts:
    if acct["health"] != "at-risk":
        continue
    email = email_chain.invoke({
        "account": acct["name"],
        "health": acct["health"],
        "notes": acct["notes"],
        "nps": acct["nps"],
        "action": "Schedule a review call to address concerns",
    })
    print(f"\\n{'='*50}")
    print(f"EMAIL TO: {acct['name']}")
    print("=" * 50)
    print(email[:500])
"""),
        md("## What You Learned\n- **CRM risk scoring** with structured analysis\n- **Churn prediction** from account signals\n- **Action plan generation** with timelines\n- **Automated outreach email drafting**"),
    ]))

    # ── 60 — Browser Task Agent ─────────────────────────────────────────
    paths.append(write_nb(5, "60_Local_Browser_Task_Agent", [
        md("# Project 60 — Local Browser Task Agent\n## Task Planning → DOM Simulation → Validation\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Browser Action Schema"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

class BrowserAction(BaseModel):
    step: int
    action: str = Field(description="navigate, click, type, scroll, wait, extract, screenshot")
    selector: str = Field(description="CSS selector or URL")
    value: str = Field(default="", description="Text to type or expected value")
    wait_ms: int = Field(default=0, description="Milliseconds to wait after action")
    explanation: str

class BrowserPlan(BaseModel):
    task: str
    url: str
    actions: list[BrowserAction]
    expected_result: str
    error_handling: list[str]
    estimated_seconds: int

planner = llm.with_structured_output(BrowserPlan)
print("Browser task planner ready!")
"""),
        md("## Step 2 — Generate Task Plans"),
        code("""
tasks = [
    "Log into a web application, navigate to reports, download the latest CSV",
    "Search an e-commerce site for 'wireless keyboard', filter by price under $50, add the top result to cart",
    "Fill out a job application form with name, email, upload resume, and submit",
    "Navigate to a dashboard, take a screenshot of the revenue chart, extract the total value",
]

plans = []
for task in tasks:
    plan = planner.invoke(f"Create a browser automation plan: {task}")
    plans.append(plan)
    print(f"\\n{'='*60}")
    print(f"Task: {plan.task}")
    print(f"URL: {plan.url}")
    print(f"Steps: {len(plan.actions)} | Est: {plan.estimated_seconds}s")
    for a in plan.actions:
        print(f"  {a.step}. [{a.action:<10}] {a.selector}")
        if a.value:
            print(f"     Value: {a.value}")
        print(f"     {a.explanation}")
"""),
        md("## Step 3 — DOM Simulation & Validation"),
        code("""
# Simulate a simple DOM for testing
mock_dom = {
    "pages": {
        "/login": {
            "elements": {
                "#email": {"type": "input", "value": ""},
                "#password": {"type": "input", "value": ""},
                "#login-btn": {"type": "button", "text": "Sign In"},
            }
        },
        "/dashboard": {
            "elements": {
                "#revenue-chart": {"type": "div", "text": "Revenue: $1.2M"},
                "#download-btn": {"type": "button", "text": "Download Report"},
                ".nav-reports": {"type": "link", "href": "/reports"},
            }
        },
    }
}

class SimResult(BaseModel):
    success: bool
    steps_executed: int
    steps_failed: int
    final_state: str
    errors: list[str]

simulator = llm.with_structured_output(SimResult)

for plan in plans[:2]:
    result = simulator.invoke(
        f"Simulate this browser plan against the mock DOM.\\n\\n"
        f"Mock DOM: {json.dumps(mock_dom, indent=2)}\\n\\n"
        f"Plan: {json.dumps([a.model_dump() for a in plan.actions], indent=2)}"
    )
    print(f"\\nTask: {plan.task[:50]}...")
    print(f"  Success: {result.success}")
    print(f"  Executed: {result.steps_executed}/{result.steps_executed + result.steps_failed}")
    if result.errors:
        for e in result.errors:
            print(f"  ✗ {e}")
"""),
        md("## Step 4 — Generate Playwright Code"),
        code("""
codegen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert this browser automation plan into Playwright Python code. "
     "Include proper waits, error handling, and assertions."),
    ("human", "Plan:\\n{plan}")
])
codegen_chain = codegen_prompt | llm | StrOutputParser()

for plan in plans[:2]:
    plan_json = json.dumps({
        "task": plan.task,
        "url": plan.url,
        "actions": [a.model_dump() for a in plan.actions],
    })
    playwright_code = codegen_chain.invoke({"plan": plan_json})
    print(f"\\n{'='*60}")
    print(f"PLAYWRIGHT CODE — {plan.task[:40]}...")
    print("=" * 60)
    print(playwright_code[:600])
"""),
        md("## What You Learned\n- **Browser task decomposition** into atomic actions\n- **DOM simulation** for plan validation\n- **Playwright code generation** from plans\n- **Error handling** for robust automation"),
    ]))

    print(f"\\nEnriched {len(paths)} notebooks (54-60)")
    for p in paths:
        print(f"  ✓ {p}")

if __name__ == "__main__":
    build()
