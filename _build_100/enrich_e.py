"""Enrich batch E — Projects 93-99 (Coding & Developer Agents) to 10+ cells each."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── 93 — PR Review Assistant ────────────────────────────────────────
    paths.append(write_nb(10, "93_Local_PR_Review_Assistant", [
        md("# Project 93 — Local PR Review Assistant\n## Diff Analysis → Code Review → Risk Assessment\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Simulated PR Diffs"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

pull_requests = [
    {
        "id": "PR-142",
        "title": "Add rate limiting to API endpoints",
        "author": "alice",
        "files_changed": 4,
        "diff": \"\"\"
--- a/api/middleware.py
+++ b/api/middleware.py
@@ -1,5 +1,28 @@
+from time import time
+from collections import defaultdict
+
+class RateLimiter:
+    def __init__(self, max_requests=100, window=60):
+        self.max_requests = max_requests
+        self.window = window
+        self.requests = defaultdict(list)
+
+    def is_allowed(self, client_ip):
+        now = time()
+        self.requests[client_ip] = [
+            t for t in self.requests[client_ip] if now - t < self.window
+        ]
+        if len(self.requests[client_ip]) >= self.max_requests:
+            return False
+        self.requests[client_ip].append(now)
+        return True
+
 from flask import Flask, request, jsonify
+
+limiter = RateLimiter()
+
 @app.before_request
 def check_rate_limit():
+    if not limiter.is_allowed(request.remote_addr):
+        return jsonify({"error": "rate limited"}), 429
\"\"\"
    },
    {
        "id": "PR-143",
        "title": "Fix SQL injection in user search",
        "author": "bob",
        "files_changed": 2,
        "diff": \"\"\"
--- a/api/users.py
+++ b/api/users.py
@@ -10,7 +10,8 @@
 def search_users(query):
-    sql = f"SELECT * FROM users WHERE name LIKE '%{query}%'"
-    results = db.execute(sql)
+    sql = "SELECT * FROM users WHERE name LIKE :query"
+    results = db.execute(sql, {"query": f"%{query}%"})
     return results
\"\"\"
    },
    {
        "id": "PR-144",
        "title": "Add user preferences table",
        "author": "carol",
        "files_changed": 3,
        "diff": \"\"\"
--- a/models/user.py
+++ b/models/user.py
@@ -15,6 +15,20 @@
+class UserPreference(Base):
+    __tablename__ = 'user_preferences'
+    id = Column(Integer, primary_key=True)
+    user_id = Column(Integer, ForeignKey('users.id'))
+    key = Column(String(100))
+    value = Column(Text)
+    created_at = Column(DateTime, default=datetime.utcnow)
+
--- /dev/null
+++ b/migrations/add_preferences.py
+def upgrade():
+    op.create_table('user_preferences',
+        sa.Column('id', sa.Integer, primary_key=True),
+        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id')),
+        sa.Column('key', sa.String(100)),
+        sa.Column('value', sa.Text),
+    )
\"\"\"
    },
]
print(f"Pull requests to review: {len(pull_requests)}")
"""),
        md("## Step 2 — Automated Code Review"),
        code("""
class ReviewComment(BaseModel):
    file: str
    line_ref: str
    severity: str = Field(description="critical, warning, suggestion, praise")
    comment: str
    suggested_fix: str = ""

class PRReview(BaseModel):
    pr_id: str
    approval: str = Field(description="approve, request_changes, comment")
    risk_level: str = Field(description="low, medium, high, critical")
    summary: str
    comments: list[ReviewComment]
    security_concerns: list[str]
    test_coverage_needed: list[str]
    breaking_changes: bool

reviewer = llm.with_structured_output(PRReview)

reviews = []
for pr in pull_requests:
    review = reviewer.invoke(
        f"Review this pull request:\\n"
        f"Title: {pr['title']}\\nAuthor: {pr['author']}\\n"
        f"Files changed: {pr['files_changed']}\\n\\nDiff:\\n{pr['diff']}"
    )
    reviews.append(review)
    print(f"\\n{'='*50}")
    print(f"{pr['id']}: {pr['title']}")
    print(f"  Decision: {review.approval.upper()} | Risk: {review.risk_level}")
    print(f"  Comments: {len(review.comments)}")
    for c in review.comments:
        icon = {"critical": "🔴", "warning": "🟡", "suggestion": "🔵", "praise": "🟢"}.get(c.severity, "⚪")
        print(f"    {icon} [{c.severity}] {c.comment[:70]}")
    if review.security_concerns:
        print(f"  Security: {review.security_concerns}")
"""),
        md("## Step 3 — Risk Dashboard"),
        code("""
risk_rows = []
for pr, review in zip(pull_requests, reviews):
    risk_rows.append({
        "PR": pr["id"],
        "Title": pr["title"][:30],
        "Author": pr["author"],
        "Risk": review.risk_level,
        "Decision": review.approval,
        "Comments": len(review.comments),
        "Security": len(review.security_concerns),
        "Breaking": review.breaking_changes,
    })

df = pd.DataFrame(risk_rows)
print("PR REVIEW DASHBOARD")
print("=" * 60)
print(df.to_string(index=False))

print(f"\\nSummary:")
print(f"  Approved: {sum(1 for r in reviews if r.approval == 'approve')}")
print(f"  Changes requested: {sum(1 for r in reviews if r.approval == 'request_changes')}")
print(f"  Critical risks: {sum(1 for r in reviews if r.risk_level == 'critical')}")
"""),
        md("## Step 4 — Review Summary for Team"),
        code("""
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a team-facing summary of today's PR reviews. "
     "Highlight security fixes, breaking changes, and items needing attention."),
    ("human", "Reviews:\\n{reviews}")
])
summary_chain = summary_prompt | llm | StrOutputParser()

team_summary = summary_chain.invoke({
    "reviews": json.dumps([r.model_dump() for r in reviews], indent=2)
})
print("TEAM SUMMARY")
print("=" * 50)
print(team_summary[:500])
"""),
        md("## What You Learned\n- **Automated PR review** with structured feedback\n- **Security concern detection** in code changes\n- **Risk-level assessment** per change\n- **Review dashboard** for team visibility"),
    ]))

    # ── 94 — Notebook Refactor Assistant ────────────────────────────────
    paths.append(write_nb(10, "94_Local_Notebook_Refactor_Assistant", [
        md("# Project 94 — Local Notebook Refactor Assistant\n## Analyze → Detect Smells → Suggest Refactors → Generate Clean Code\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Sample Messy Code"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

messy_code_samples = [
    {
        "name": "data_processor.py",
        "code": '''
import pandas as pd
def process(f):
    d = pd.read_csv(f)
    d = d.dropna()
    d = d[d['age'] > 0]
    d = d[d['age'] < 150]
    d['name'] = d['name'].str.strip().str.title()
    d['email'] = d['email'].str.lower()
    d['score'] = d['grade'] * 10 + d['bonus']
    d['level'] = d['score'].apply(lambda x: 'A' if x > 90 else ('B' if x > 80 else ('C' if x > 70 else 'F')))
    d.to_csv('output.csv')
    print('done')
    return d
'''
    },
    {
        "name": "api_handler.py",
        "code": '''
from flask import request, jsonify
def handle():
    try:
        data = request.json
        name = data['name']
        email = data['email']
        age = data['age']
        if not name or len(name) < 2:
            return jsonify({'error': 'bad name'}), 400
        if '@' not in email:
            return jsonify({'error': 'bad email'}), 400
        if age < 0 or age > 200:
            return jsonify({'error': 'bad age'}), 400
        # save to db
        import sqlite3
        conn = sqlite3.connect('users.db')
        conn.execute(f"INSERT INTO users VALUES ('{name}', '{email}', {age})")
        conn.commit()
        conn.close()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
'''
    },
    {
        "name": "report_gen.py",
        "code": '''
def report(data):
    r = ""
    r += "Report\\n"
    r += "======\\n"
    total = 0
    count = 0
    for item in data:
        total += item['amount']
        count += 1
        r += f"{item['name']}: ${item['amount']}\\n"
    r += f"Total: ${total}\\n"
    r += f"Average: ${total/count}\\n"
    r += f"Items: {count}\\n"
    print(r)
    return r
'''
    },
]
print(f"Code samples to refactor: {len(messy_code_samples)}")
"""),
        md("## Step 2 — Code Smell Detection"),
        code("""
class CodeSmell(BaseModel):
    smell_type: str = Field(description="e.g., long_method, sql_injection, magic_number, poor_naming")
    severity: str = Field(description="critical, high, medium, low")
    location: str
    description: str
    fix_category: str = Field(description="security, readability, maintainability, performance")

class CodeAnalysis(BaseModel):
    file_name: str
    quality_score: float = Field(ge=0, le=10)
    smells: list[CodeSmell]
    positive_aspects: list[str]
    complexity_rating: str = Field(description="simple, moderate, complex")
    refactor_priority: str = Field(description="urgent, high, medium, low")

analyzer = llm.with_structured_output(CodeAnalysis)

analyses = []
for sample in messy_code_samples:
    analysis = analyzer.invoke(
        f"Analyze this code for smells and quality issues:\\n\\n"
        f"File: {sample['name']}\\n```python\\n{sample['code']}\\n```"
    )
    analyses.append(analysis)
    print(f"\\n{analysis.file_name}: score={analysis.quality_score}/10 priority={analysis.refactor_priority}")
    for smell in analysis.smells:
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(smell.severity, "⚪")
        print(f"  {icon} {smell.smell_type}: {smell.description[:60]}")
"""),
        md("## Step 3 — Generate Refactored Code"),
        code("""
refactor_prompt = ChatPromptTemplate.from_messages([
    ("system", "Refactor this code addressing ALL identified smells. "
     "Apply best practices: meaningful names, single responsibility, "
     "input validation, parameterized queries, type hints. "
     "Return ONLY the refactored Python code."),
    ("human", "Original code:\\n```python\\n{code}\\n```\\n\\n"
     "Smells to fix:\\n{smells}\\n\\nRefactored code:")
])
refactor_chain = refactor_prompt | llm | StrOutputParser()

for sample, analysis in zip(messy_code_samples, analyses):
    smells_desc = "\\n".join(f"- [{s.severity}] {s.smell_type}: {s.description}" for s in analysis.smells)
    refactored = refactor_chain.invoke({
        "code": sample["code"],
        "smells": smells_desc,
    })
    print(f"\\n{'='*50}")
    print(f"REFACTORED: {sample['name']}")
    print(f"{'='*50}")
    print(refactored[:500])
"""),
        md("## Step 4 — Refactoring Report"),
        code("""
import pandas as pd

rows = []
for a in analyses:
    for smell in a.smells:
        rows.append({
            "file": a.file_name,
            "smell": smell.smell_type,
            "severity": smell.severity,
            "category": smell.fix_category,
            "priority": a.refactor_priority,
        })

df = pd.DataFrame(rows)
print("REFACTORING REPORT")
print("=" * 50)
print(f"Total smells: {len(df)}")
print(f"\\nBy severity: {df['severity'].value_counts().to_dict()}")
print(f"By category: {df['category'].value_counts().to_dict()}")

print(f"\\nFile quality scores:")
for a in analyses:
    bar = "█" * int(a.quality_score)
    print(f"  {a.file_name:<25} {bar} {a.quality_score}/10")
"""),
        md("## What You Learned\n- **Automated code smell detection** with severity levels\n- **Quality scoring** for code files\n- **AI-powered refactoring** with targeted fixes\n- **Refactoring reports** for team prioritization"),
    ]))

    # ── 95 — Debugging Workflow Agent ───────────────────────────────────
    paths.append(write_nb(10, "95_Local_Debugging_Workflow_Agent", [
        md("# Project 95 — Local Debugging Workflow Agent\n## Error → Hypothesize → Investigate → Fix\n\n**Stack:** LangChain · Ollama · Pydantic · LangGraph · Jupyter"),
        code("# !pip install -q langchain langchain-ollama langgraph pydantic"),
        md("## Step 1 — Define Bug Reports"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, Optional
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

bug_reports = [
    {
        "id": "BUG-001",
        "title": "Users get 500 error on login",
        "description": "Since deploy v2.3.1, random users get HTTP 500 on login. "
                       "Happens ~10% of requests. Error log shows: "
                       "'ConnectionPool: max retries exceeded with url: /auth/token'. "
                       "Started after we increased session timeout from 30min to 2hrs.",
        "stack_trace": "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='auth-svc', port=443): "
                       "Max retries exceeded",
        "env": "production, k8s, 3 replicas",
    },
    {
        "id": "BUG-002",
        "title": "Dashboard data shows yesterday's numbers",
        "description": "The analytics dashboard consistently shows data from the previous day. "
                       "Cache was recently moved from Redis to memcached. "
                       "TTL is set to 3600 seconds.",
        "stack_trace": "",
        "env": "production, AWS",
    },
    {
        "id": "BUG-003",
        "title": "Memory leak in background job processor",
        "description": "Worker pods OOM crash every 6 hours. Memory usage grows linearly. "
                       "Processing ~1000 jobs/hour. Each job downloads a file, processes it, "
                       "and uploads the result. File sizes range 1MB-50MB.",
        "stack_trace": "signal: killed (OOMKilled), last memory: 3.8Gi (limit: 4Gi)",
        "env": "production, k8s, 4Gi memory limit",
    },
]
print(f"Bug reports: {len(bug_reports)}")
"""),
        md("## Step 2 — Hypothesis Generation"),
        code("""
class Hypothesis(BaseModel):
    description: str
    probability: float = Field(ge=0, le=1)
    evidence_for: list[str]
    evidence_against: list[str]
    investigation_steps: list[str]

class BugAnalysis(BaseModel):
    bug_id: str
    severity: str = Field(description="P0, P1, P2, P3")
    category: str = Field(description="infrastructure, logic, data, config, resource")
    hypotheses: list[Hypothesis]
    most_likely: str
    immediate_mitigation: str

analyzer = llm.with_structured_output(BugAnalysis)

analyses = []
for bug in bug_reports:
    analysis = analyzer.invoke(
        f"Analyze this bug and generate hypotheses:\\n\\n"
        f"Title: {bug['title']}\\n"
        f"Description: {bug['description']}\\n"
        f"Stack trace: {bug['stack_trace']}\\n"
        f"Environment: {bug['env']}"
    )
    analyses.append(analysis)
    print(f"\\n{analysis.bug_id} [{analysis.severity}] {analysis.category}")
    print(f"  Most likely: {analysis.most_likely}")
    print(f"  Hypotheses: {len(analysis.hypotheses)}")
    for h in analysis.hypotheses[:3]:
        print(f"    • {h.description[:60]}... (P={h.probability:.0%})")
    print(f"  Mitigation: {analysis.immediate_mitigation[:80]}")
"""),
        md("## Step 3 — Debug Investigation Flow (LangGraph)"),
        code("""
from langgraph.graph import StateGraph, END

class DebugState(TypedDict):
    bug: dict
    analysis: dict
    investigation_log: list[str]
    root_cause: str
    fix_plan: str
    status: str

def investigate(state: DebugState) -> DebugState:
    chain = ChatPromptTemplate.from_messages([
        ("system", "You are investigating a bug. Simulate running the investigation steps "
         "and report findings. Be specific about what each step revealed."),
        ("human", "Bug: {bug}\\nHypotheses: {hypotheses}\\nInvestigate:")
    ]) | llm | StrOutputParser()
    findings = chain.invoke({
        "bug": json.dumps(state["bug"]),
        "hypotheses": json.dumps(state["analysis"]),
    })
    state["investigation_log"].append(findings)
    return state

def diagnose(state: DebugState) -> DebugState:
    chain = ChatPromptTemplate.from_messages([
        ("system", "Based on investigation, determine the root cause. Be specific."),
        ("human", "Bug: {bug}\\nFindings: {findings}\\nRoot cause:")
    ]) | llm | StrOutputParser()
    state["root_cause"] = chain.invoke({
        "bug": json.dumps(state["bug"]),
        "findings": state["investigation_log"][-1],
    })
    return state

def plan_fix(state: DebugState) -> DebugState:
    chain = ChatPromptTemplate.from_messages([
        ("system", "Create a detailed fix plan with code changes, config updates, and verification steps."),
        ("human", "Root cause: {cause}\\nEnvironment: {env}\\nFix plan:")
    ]) | llm | StrOutputParser()
    state["fix_plan"] = chain.invoke({
        "cause": state["root_cause"],
        "env": state["bug"].get("env", "unknown"),
    })
    state["status"] = "fix_ready"
    return state

graph = StateGraph(DebugState)
graph.add_node("investigate", investigate)
graph.add_node("diagnose", diagnose)
graph.add_node("plan_fix", plan_fix)
graph.set_entry_point("investigate")
graph.add_edge("investigate", "diagnose")
graph.add_edge("diagnose", "plan_fix")
graph.add_edge("plan_fix", END)
workflow = graph.compile()

print("Debug workflow: investigate → diagnose → plan_fix")
"""),
        md("## Step 4 — Run Debugging for All Bugs"),
        code("""
debug_results = []
for bug, analysis in zip(bug_reports, analyses):
    result = workflow.invoke({
        "bug": bug,
        "analysis": analysis.model_dump(),
        "investigation_log": [],
        "root_cause": "",
        "fix_plan": "",
        "status": "investigating",
    })
    debug_results.append(result)
    print(f"\\n{'='*50}")
    print(f"{bug['id']}: {bug['title']}")
    print(f"Root cause: {result['root_cause'][:150]}")
    print(f"\\nFix plan: {result['fix_plan'][:200]}...")
"""),
        md("## What You Learned\n- **Hypothesis-driven debugging** with structured analysis\n- **LangGraph debug workflow** (investigate → diagnose → fix)\n- **Root cause analysis** from error traces\n- **Automated fix planning** with verification steps"),
    ]))

    # ── 96 — Documentation Writer ───────────────────────────────────────
    paths.append(write_nb(10, "96_Local_Documentation_Writer", [
        md("# Project 96 — Local Documentation Writer\n## Source Code → API Docs → Tutorials → Reference\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Source Code to Document"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

source_files = [
    {
        "name": "auth.py",
        "code": '''
import jwt
from datetime import datetime, timedelta
from hashlib import sha256

class AuthService:
    def __init__(self, secret_key: str, token_expiry: int = 3600):
        self.secret_key = secret_key
        self.token_expiry = token_expiry

    def hash_password(self, password: str) -> str:
        return sha256(password.encode()).hexdigest()

    def create_token(self, user_id: int, roles: list[str]) -> str:
        payload = {
            "sub": user_id,
            "roles": roles,
            "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> dict | None:
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def refresh_token(self, token: str) -> str | None:
        payload = self.verify_token(token)
        if payload:
            return self.create_token(payload["sub"], payload["roles"])
        return None
'''
    },
    {
        "name": "cache.py",
        "code": '''
from typing import Any, Optional
from time import time

class LRUCache:
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: dict[str, tuple[Any, float]] = {}
        self.access_order: list[str] = []

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time() - timestamp < self.ttl:
                self.access_order.remove(key)
                self.access_order.append(key)
                return value
            else:
                del self.cache[key]
                self.access_order.remove(key)
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = (value, time())
        self.access_order.append(key)

    def invalidate(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
            return True
        return False

    def stats(self) -> dict:
        return {"size": len(self.cache), "max_size": self.max_size, "ttl": self.ttl}
'''
    },
]
print(f"Source files to document: {len(source_files)}")
"""),
        md("## Step 2 — Generate API Reference"),
        code("""
class MethodDoc(BaseModel):
    name: str
    signature: str
    description: str
    parameters: list[str]
    returns: str
    raises: list[str] = Field(default_factory=list)
    example: str

class ClassDoc(BaseModel):
    class_name: str
    module: str
    description: str
    constructor_params: list[str]
    methods: list[MethodDoc]
    usage_example: str

documenter = llm.with_structured_output(ClassDoc)

api_docs = []
for src in source_files:
    doc = documenter.invoke(
        f"Generate comprehensive API documentation for this class:\\n\\n"
        f"Module: {src['name']}\\n```python\\n{src['code']}\\n```"
    )
    api_docs.append(doc)
    print(f"\\n{'='*50}")
    print(f"## {doc.class_name} ({doc.module})")
    print(f"{doc.description}")
    print(f"\\nMethods: {len(doc.methods)}")
    for m in doc.methods:
        print(f"  • {m.name}({', '.join(m.parameters[:3])}) → {m.returns}")
"""),
        md("## Step 3 — Generate Tutorial"),
        code("""
tutorial_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a beginner-friendly tutorial for this class. Include:\\n"
     "1. Installation steps\\n2. Quick start example\\n3. Common use cases\\n"
     "4. Error handling\\n5. Best practices\\n\\nUse Markdown with code blocks."),
    ("human", "Class: {class_name}\\nAPI: {api}\\n\\nTutorial:")
])
tutorial_chain = tutorial_prompt | llm | StrOutputParser()

for doc in api_docs:
    tutorial = tutorial_chain.invoke({
        "class_name": doc.class_name,
        "api": json.dumps(doc.model_dump(), indent=2),
    })
    print(f"\\n{'='*50}")
    print(f"TUTORIAL: {doc.class_name}")
    print(f"{'='*50}")
    print(tutorial[:600])
"""),
        md("## Step 4 — Generate Markdown Files"),
        code("""
from pathlib import Path
Path("sample_data/docs").mkdir(parents=True, exist_ok=True)

for doc in api_docs:
    content = f"# {doc.class_name}\\n\\n{doc.description}\\n\\n"
    content += f"## Constructor\\n\\n"
    content += f"Parameters: {', '.join(doc.constructor_params)}\\n\\n"
    content += f"## Methods\\n\\n"
    for m in doc.methods:
        content += f"### `{m.signature}`\\n\\n"
        content += f"{m.description}\\n\\n"
        content += f"**Parameters:**\\n"
        for p in m.parameters:
            content += f"- {p}\\n"
        content += f"\\n**Returns:** {m.returns}\\n\\n"
        if m.example:
            content += f"**Example:**\\n```python\\n{m.example}\\n```\\n\\n"

    content += f"## Quick Start\\n\\n```python\\n{doc.usage_example}\\n```\\n"

    filepath = f"sample_data/docs/{doc.module.replace('.py','')}.md"
    Path(filepath).write_text(content)
    print(f"✓ {filepath} ({len(content)} chars)")
"""),
        md("## What You Learned\n- **Automated API documentation** from source code\n- **Tutorial generation** with examples\n- **Markdown export** for documentation sites\n- **Structured doc models** for consistent output"),
    ]))

    # ── 97 — API Spec Explainer ─────────────────────────────────────────
    paths.append(write_nb(10, "97_Local_API_Spec_Explainer", [
        md("# Project 97 — Local API Spec Explainer\n## OpenAPI Spec → Plain English Docs → SDK Examples\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Sample API Specification"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

api_spec = {
    "openapi": "3.0.0",
    "info": {"title": "Task Manager API", "version": "2.0"},
    "paths": {
        "/tasks": {
            "get": {
                "summary": "List tasks",
                "parameters": [
                    {"name": "status", "in": "query", "schema": {"type": "string", "enum": ["pending","done","archived"]}},
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
                    {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
                ],
                "responses": {"200": {"description": "Array of tasks"}, "401": {"description": "Unauthorized"}},
            },
            "post": {
                "summary": "Create task",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["title"],
                    "properties": {
                        "title": {"type": "string", "maxLength": 200},
                        "description": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low","medium","high"]},
                        "due_date": {"type": "string", "format": "date"},
                    }
                }}}},
                "responses": {"201": {"description": "Task created"}, "400": {"description": "Validation error"}},
            },
        },
        "/tasks/{id}": {
            "get": {"summary": "Get task by ID", "responses": {"200": {}, "404": {}}},
            "put": {"summary": "Update task", "responses": {"200": {}, "404": {}}},
            "delete": {"summary": "Delete task", "responses": {"204": {}, "404": {}}},
        },
    },
}
print(f"API: {api_spec['info']['title']} v{api_spec['info']['version']}")
print(f"Endpoints: {sum(len(v) for v in api_spec['paths'].values())}")
"""),
        md("## Step 2 — Generate Plain English Documentation"),
        code("""
class EndpointDoc(BaseModel):
    method: str
    path: str
    description: str = Field(description="Plain English explanation")
    use_case: str
    required_params: list[str]
    optional_params: list[str]
    success_response: str
    error_responses: list[str]
    rate_limit_note: str = ""

class APIGuide(BaseModel):
    title: str
    overview: str
    authentication: str
    base_url_example: str
    endpoints: list[EndpointDoc]
    common_workflows: list[str]
    error_handling_tips: list[str]

guide_gen = llm.with_structured_output(APIGuide)

guide = guide_gen.invoke(
    f"Create a beginner-friendly API guide from this OpenAPI spec:\\n\\n"
    f"{json.dumps(api_spec, indent=2)}"
)

print(f"API Guide: {guide.title}")
print(f"Overview: {guide.overview}")
print(f"\\nEndpoints ({len(guide.endpoints)}):")
for ep in guide.endpoints:
    print(f"  {ep.method.upper()} {ep.path}")
    print(f"    {ep.description}")
    print(f"    Use case: {ep.use_case}")
"""),
        md("## Step 3 — Generate SDK Code Examples"),
        code("""
sdk_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate code examples for this API endpoint in the specified language. "
     "Include error handling and comments."),
    ("human", "Endpoint: {method} {path}\\nDescription: {desc}\\nLanguage: {lang}\\nCode:")
])
sdk_chain = sdk_prompt | llm | StrOutputParser()

languages = ["Python (requests)", "JavaScript (fetch)", "cURL"]
for ep in guide.endpoints[:3]:
    print(f"\\n{'='*50}")
    print(f"{ep.method.upper()} {ep.path}")
    for lang in languages:
        example = sdk_chain.invoke({
            "method": ep.method, "path": ep.path,
            "desc": ep.description, "lang": lang,
        })
        print(f"\\n  [{lang}]")
        print(f"  {example[:200]}...")
"""),
        md("## Step 4 — Generate Quickstart Guide"),
        code("""
quickstart_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a 'Getting Started in 5 Minutes' guide for this API. "
     "Include: setup, first request, common patterns, error handling."),
    ("human", "API: {title}\\nEndpoints: {endpoints}\\n\\nQuickstart:")
])
quickstart_chain = quickstart_prompt | llm | StrOutputParser()

quickstart = quickstart_chain.invoke({
    "title": guide.title,
    "endpoints": json.dumps([{"method": e.method, "path": e.path, "desc": e.description}
                             for e in guide.endpoints]),
})
print("QUICKSTART GUIDE")
print("=" * 50)
print(quickstart[:600])
"""),
        md("## What You Learned\n- **OpenAPI → plain English** documentation\n- **Multi-language SDK examples** from specs\n- **Quickstart guide generation**\n- **API documentation automation** pipeline"),
    ]))

    # ── 98 — Data Pipeline Reviewer ─────────────────────────────────────
    paths.append(write_nb(10, "98_Local_Data_Pipeline_Reviewer", [
        md("# Project 98 — Local Data Pipeline Reviewer\n## Pipeline Config → Quality Audit → Optimization Suggestions\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Pipeline Definitions"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

pipelines = [
    {
        "name": "user_events_etl",
        "stages": [
            {"name": "ingest", "type": "kafka_consumer", "config": {"topic": "user_events", "batch_size": 1000}},
            {"name": "validate", "type": "schema_check", "config": {"schema": "user_event_v2", "strict": False}},
            {"name": "transform", "type": "python", "config": {"script": "transform_events.py", "timeout": 300}},
            {"name": "deduplicate", "type": "window_dedup", "config": {"window": "1h", "key": "event_id"}},
            {"name": "load", "type": "postgres_insert", "config": {"table": "events", "batch_size": 500}},
        ],
        "schedule": "*/5 * * * *",
        "owner": "data-team",
    },
    {
        "name": "daily_aggregates",
        "stages": [
            {"name": "extract", "type": "sql_query", "config": {"query": "SELECT * FROM events WHERE date = CURRENT_DATE - 1"}},
            {"name": "aggregate", "type": "pandas", "config": {"groupby": ["user_id", "event_type"], "agg": "count"}},
            {"name": "enrich", "type": "api_call", "config": {"endpoint": "/users/batch", "batch_size": 100}},
            {"name": "load", "type": "bigquery_write", "config": {"table": "analytics.daily_summary", "mode": "append"}},
        ],
        "schedule": "0 2 * * *",
        "owner": "analytics-team",
    },
    {
        "name": "ml_feature_pipeline",
        "stages": [
            {"name": "source", "type": "s3_read", "config": {"bucket": "raw-data", "prefix": "features/"}},
            {"name": "clean", "type": "spark", "config": {"script": "clean_features.py", "memory": "8g"}},
            {"name": "compute", "type": "spark", "config": {"script": "compute_features.py", "memory": "16g"}},
            {"name": "validate", "type": "great_expectations", "config": {"suite": "feature_suite"}},
            {"name": "publish", "type": "feature_store_write", "config": {"store": "feast", "entity": "user_features"}},
        ],
        "schedule": "0 4 * * *",
        "owner": "ml-team",
    },
]
print(f"Pipelines to review: {len(pipelines)}")
for p in pipelines:
    print(f"  {p['name']}: {len(p['stages'])} stages, schedule={p['schedule']}")
"""),
        md("## Step 2 — Pipeline Quality Audit"),
        code("""
class StageIssue(BaseModel):
    stage_name: str
    issue_type: str = Field(description="performance, reliability, security, data_quality")
    severity: str = Field(description="critical, high, medium, low")
    description: str
    recommendation: str

class PipelineAudit(BaseModel):
    pipeline_name: str
    health_score: float = Field(ge=0, le=10)
    issues: list[StageIssue]
    missing_stages: list[str]
    redundant_stages: list[str]
    data_quality_gaps: list[str]
    scalability_concerns: list[str]
    estimated_reliability: float = Field(ge=0, le=1)

auditor = llm.with_structured_output(PipelineAudit)

audits = []
for pipeline in pipelines:
    audit = auditor.invoke(
        f"Audit this data pipeline for quality and reliability:\\n\\n"
        f"{json.dumps(pipeline, indent=2)}"
    )
    audits.append(audit)
    print(f"\\n{audit.pipeline_name}: score={audit.health_score}/10 reliability={audit.estimated_reliability:.0%}")
    print(f"  Issues: {len(audit.issues)}")
    for issue in audit.issues[:3]:
        print(f"    [{issue.severity}] {issue.stage_name}: {issue.description[:60]}")
    if audit.missing_stages:
        print(f"  Missing: {audit.missing_stages}")
"""),
        md("## Step 3 — Optimization Suggestions"),
        code("""
optimize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Suggest specific optimizations for this pipeline. "
     "Include: performance improvements, cost reductions, reliability enhancements. "
     "Provide concrete config changes."),
    ("human", "Pipeline: {name}\\nStages: {stages}\\nIssues: {issues}\\n\\nOptimizations:")
])
optimize_chain = optimize_prompt | llm | StrOutputParser()

for pipeline, audit in zip(pipelines, audits):
    optimizations = optimize_chain.invoke({
        "name": pipeline["name"],
        "stages": json.dumps(pipeline["stages"]),
        "issues": json.dumps([i.model_dump() for i in audit.issues]),
    })
    print(f"\\n{'='*50}")
    print(f"OPTIMIZATIONS: {pipeline['name']}")
    print(optimizations[:400])
"""),
        md("## Step 4 — Pipeline Health Dashboard"),
        code("""
rows = []
for audit in audits:
    rows.append({
        "pipeline": audit.pipeline_name,
        "health": audit.health_score,
        "reliability": f"{audit.estimated_reliability:.0%}",
        "issues": len(audit.issues),
        "critical": sum(1 for i in audit.issues if i.severity == "critical"),
        "missing_stages": len(audit.missing_stages),
    })

df = pd.DataFrame(rows)
print("PIPELINE HEALTH DASHBOARD")
print("=" * 60)
print(df.to_string(index=False))

print(f"\\nOverall health: {df['health'].mean():.1f}/10")
print(f"Critical issues: {df['critical'].sum()}")

# Issue breakdown
all_issues = []
for audit in audits:
    for issue in audit.issues:
        all_issues.append({
            "pipeline": audit.pipeline_name,
            "type": issue.issue_type,
            "severity": issue.severity,
        })
idf = pd.DataFrame(all_issues)
if len(idf) > 0:
    print(f"\\nIssues by type: {idf['issue_type'].value_counts().to_dict()}")
    print(f"Issues by severity: {idf['severity'].value_counts().to_dict()}")
"""),
        md("## What You Learned\n- **Pipeline configuration auditing** with structured analysis\n- **Multi-dimension quality scoring** (performance, reliability, security)\n- **Optimization recommendations** with concrete changes\n- **Health dashboard** for pipeline monitoring"),
    ]))

    # ── 99 — AI Project Critic ──────────────────────────────────────────
    paths.append(write_nb(10, "99_Local_AI_Project_Critic", [
        md("# Project 99 — Local AI Project Critic\n## Project Proposal → Feasibility Analysis → Risk Assessment → Recommendations\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Project Proposals to Evaluate"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

proposals = [
    {
        "title": "AI-Powered Customer Support Chatbot",
        "description": "Build a chatbot using RAG to answer customer questions from our knowledge base. "
                       "Handle 80% of tier-1 support tickets automatically.",
        "tech_stack": "LangChain, ChromaDB, Ollama, FastAPI",
        "timeline": "3 months",
        "team_size": 2,
        "budget": "$50K",
        "data_available": "5000 support tickets, 200 FAQ docs, product manuals",
    },
    {
        "title": "Real-Time Fraud Detection with ML",
        "description": "Detect fraudulent transactions in real-time using ML models. "
                       "Process 10K transactions per second with <100ms latency.",
        "tech_stack": "XGBoost, Kafka, Redis, Python",
        "timeline": "6 months",
        "team_size": 4,
        "budget": "$200K",
        "data_available": "2 years of transaction data, 0.1% fraud rate, PII restrictions",
    },
    {
        "title": "Internal Code Review AI",
        "description": "AI that reviews pull requests and suggests improvements. "
                       "Integrate with GitHub, learn from team's coding standards.",
        "tech_stack": "GPT-4, GitHub API, Python",
        "timeline": "2 months",
        "team_size": 1,
        "budget": "$5K",
        "data_available": "GitHub repo history, style guide document",
    },
]
print(f"Proposals to evaluate: {len(proposals)}")
"""),
        md("## Step 2 — Feasibility Analysis"),
        code("""
class FeasibilityDimension(BaseModel):
    dimension: str = Field(description="technical, data, resource, timeline, business")
    score: float = Field(ge=0, le=1)
    assessment: str
    blockers: list[str]

class ProjectCritique(BaseModel):
    title: str
    overall_feasibility: float = Field(ge=0, le=1)
    dimensions: list[FeasibilityDimension]
    strengths: list[str]
    weaknesses: list[str]
    risks: list[str]
    missing_requirements: list[str]
    recommendation: str = Field(description="proceed, proceed_with_changes, delay, reject")
    suggested_changes: list[str]

critic = llm.with_structured_output(ProjectCritique)

critiques = []
for proposal in proposals:
    critique = critic.invoke(
        f"Critically evaluate this AI project proposal:\\n\\n"
        f"{json.dumps(proposal, indent=2)}"
    )
    critiques.append(critique)
    print(f"\\n{'='*50}")
    print(f"{critique.title}")
    print(f"  Feasibility: {critique.overall_feasibility:.0%}")
    print(f"  Recommendation: {critique.recommendation.upper()}")
    for dim in critique.dimensions:
        bar = "█" * int(dim.score * 10)
        print(f"    {dim.dimension:<12} {bar} {dim.score:.0%}")
"""),
        md("## Step 3 — Risk Mitigation Plans"),
        code("""
risk_prompt = ChatPromptTemplate.from_messages([
    ("system", "For each risk, create a specific mitigation plan with "
     "actions, owners, and success criteria."),
    ("human", "Project: {title}\\nRisks: {risks}\\n\\nMitigation plans:")
])
risk_chain = risk_prompt | llm | StrOutputParser()

for critique in critiques:
    if critique.risks:
        plan = risk_chain.invoke({
            "title": critique.title,
            "risks": json.dumps(critique.risks),
        })
        print(f"\\n{'='*50}")
        print(f"RISK MITIGATION: {critique.title}")
        print(plan[:400])
"""),
        md("## Step 4 — Comparison Dashboard"),
        code("""
rows = []
for c in critiques:
    row = {"project": c.title[:30], "feasibility": f"{c.overall_feasibility:.0%}",
           "recommendation": c.recommendation, "risks": len(c.risks),
           "strengths": len(c.strengths), "weaknesses": len(c.weaknesses)}
    for dim in c.dimensions:
        row[dim.dimension] = f"{dim.score:.0%}"
    rows.append(row)

df = pd.DataFrame(rows)
print("PROJECT COMPARISON DASHBOARD")
print("=" * 60)
print(df.to_string(index=False))

# Ranking
print(f"\\nRANKING (by feasibility):")
for i, c in enumerate(sorted(critiques, key=lambda x: x.overall_feasibility, reverse=True), 1):
    print(f"  {i}. {c.title} — {c.overall_feasibility:.0%} [{c.recommendation}]")
    if c.suggested_changes:
        print(f"     Changes: {c.suggested_changes[0]}")
"""),
        md("## What You Learned\n- **Multi-dimensional feasibility analysis** for AI projects\n- **Risk identification and mitigation planning**\n- **Project comparison and ranking**\n- **Structured recommendation framework** (proceed/delay/reject)"),
    ]))

    print(f"\nEnriched {len(paths)} notebooks (93-99)")
    for p in paths:
        print(f"  ✓ {p}")

if __name__ == "__main__":
    build()
