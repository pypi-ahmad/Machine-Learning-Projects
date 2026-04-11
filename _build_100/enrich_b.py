"""Enrich batch B — Projects 64-70 (Eval & Observability) to 10+ cells each."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── 64 — Tool Selection Benchmark ───────────────────────────────────
    paths.append(write_nb(7, "64_Local_Tool_Selection_Benchmark", [
        md("# Project 64 — Local Tool Selection Benchmark\n## Evaluate LLM Tool-Routing Accuracy with Ground Truth\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Define Tool Registry & Test Suite"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import pandas as pd, json, time

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

tool_registry = [
    {"name": "calculator",      "desc": "Perform math calculations and unit conversions"},
    {"name": "web_search",      "desc": "Search the internet for current information"},
    {"name": "file_reader",     "desc": "Read and display file contents"},
    {"name": "code_runner",     "desc": "Execute Python code snippets"},
    {"name": "database_query",  "desc": "Run SQL queries against a database"},
    {"name": "email_sender",    "desc": "Compose and send emails"},
    {"name": "calendar",        "desc": "Schedule meetings and check availability"},
    {"name": "translator",      "desc": "Translate text between languages"},
]

test_cases = [
    ("What is 15% of $230?", "calculator"),
    ("Find recent papers on transformers", "web_search"),
    ("Show me the README.md file", "file_reader"),
    ("Execute the data processing script", "code_runner"),
    ("How many orders were placed last month?", "database_query"),
    ("Send the weekly report to the team", "email_sender"),
    ("Am I free at 3pm on Thursday?", "calendar"),
    ("Translate 'hello world' to Japanese", "translator"),
    ("Calculate compound interest on $5000 at 4% for 3 years", "calculator"),
    ("What files are in the /data directory?", "file_reader"),
    ("Schedule a meeting with Alice for next Tuesday", "calendar"),
    ("What's the current stock price of AAPL?", "web_search"),
    ("Run the test suite and show results", "code_runner"),
    ("How many users signed up this week?", "database_query"),
    ("Notify the team about the deployment", "email_sender"),
    ("Convert this paragraph to Spanish", "translator"),
]
print(f"Registry: {len(tool_registry)} tools | Test suite: {len(test_cases)} cases")
"""),
        md("## Step 2 — Run Benchmark"),
        code("""
class ToolChoice(BaseModel):
    selected_tool: str = Field(description="Tool name from registry")
    confidence: float = Field(ge=0, le=1)
    reasoning: str

selector = llm.with_structured_output(ToolChoice)
tool_desc_block = "\\n".join([f"- {t['name']}: {t['desc']}" for t in tool_registry])

results = []
for query, expected in test_cases:
    start = time.time()
    choice = selector.invoke(
        f"Available tools:\\n{tool_desc_block}\\n\\nUser request: {query}\\nSelect the best tool."
    )
    elapsed = time.time() - start
    correct = choice.selected_tool.strip().lower() == expected.lower()
    results.append({
        "query": query[:45], "expected": expected,
        "selected": choice.selected_tool, "correct": correct,
        "confidence": choice.confidence, "latency": round(elapsed, 2),
    })

df = pd.DataFrame(results)
accuracy = df["correct"].mean()
print(f"Overall accuracy: {accuracy:.0%} ({df['correct'].sum()}/{len(df)})")
"""),
        md("## Step 3 — Error Analysis"),
        code("""
print("DETAILED RESULTS")
print("=" * 70)
for _, row in df.iterrows():
    icon = "✓" if row["correct"] else "✗"
    print(f"  {icon} {row['query']:<47} exp={row['expected']:<15} got={row['selected']}")

# Misclassified
errors = df[~df["correct"]]
if len(errors) > 0:
    print(f"\\nERROR ANALYSIS — {len(errors)} misclassifications:")
    for _, row in errors.iterrows():
        print(f"  '{row['query']}' → expected {row['expected']}, got {row['selected']}")
else:
    print("\\nPerfect score — no misclassifications!")

# Confidence distribution
print(f"\\nConfidence stats:")
print(f"  Mean:  {df['confidence'].mean():.2f}")
print(f"  Correct items avg confidence: {df[df['correct']]['confidence'].mean():.2f}")
if len(errors) > 0:
    print(f"  Wrong items avg confidence:   {errors['confidence'].mean():.2f}")
"""),
        md("## Step 4 — Confusion Matrix & per-Tool Precision"),
        code("""
# Per-tool precision
tool_stats = []
for tool in [t["name"] for t in tool_registry]:
    predicted = df[df["selected"] == tool]
    expected_set = df[df["expected"] == tool]
    tp = len(predicted[predicted["correct"]])
    fp = len(predicted[~predicted["correct"]])
    fn = len(expected_set) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    tool_stats.append({"tool": tool, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn})

stats_df = pd.DataFrame(tool_stats)
print("PER-TOOL PRECISION & RECALL")
print("=" * 60)
print(stats_df.to_string(index=False))

print(f"\\nAvg latency: {df['latency'].mean():.2f}s")
print(f"P95 latency: {df['latency'].quantile(0.95):.2f}s")
"""),
        md("## What You Learned\n- **Tool selection benchmarking** with ground truth labels\n- **Error analysis** for misclassified routing decisions\n- **Per-tool precision & recall** metrics\n- **Confidence calibration** analysis"),
    ]))

    # ── 65 — Hallucination Audit ────────────────────────────────────────
    paths.append(write_nb(7, "65_Local_Hallucination_Audit", [
        md("# Project 65 — Local Hallucination Audit\n## Claim Extraction → Source Verification → Scoring Dashboard\n\n**Stack:** LangChain · Ollama · Pydantic · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Ground Truth Sources"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

sources = {
    "company_report": (
        "Acme Corp revenue was $2.3M in Q3 2024, up 15% from Q2. "
        "The company has 150 employees across 3 offices in NYC, London, and Tokyo. "
        "CEO is Jane Smith, appointed in 2021. CTO is Bob Chen."
    ),
    "product_specs": (
        "Widget X weighs 2.5 kg, costs $49.99 retail. Available in blue and red. "
        "Battery life is 8 hours. Launched March 2024. Uses USB-C charging. "
        "IP67 water resistance rating."
    ),
    "policy_doc": (
        "Employees get 20 PTO days per year. Health insurance covers 80% of premiums. "
        "401k match is 4% of salary. Remote work allowed 3 days per week. "
        "Parental leave is 12 weeks paid."
    ),
}
print(f"Source documents: {len(sources)}")
for k, v in sources.items():
    print(f"  {k}: {len(v)} chars")
"""),
        md("## Step 2 — Generate Answers to Audit"),
        code("""
# Generate answers with varying temperatures (to induce hallucinations)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the source. Include specific numbers and details."),
    ("human", "Source: {source}\\n\\nQuestion: {question}")
])

audit_cases = [
    {"q": "What was Acme Corp's Q3 revenue and growth?", "source": "company_report",
     "temp": 0.1},
    {"q": "Tell me about Acme Corp", "source": "company_report",
     "temp": 0.9},  # Higher temp = more hallucination risk
    {"q": "What are Widget X's specifications?", "source": "product_specs",
     "temp": 0.1},
    {"q": "Describe the Widget X product", "source": "product_specs",
     "temp": 0.8},
    {"q": "What are the employee benefits?", "source": "policy_doc",
     "temp": 0.1},
    {"q": "Summarize the company benefits package", "source": "policy_doc",
     "temp": 0.7},
]

answers = []
for case in audit_cases:
    gen = ChatOllama(model="qwen3:8b", temperature=case["temp"])
    chain = qa_prompt | gen | StrOutputParser()
    answer = chain.invoke({"source": sources[case["source"]], "question": case["q"]})
    answers.append({**case, "answer": answer})
    print(f"  T={case['temp']} | {case['q'][:40]}... → {len(answer)} chars")
"""),
        md("## Step 3 — Claim Extraction & Verification"),
        code("""
class Claim(BaseModel):
    claim_text: str
    status: str = Field(description="supported, contradicted, unverifiable")
    source_evidence: str = Field(description="Quote from source or 'not found'")
    severity: str = Field(description="minor, major, critical")

class AuditResult(BaseModel):
    total_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    hallucination_score: float = Field(ge=0, le=1)
    claims: list[Claim]

auditor = llm.with_structured_output(AuditResult)

all_results = []
for case in answers:
    source = sources[case["source"]]
    result = auditor.invoke(
        f"SOURCE (ground truth):\\n{source}\\n\\n"
        f"ANSWER TO AUDIT:\\n{case['answer']}\\n\\n"
        f"Extract EVERY factual claim. Verify each against the source."
    )
    all_results.append({"case": case, "audit": result})
    print(f"\\nQ: {case['q'][:40]}... (temp={case['temp']})")
    print(f"  Claims: {result.total_claims} | Hallucination: {result.hallucination_score:.0%}")
    for c in result.claims:
        icon = {"supported":"✓","contradicted":"✗","unverifiable":"?"}[c.status]
        print(f"    {icon} [{c.status}] {c.claim_text[:60]}")
"""),
        md("## Step 4 — Hallucination Dashboard"),
        code("""
rows = []
for r in all_results:
    rows.append({
        "question": r["case"]["q"][:40],
        "temperature": r["case"]["temp"],
        "total_claims": r["audit"].total_claims,
        "supported": r["audit"].supported,
        "contradicted": r["audit"].contradicted,
        "unverifiable": r["audit"].unverifiable,
        "hallucination_score": r["audit"].hallucination_score,
    })

dashboard = pd.DataFrame(rows)
print("HALLUCINATION AUDIT DASHBOARD")
print("=" * 70)
print(dashboard.to_string(index=False))

print(f"\\nKEY METRICS:")
print(f"  Avg hallucination score: {dashboard['hallucination_score'].mean():.0%}")
print(f"  Worst case: {dashboard['hallucination_score'].max():.0%}")
print(f"  Total claims audited: {dashboard['total_claims'].sum()}")
print(f"  Total contradicted: {dashboard['contradicted'].sum()}")

# Temperature correlation
by_temp = dashboard.groupby("temperature")["hallucination_score"].mean()
print(f"\\nHallucination by temperature:")
for temp, score in by_temp.items():
    print(f"  T={temp}: {score:.0%}")
"""),
        md("## What You Learned\n- **Claim-level fact checking** against source documents\n- **Temperature impact** on hallucination rates\n- **Hallucination scoring** with severity classification\n- **Audit dashboard** for systematic quality tracking"),
    ]))

    # ── 66 — Groundedness Checker ───────────────────────────────────────
    paths.append(write_nb(7, "66_Local_Groundedness_Checker", [
        md("# Project 66 — Local Groundedness Checker\n## Sentence-Level Grounding Analysis for RAG Outputs\n\n**Stack:** LangChain · Ollama · Pydantic · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Test Cases with Context & Answers"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import pandas as pd, json

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

test_pairs = [
    {
        "context": "Python was created by Guido van Rossum and first released in 1991. "
                   "It emphasizes code readability with significant whitespace. "
                   "Python supports multiple paradigms: procedural, OOP, and functional.",
        "answer": "Python was created by Guido van Rossum in 1991. It's the most popular "
                  "programming language in the world and emphasizes readability. "
                  "It supports OOP, procedural, and functional programming."
    },
    {
        "context": "Docker containers share the host OS kernel and are lighter than VMs. "
                   "Images are built from Dockerfiles. Containers are ephemeral by default.",
        "answer": "Docker uses containers that share the OS kernel. They're lighter than VMs. "
                  "Docker was created in 2013 by Solomon Hykes at dotCloud. "
                  "Kubernetes manages Docker containers at scale."
    },
    {
        "context": "PostgreSQL is an open-source relational database. It supports JSON, "
                   "full-text search, and ACID transactions. Default port is 5432.",
        "answer": "PostgreSQL is an open-source database supporting JSON and ACID. "
                  "It runs on port 5432 by default. It's faster than MySQL for complex queries."
    },
]
print(f"Test pairs: {len(test_pairs)}")
"""),
        md("## Step 2 — Sentence-Level Grounding Check"),
        code("""
class SentenceGround(BaseModel):
    sentence: str
    grounded: bool
    evidence: str = Field(description="Supporting quote from context, or 'none'")
    category: str = Field(description="fully_supported, partially_supported, unsupported, fabricated")

class GroundednessReport(BaseModel):
    total_sentences: int
    grounded_count: int
    ungrounded_count: int
    score: float = Field(ge=0, le=1)
    sentences: list[SentenceGround]

checker = llm.with_structured_output(GroundednessReport)

reports = []
for i, pair in enumerate(test_pairs):
    report = checker.invoke(
        f"Check each sentence in the ANSWER for grounding in the CONTEXT.\\n\\n"
        f"CONTEXT:\\n{pair['context']}\\n\\nANSWER:\\n{pair['answer']}"
    )
    reports.append(report)
    print(f"\\nPair {i+1}: score={report.score:.0%} ({report.grounded_count}/{report.total_sentences} grounded)")
    for s in report.sentences:
        icon = "✓" if s.grounded else "✗"
        print(f"  {icon} [{s.category}] {s.sentence[:60]}")
        if not s.grounded:
            print(f"    Evidence: {s.evidence}")
"""),
        md("## Step 3 — Aggregated Metrics"),
        code("""
rows = []
for i, report in enumerate(reports):
    for s in report.sentences:
        rows.append({
            "pair": i + 1,
            "sentence": s.sentence[:50],
            "grounded": s.grounded,
            "category": s.category,
        })

df = pd.DataFrame(rows)
print("GROUNDEDNESS SUMMARY")
print("=" * 50)
print(f"Total sentences analyzed: {len(df)}")
print(f"Grounded:   {df['grounded'].sum()} ({df['grounded'].mean():.0%})")
print(f"Ungrounded: {(~df['grounded']).sum()}")
print(f"\\nBy category:")
print(df["category"].value_counts().to_string())
print(f"\\nPer-pair scores:")
for i, report in enumerate(reports):
    print(f"  Pair {i+1}: {report.score:.0%}")
"""),
        md("## Step 4 — Improvement Suggestions"),
        code("""
improve_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given ungrounded sentences, suggest how to fix the RAG answer "
     "to only contain information from the context. Be specific."),
    ("human", "Context: {context}\\n\\nUngrounded sentences: {ungrounded}")
])
improve_chain = improve_prompt | llm | StrOutputParser()

for i, (pair, report) in enumerate(zip(test_pairs, reports)):
    ungrounded = [s.sentence for s in report.sentences if not s.grounded]
    if ungrounded:
        fix = improve_chain.invoke({
            "context": pair["context"],
            "ungrounded": "\\n".join(ungrounded),
        })
        print(f"\\nPair {i+1} improvements:")
        print(fix[:300])
"""),
        md("## What You Learned\n- **Sentence-level grounding** analysis\n- **Evidence mapping** from context to claims\n- **Category classification**: supported, partial, unsupported, fabricated\n- **Auto-improvement suggestions** for RAG outputs"),
    ]))

    # ── 67 — Structured Output Test ─────────────────────────────────────
    paths.append(write_nb(7, "67_Local_Structured_Output_Test", [
        md("# Project 67 — Local Structured Output Reliability Test\n## Test JSON/Pydantic Schema Adherence Across Complexities\n\n**Stack:** LangChain · Ollama · Pydantic · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Define Test Schemas (Easy → Hard)"),
        code("""
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import json, time, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

# Level 1: Simple flat schema
class PersonInfo(BaseModel):
    name: str
    age: int
    city: str
    occupation: str

# Level 2: Nested schema
class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    country: str

class ContactInfo(BaseModel):
    person: PersonInfo
    address: Address
    phone: str
    email: str

# Level 3: Lists and enums
class SkillEntry(BaseModel):
    name: str
    level: str = Field(description="beginner, intermediate, expert")
    years: int

class Resume(BaseModel):
    name: str
    title: str
    skills: list[SkillEntry]
    education: list[str]
    summary: str

# Level 4: Complex nested with constraints
class APIEndpoint(BaseModel):
    method: str = Field(description="GET, POST, PUT, DELETE")
    path: str
    description: str
    parameters: list[str]
    response_codes: list[int]
    requires_auth: bool

class APISpec(BaseModel):
    service_name: str
    version: str
    base_url: str
    endpoints: list[APIEndpoint]

test_schemas = [
    ("L1-flat", PersonInfo, "Extract: Alice Johnson, 32, from Seattle, software engineer"),
    ("L2-nested", ContactInfo, "Extract: Bob Smith, 45, NYC, teacher. "
     "Address: 123 Main St, New York, 10001, USA. Phone: 555-1234, bob@email.com"),
    ("L3-lists", Resume, "Create a resume for Carol, Senior Data Scientist, "
     "skills: Python (expert, 8yr), SQL (intermediate, 4yr), ML (expert, 6yr). "
     "Education: MIT CS PhD, Stanford BS. Summary: Experienced DS leader."),
    ("L4-complex", APISpec, "Design a user management API v2.0 at /api with endpoints: "
     "GET /users (list, params: page,limit, codes: 200,401), "
     "POST /users (create, params: name,email, codes: 201,400,401, auth required), "
     "DELETE /users/:id (delete, codes: 204,404,401, auth required)"),
]
print(f"Test schemas: {len(test_schemas)} levels of complexity")
"""),
        md("## Step 2 — Run Structure Tests"),
        code("""
results = []
for level, schema, prompt in test_schemas:
    structured_llm = llm.with_structured_output(schema)
    start = time.time()
    try:
        output = structured_llm.invoke(prompt)
        valid = True
        output_dict = output.model_dump()
        error = ""
    except Exception as e:
        valid = False
        output_dict = {}
        error = str(e)[:100]
    elapsed = time.time() - start

    results.append({
        "level": level, "valid": valid, "latency": round(elapsed, 2),
        "output_size": len(json.dumps(output_dict, default=str)),
        "error": error,
    })
    icon = "✓" if valid else "✗"
    print(f"  {icon} {level}: {'PASS' if valid else 'FAIL'} ({elapsed:.1f}s)")
    if valid:
        print(f"    {json.dumps(output_dict, indent=2, default=str)[:250]}")
    else:
        print(f"    Error: {error}")
    print()
"""),
        md("## Step 3 — Robustness: Multiple Runs"),
        code("""
# Run each schema 3 times to check consistency
consistency_results = []
for level, schema, prompt in test_schemas:
    structured_llm = llm.with_structured_output(schema)
    passes = 0
    for trial in range(3):
        try:
            output = structured_llm.invoke(prompt)
            passes += 1
        except Exception:
            pass
    consistency_results.append({
        "level": level,
        "pass_rate": f"{passes}/3",
        "consistent": passes == 3,
    })
    print(f"  {level}: {passes}/3 passes {'✓' if passes == 3 else '⚠'}")

print(f"\\nFully consistent schemas: {sum(1 for r in consistency_results if r['consistent'])}/{len(consistency_results)}")
"""),
        md("## Step 4 — Results Dashboard"),
        code("""
rdf = pd.DataFrame(results)
print("STRUCTURED OUTPUT BENCHMARK")
print("=" * 60)
print(rdf[["level","valid","latency","output_size","error"]].to_string(index=False))

pass_rate = rdf["valid"].mean()
print(f"\\nOverall pass rate: {pass_rate:.0%}")
print(f"Avg latency: {rdf['latency'].mean():.2f}s")
print(f"Fastest: {rdf['latency'].min():.2f}s ({rdf.loc[rdf['latency'].idxmin(), 'level']})")
print(f"Slowest: {rdf['latency'].max():.2f}s ({rdf.loc[rdf['latency'].idxmax(), 'level']})")
"""),
        md("## What You Learned\n- **Schema complexity tiers** from flat to deeply nested\n- **Structured output reliability** testing\n- **Consistency checks** with repeated runs\n- **Performance benchmarking** across schema types"),
    ]))

    # ── 68 — Cost & Latency Benchmark ───────────────────────────────────
    paths.append(write_nb(7, "68_Local_Cost_Latency_Benchmark", [
        md("# Project 68 — Local Cost & Latency Benchmark\n## Profile Model Speed × Task Type × Config Variations\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Define Benchmark Matrix"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time, pandas as pd, json

configs = [
    {"name": "precise",  "temp": 0.0, "desc": "Deterministic"},
    {"name": "balanced", "temp": 0.3, "desc": "Balanced"},
    {"name": "creative", "temp": 0.8, "desc": "High creativity"},
]

tasks = [
    ("classify", "Classify as positive/negative/neutral: 'The product exceeded expectations!'"),
    ("extract", "Extract name and date: 'Meeting with Dr. Sarah Chen on March 15, 2025'"),
    ("summarize", "Summarize in 1 sentence: 'Machine learning models learn patterns from training data. "
     "They generalize to new unseen data. Overfitting occurs when models memorize training data.'"),
    ("generate", "Write a product description for noise-cancelling headphones under $100"),
    ("reason", "A snail climbs 3 feet up a wall during the day, slides 2 feet at night. "
     "How many days to reach the top of a 10-foot wall?"),
    ("code", "Write a Python function to check if a number is prime"),
]

print(f"Benchmark: {len(configs)} configs × {len(tasks)} tasks = {len(configs)*len(tasks)} runs")
"""),
        md("## Step 2 — Execute Benchmark"),
        code("""
results = []
for config in configs:
    llm = ChatOllama(model="qwen3:8b", temperature=config["temp"])
    chain = ChatPromptTemplate.from_template("{task}") | llm | StrOutputParser()
    for task_name, prompt in tasks:
        start = time.time()
        output = chain.invoke({"task": prompt})
        elapsed = time.time() - start
        results.append({
            "config": config["name"],
            "task": task_name,
            "latency_s": round(elapsed, 2),
            "input_tokens": len(prompt.split()),
            "output_tokens": len(output.split()),
            "output_chars": len(output),
        })
        print(f"  {config['name']:<10} {task_name:<12} {elapsed:.1f}s  {len(output)} chars")

df = pd.DataFrame(results)
print(f"\\nTotal runs: {len(df)}")
"""),
        md("## Step 3 — Latency Analysis"),
        code("""
# Pivot: task × config
pivot_latency = df.pivot_table(index="task", columns="config", values="latency_s", aggfunc="mean")
print("LATENCY (seconds) — Task × Config")
print("=" * 50)
print(pivot_latency.round(2).to_string())

pivot_output = df.pivot_table(index="task", columns="config", values="output_chars", aggfunc="mean")
print("\\nOUTPUT LENGTH (chars) — Task × Config")
print("=" * 50)
print(pivot_output.round(0).to_string())
"""),
        md("## Step 4 — Throughput & Efficiency"),
        code("""
# Tokens per second estimate
df["tokens_per_sec"] = df["output_tokens"] / df["latency_s"]

print("EFFICIENCY METRICS")
print("=" * 50)
print(f"Avg latency:        {df['latency_s'].mean():.2f}s")
print(f"P50 latency:        {df['latency_s'].median():.2f}s")
print(f"P95 latency:        {df['latency_s'].quantile(0.95):.2f}s")
print(f"Avg tokens/sec:     {df['tokens_per_sec'].mean():.1f}")
print(f"Total output tokens:{df['output_tokens'].sum()}")

print("\\nBy config:")
by_config = df.groupby("config").agg({
    "latency_s": ["mean", "std"],
    "output_chars": "mean",
    "tokens_per_sec": "mean",
}).round(2)
print(by_config.to_string())

print("\\nBy task:")
by_task = df.groupby("task").agg({
    "latency_s": ["mean", "min", "max"],
    "output_chars": "mean",
}).round(2)
print(by_task.to_string())
"""),
        md("## What You Learned\n- **Systematic benchmarking** across configs and tasks\n- **Latency profiling** with P50/P95 percentiles\n- **Throughput measurement** (tokens/sec)\n- **Task-specific performance** characteristics"),
    ]))

    # ── 69 — Memory Strategy Benchmark ──────────────────────────────────
    paths.append(write_nb(7, "69_Local_Memory_Strategy_Benchmark", [
        md("# Project 69 — Local Memory Strategy Benchmark\n## Compare Buffer, Summary, Window & Vector Memory\n\n**Stack:** LangChain · Ollama · ChromaDB · Jupyter"),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb"),
        md("## Step 1 — Shared Conversation History"),
        code("""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import time, json, shutil

llm = ChatOllama(model="qwen3:8b", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

conversation = [
    ("user", "Hi, I'm Alice and I work at TechCorp as a data scientist."),
    ("assistant", "Hello Alice! Nice to meet you. Data science at TechCorp sounds exciting."),
    ("user", "I'm building a churn prediction model using XGBoost on 500K customer records."),
    ("assistant", "XGBoost is excellent for churn. What features are you engineering?"),
    ("user", "Usage frequency, last login recency, support ticket count, and plan tier."),
    ("assistant", "Good feature set. Consider adding engagement metrics like session duration."),
    ("user", "Great idea! I'm also adding NPS scores from our quarterly surveys."),
    ("assistant", "NPS is a strong signal. How's the class imbalance in your dataset?"),
    ("user", "About 5% churn — quite imbalanced. I'm using SMOTE and class weights."),
    ("assistant", "Smart approach. Monitor precision-recall, not just accuracy."),
    ("user", "My current AUC is 0.87. Target is 0.92 by Q2."),
    ("assistant", "0.87 is solid. Feature interactions and hyperparameter tuning could help."),
]

test_questions = [
    "What's my name and where do I work?",
    "What model am I using and what's my AUC?",
    "What features am I using for churn prediction?",
    "How am I handling class imbalance?",
]
print(f"Conversation: {len(conversation)} messages")
print(f"Test questions: {len(test_questions)}")
"""),
        md("## Step 2 — Implement Memory Strategies"),
        code("""
def full_buffer(messages):
    return "\\n".join([f"{role}: {msg}" for role, msg in messages])

def window_memory(messages, window=6):
    recent = messages[-window:]
    return "\\n".join([f"{role}: {msg}" for role, msg in recent])

def summary_memory(messages):
    full = "\\n".join([f"{role}: {msg}" for role, msg in messages])
    chain = ChatPromptTemplate.from_template(
        "Summarize this conversation, preserving ALL key facts (names, numbers, tools, dates): {conv}"
    ) | llm | StrOutputParser()
    return chain.invoke({"conv": full})

def vector_memory(messages):
    docs = [Document(page_content=f"{role}: {msg}", metadata={"turn": i})
            for i, (role, msg) in enumerate(messages)]
    shutil.rmtree("chroma_mem", ignore_errors=True)
    store = Chroma.from_documents(docs, embeddings, persist_directory="chroma_mem")
    return store

strategies = {
    "full_buffer": lambda: full_buffer(conversation[:-2]),  # exclude test q
    "window_4": lambda: window_memory(conversation[:-2], 4),
    "window_8": lambda: window_memory(conversation[:-2], 8),
    "summary": lambda: summary_memory(conversation[:-2]),
}

# Pre-build contexts
contexts = {}
for name, builder in strategies.items():
    start = time.time()
    contexts[name] = builder()
    elapsed = time.time() - start
    ctx_len = len(contexts[name])
    print(f"  {name:<15} {ctx_len:>5} chars  ({elapsed:.1f}s)")

# Vector memory needs special handling
start = time.time()
vector_store = vector_memory(conversation[:-2])
vector_time = time.time() - start
print(f"  {'vector':15} index built ({vector_time:.1f}s)")
"""),
        md("## Step 3 — Evaluate Each Strategy"),
        code("""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the conversation context. Be specific with facts."),
    ("human", "Context:\\n{context}\\n\\nQuestion: {question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

all_results = []
for q in test_questions:
    print(f"\\nQ: {q}")
    for name, context in contexts.items():
        start = time.time()
        answer = qa_chain.invoke({"context": context, "question": q})
        elapsed = time.time() - start
        all_results.append({
            "strategy": name, "question": q[:40],
            "answer": answer[:150], "latency": round(elapsed, 2),
            "context_len": len(context),
        })
        print(f"  [{name}] {answer[:80]}...")

    # Vector memory
    start = time.time()
    relevant = vector_store.similarity_search(q, k=4)
    v_context = "\\n".join([d.page_content for d in relevant])
    answer = qa_chain.invoke({"context": v_context, "question": q})
    elapsed = time.time() - start
    all_results.append({
        "strategy": "vector", "question": q[:40],
        "answer": answer[:150], "latency": round(elapsed, 2),
        "context_len": len(v_context),
    })
    print(f"  [vector] {answer[:80]}...")
"""),
        md("## Step 4 — Comparison Dashboard"),
        code("""
import pandas as pd

rdf = pd.DataFrame(all_results)
print("MEMORY STRATEGY COMPARISON")
print("=" * 60)
summary = rdf.groupby("strategy").agg({
    "latency": ["mean", "max"],
    "context_len": "mean",
}).round(2)
print(summary.to_string())

print("\\nContext efficiency (lower is better):")
for strat in rdf["strategy"].unique():
    sub = rdf[rdf["strategy"] == strat]
    print(f"  {strat:<15} avg_context={sub['context_len'].mean():.0f} chars  avg_latency={sub['latency'].mean():.2f}s")
"""),
        md("## What You Learned\n- **Four memory strategies** compared head-to-head\n- **Tradeoffs**: context length vs. information retention\n- **Vector memory** for selective retrieval from long histories\n- **Quantitative benchmarking** of memory approaches"),
    ]))

    # ── 70 — Agent Trace Analyzer ───────────────────────────────────────
    paths.append(write_nb(7, "70_Local_Agent_Trace_Analyzer", [
        md("# Project 70 — Local Agent Trace Analyzer\n## Instrument, Log & Analyze Agent Execution Traces\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Build Trace Logger"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, time, pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

@dataclass
class TraceStep:
    name: str
    input_summary: str
    output_summary: str
    duration_s: float
    tokens_est: int
    success: bool
    error: str = ""

@dataclass
class Trace:
    task_id: str
    task: str
    start_time: str = ""
    steps: list = field(default_factory=list)
    status: str = "running"
    total_duration: float = 0.0

class TraceCollector:
    def __init__(self):
        self.traces: list[Trace] = []
        self._current: Trace = None
        self._step_start: float = 0

    def begin(self, task_id, task):
        self._current = Trace(task_id=task_id, task=task,
                              start_time=datetime.now().isoformat())

    def step_start(self, name, input_data):
        self._step_start = time.time()
        self._step_name = name
        self._step_input = str(input_data)[:100]

    def step_end(self, output_data, success=True, error=""):
        duration = time.time() - self._step_start
        self._current.steps.append(TraceStep(
            name=self._step_name,
            input_summary=self._step_input,
            output_summary=str(output_data)[:100],
            duration_s=round(duration, 3),
            tokens_est=len(str(output_data).split()),
            success=success, error=error,
        ))

    def end(self, status="success"):
        self._current.status = status
        self._current.total_duration = sum(s.duration_s for s in self._current.steps)
        self.traces.append(self._current)
        self._current = None

collector = TraceCollector()
print("Trace collector ready")
"""),
        md("## Step 2 — Run Traced Agent Tasks"),
        code("""
tasks = [
    ("T1", "Summarize the benefits of containerization"),
    ("T2", "List 5 sorting algorithms with time complexities"),
    ("T3", "Explain the CAP theorem in distributed systems"),
    ("T4", "Compare REST vs GraphQL APIs"),
    ("T5", "What is eventual consistency?"),
]

chain = ChatPromptTemplate.from_template("{task}") | llm | StrOutputParser()

for task_id, task in tasks:
    collector.begin(task_id, task)
    try:
        # Step 1: prompt construction
        collector.step_start("prompt_build", task)
        prompt_text = f"Explain clearly: {task}"
        collector.step_end(prompt_text)

        # Step 2: LLM call
        collector.step_start("llm_generate", prompt_text)
        result = chain.invoke({"task": task})
        collector.step_end(result)

        # Step 3: post-processing
        collector.step_start("postprocess", result[:50])
        word_count = len(result.split())
        collector.step_end(f"{word_count} words")

        collector.end("success")
        print(f"  ✓ {task_id}: {task[:40]}... ({collector.traces[-1].total_duration:.1f}s)")
    except Exception as e:
        collector.step_end("", success=False, error=str(e))
        collector.end("failed")
        print(f"  ✗ {task_id}: {e}")
"""),
        md("## Step 3 — Trace Analysis Dashboard"),
        code("""
# Build analysis dataframe
rows = []
for trace in collector.traces:
    for step in trace.steps:
        rows.append({
            "task_id": trace.task_id,
            "task": trace.task[:30],
            "step": step.name,
            "duration_s": step.duration_s,
            "tokens": step.tokens_est,
            "success": step.success,
        })

tdf = pd.DataFrame(rows)

print("TRACE ANALYSIS")
print("=" * 60)

# Per-task summary
task_summary = tdf.groupby("task_id").agg({
    "duration_s": "sum",
    "tokens": "sum",
    "success": "all",
}).round(3)
print("Per-task:")
print(task_summary.to_string())

# Per-step breakdown
print("\\nPer-step averages:")
step_summary = tdf.groupby("step").agg({
    "duration_s": ["mean", "max"],
    "tokens": "mean",
}).round(3)
print(step_summary.to_string())

# Bottleneck identification
slowest = tdf.loc[tdf["duration_s"].idxmax()]
print(f"\\nBottleneck: {slowest['step']} in {slowest['task_id']} ({slowest['duration_s']:.2f}s)")
"""),
        md("## Step 4 — LLM-Powered Trace Diagnosis"),
        code("""
trace_json = json.dumps([{
    "task_id": t.task_id, "task": t.task, "status": t.status,
    "total_s": t.total_duration,
    "steps": [{"name": s.name, "duration": s.duration_s, "ok": s.success} for s in t.steps]
} for t in collector.traces], indent=2)

diagnosis_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze these agent execution traces. Identify: "
     "1) Performance bottlenecks, 2) Failure patterns, "
     "3) Optimization opportunities, 4) Recommended changes."),
    ("human", "{traces}")
])
diagnosis_chain = diagnosis_prompt | llm | StrOutputParser()

diagnosis = diagnosis_chain.invoke({"traces": trace_json})
print("TRACE DIAGNOSIS")
print("=" * 50)
print(diagnosis[:500])
"""),
        md("## What You Learned\n- **Execution tracing** with step-level instrumentation\n- **Performance profiling** per task and per step\n- **Bottleneck identification** from trace data\n- **LLM-powered diagnosis** of agent behavior"),
    ]))

    print(f"\\nEnriched {len(paths)} notebooks (64-70)")
    for p in paths:
        print(f"  ✓ {p}")

if __name__ == "__main__":
    build()
