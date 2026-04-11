"""Group 7 — Projects 61-70: Local Eval & Observability."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 61: Local Prompt Evaluation Lab ─────────────────────────
    paths.append(write_nb(7, "61_Local_Prompt_Evaluation_Lab", [
        md("# Project 61 — Local Prompt Evaluation Lab\n## Systematic Prompt Variant Testing & Scoring\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Define Prompt Variants"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd, time

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

# Define multiple prompt variants for the same task
prompt_variants = {
    "zero_shot": "Answer the question: {question}",
    "role_play": "You are an expert scientist. Answer clearly: {question}",
    "chain_of_thought": "Think step by step, then answer: {question}",
    "structured": \"\"\"Answer the question following this format:
ANSWER: <direct answer>
CONFIDENCE: <high/medium/low>
REASONING: <brief reasoning>

Question: {question}\"\"\",
}

test_questions = [
    "Why is the sky blue?",
    "What causes inflation?",
    "How does a neural network learn?",
]
print(f"Testing {len(prompt_variants)} prompt variants × {len(test_questions)} questions")
"""),
        md("## Step 2 — Run All Variants"),
        code("""
results = []
for variant_name, template in prompt_variants.items():
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    for q in test_questions:
        start = time.time()
        answer = chain.invoke({"question": q})
        elapsed = time.time() - start
        results.append({
            "variant": variant_name,
            "question": q[:40],
            "answer_len": len(answer),
            "latency_s": round(elapsed, 2),
            "answer_preview": answer[:120].replace("\\n", " "),
        })
        print(f"  {variant_name} | {q[:30]}... | {elapsed:.1f}s | {len(answer)} chars")

df = pd.DataFrame(results)
print(f"\\nCollected {len(df)} results")
"""),
        md("## Step 3 — Score Answers with LLM-as-Judge"),
        code("""
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Rate the answer quality from 1-5 on:
- Accuracy (is it correct?)
- Clarity (is it easy to understand?)
- Completeness (does it fully address the question?)

Return ONLY a JSON: {{"accuracy": N, "clarity": N, "completeness": N}}\"\"\"),
    ("human", "Question: {question}\\nAnswer: {answer}")
])
judge_chain = judge_prompt | llm | StrOutputParser()

import json
scores = []
for _, row in df.iterrows():
    try:
        raw = judge_chain.invoke({"question": row["question"], "answer": row["answer_preview"]})
        # Extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            score = json.loads(raw[start:end])
        else:
            score = {"accuracy": 3, "clarity": 3, "completeness": 3}
    except Exception:
        score = {"accuracy": 3, "clarity": 3, "completeness": 3}
    score["variant"] = row["variant"]
    score["question"] = row["question"]
    scores.append(score)

scores_df = pd.DataFrame(scores)
scores_df["total"] = scores_df["accuracy"] + scores_df["clarity"] + scores_df["completeness"]
"""),
        md("## Step 4 — Leaderboard"),
        code("""
leaderboard = scores_df.groupby("variant")[["accuracy","clarity","completeness","total"]].mean().round(2)
leaderboard = leaderboard.sort_values("total", ascending=False)
print("PROMPT VARIANT LEADERBOARD")
print("="*60)
print(leaderboard.to_string())

print("\\nBest variant:", leaderboard.index[0])
"""),
        md("## What You Learned\n- **Systematic prompt evaluation** with multiple variants\n- **LLM-as-judge** scoring for quality metrics\n- **Quantitative comparison** of prompt strategies"),
    ]))

    # ── Project 62: Local Output Judge ──────────────────────────────────
    paths.append(write_nb(7, "62_Local_Output_Judge_Notebook", [
        md("# Project 62 — Local Output Judge\n## Model-A Output → Model-B Critique → Structured Scoring\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Generate Candidate Outputs"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.7)

tasks = [
    "Write a function to check if a string is a palindrome",
    "Explain quantum entanglement to a 10-year-old",
    "Draft a professional email declining a meeting invitation",
]

# Generate multiple candidate outputs per task
candidates = {}
for task in tasks:
    outputs = []
    for temp in [0.1, 0.5, 0.9]:
        gen = ChatOllama(model="qwen3:8b", temperature=temp)
        chain = ChatPromptTemplate.from_template("{task}") | gen | StrOutputParser()
        outputs.append(chain.invoke({"task": task}))
    candidates[task] = outputs
    print(f"Generated {len(outputs)} candidates for: {task[:40]}...")
"""),
        md("## Step 2 — Build the Judge"),
        code("""
class Judgment(BaseModel):
    relevance: int = Field(ge=1, le=5, description="How relevant to the task")
    quality: int = Field(ge=1, le=5, description="Overall quality")
    issues: list[str] = Field(description="Problems found")
    verdict: str = Field(description="pass or fail")

judge = ChatOllama(model="qwen3:8b", temperature=0.0).with_structured_output(Judgment)

all_judgments = []
for task, outputs in candidates.items():
    print(f"\\nJudging: {task[:40]}...")
    for i, output in enumerate(outputs):
        j = judge.invoke(f"Task: {task}\\n\\nOutput to judge:\\n{output[:500]}")
        j_dict = j.model_dump()
        j_dict["task"] = task[:40]
        j_dict["candidate"] = i + 1
        all_judgments.append(j_dict)
        print(f"  Candidate {i+1}: relevance={j.relevance} quality={j.quality} verdict={j.verdict}")
        if j.issues:
            print(f"    Issues: {j.issues}")
"""),
        md("## Step 3 — Aggregate Scores"),
        code("""
import pandas as pd

jdf = pd.DataFrame(all_judgments)
print("\\nJUDGMENT SUMMARY")
print("="*60)
print(jdf[["task","candidate","relevance","quality","verdict"]].to_string(index=False))

print(f"\\nPass rate: {(jdf['verdict']=='pass').mean()*100:.0f}%")
print(f"Average quality: {jdf['quality'].mean():.1f}/5")
"""),
        md("## What You Learned\n- **LLM-as-judge** pattern with structured output\n- **Multi-temperature generation** for diversity\n- **Quantitative quality assessment** of LLM outputs"),
    ]))

    # ── Project 63: Local RAG A/B Testing ───────────────────────────────
    paths.append(write_nb(7, "63_Local_RAG_AB_Testing", [
        md("# Project 63 — Local RAG A/B Testing\n## Compare Retrieval Strategies Side-by-Side\n\n**Stack:** LangChain · ChromaDB · Ollama · Jupyter"),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb rank-bm25"),
        md("## Step 1 — Build Test Corpus"),
        code("""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil

llm = ChatOllama(model="qwen3:8b", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

docs = [
    Document(page_content="Machine learning uses algorithms to learn patterns from data. "
             "Supervised learning requires labeled training data. Common algorithms include "
             "linear regression, decision trees, and neural networks.", metadata={"source": "ml_intro"}),
    Document(page_content="Deep learning is a subset of machine learning using neural networks "
             "with many layers. CNNs excel at image tasks. RNNs and transformers handle sequences. "
             "Attention mechanisms revolutionized NLP.", metadata={"source": "deep_learning"}),
    Document(page_content="Reinforcement learning trains agents through rewards and penalties. "
             "Q-learning and policy gradient are key approaches. Applications include game playing, "
             "robotics, and recommendation systems.", metadata={"source": "rl"}),
    Document(page_content="Transfer learning uses pre-trained models on new tasks. Fine-tuning "
             "adjusts weights for specific domains. Few-shot and zero-shot learning minimize data needs. "
             "Foundation models like GPT demonstrate emergent capabilities.", metadata={"source": "transfer"}),
    Document(page_content="MLOps covers the operational aspects of ML. It includes data versioning, "
             "experiment tracking, model serving, and monitoring. Tools include MLflow, DVC, and Kubeflow. "
             "CI/CD for ML automates model deployment.", metadata={"source": "mlops"}),
]
print(f"Corpus: {len(docs)} documents")
"""),
        md("## Step 2 — Strategy A: Small Chunks + Dense Retrieval"),
        code("""
splitter_a = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks_a = splitter_a.split_documents(docs)

shutil.rmtree("chroma_a", ignore_errors=True)
store_a = Chroma.from_documents(chunks_a, embeddings, persist_directory="chroma_a")
retriever_a = store_a.as_retriever(search_kwargs={"k": 3})
print(f"Strategy A: {len(chunks_a)} small chunks (100 chars)")
"""),
        md("## Step 3 — Strategy B: Large Chunks + Dense Retrieval"),
        code("""
splitter_b = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks_b = splitter_b.split_documents(docs)

shutil.rmtree("chroma_b", ignore_errors=True)
store_b = Chroma.from_documents(chunks_b, embeddings, persist_directory="chroma_b")
retriever_b = store_b.as_retriever(search_kwargs={"k": 3})
print(f"Strategy B: {len(chunks_b)} large chunks (300 chars)")
"""),
        md("## Step 4 — A/B Comparison"),
        code("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based ONLY on the context. If unsure, say so."),
    ("human", "Context: {context}\\n\\nQuestion: {question}")
])
chain = qa_prompt | llm | StrOutputParser()

test_questions = [
    "What is transfer learning?",
    "What tools are used in MLOps?",
    "How do CNNs differ from RNNs?",
]

results = []
for q in test_questions:
    for label, retriever in [("A-small", retriever_a), ("B-large", retriever_b)]:
        start = time.time()
        docs_found = retriever.invoke(q)
        context = "\\n".join([d.page_content for d in docs_found])
        answer = chain.invoke({"context": context, "question": q})
        elapsed = time.time() - start
        results.append({
            "strategy": label, "question": q[:30],
            "chunks_found": len(docs_found),
            "context_len": len(context),
            "answer_len": len(answer),
            "latency": round(elapsed, 2),
        })
        print(f"  {label} | {q[:30]} | {len(docs_found)} chunks | {elapsed:.1f}s")

import pandas as pd
rdf = pd.DataFrame(results)
print("\\nA/B COMPARISON")
print(rdf.groupby("strategy")[["context_len","answer_len","latency"]].mean().round(2).to_string())
"""),
        md("## What You Learned\n- **A/B testing** for retrieval strategies\n- **Chunk size impact** on context and answer quality\n- **Quantitative comparison** framework for RAG"),
    ]))

    # ── Project 64: Local Tool Selection Benchmark ──────────────────────
    paths.append(write_nb(7, "64_Local_Tool_Selection_Benchmark", [
        md("# Project 64 — Local Tool Selection Benchmark\n## Evaluate LLM Ability to Choose the Right Tool\n\n**Stack:** LangChain · Ollama · Jupyter"),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Define Tool Registry"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

tool_registry = [
    {"name": "calculator", "description": "Perform math calculations", "example": "What is 15% of 230?"},
    {"name": "web_search", "description": "Search the internet for information", "example": "Latest news on AI"},
    {"name": "file_reader", "description": "Read contents of a file", "example": "Show me the config file"},
    {"name": "code_runner", "description": "Execute Python code", "example": "Run this script"},
    {"name": "database_query", "description": "Query a SQL database", "example": "How many users signed up?"},
    {"name": "email_sender", "description": "Send an email", "example": "Email the report to the team"},
]

# Ground truth test cases
test_cases = [
    ("What's the square root of 144?", "calculator"),
    ("Find recent papers on transformers", "web_search"),
    ("Show the contents of README.md", "file_reader"),
    ("Execute the data processing script", "code_runner"),
    ("How many orders were placed last month?", "database_query"),
    ("Send the weekly report to the team", "email_sender"),
    ("Calculate the compound interest on $1000 at 5% for 3 years", "calculator"),
    ("What files are in the /data directory?", "file_reader"),
]
print(f"Tool registry: {len(tool_registry)} tools, {len(test_cases)} test cases")
"""),
        md("## Step 2 — Run Benchmark"),
        code("""
class ToolSelection(BaseModel):
    selected_tool: str = Field(description="Name of the tool to use")
    confidence: float = Field(ge=0, le=1, description="Confidence 0-1")
    reasoning: str

selector = llm.with_structured_output(ToolSelection)

tool_descriptions = "\\n".join([f"- {t['name']}: {t['description']}" for t in tool_registry])

correct = 0
results = []
for query, expected in test_cases:
    selection = selector.invoke(
        f"Available tools:\\n{tool_descriptions}\\n\\nUser request: {query}\\n\\nSelect the best tool."
    )
    is_correct = selection.selected_tool == expected
    correct += is_correct
    results.append({
        "query": query[:40], "expected": expected,
        "selected": selection.selected_tool,
        "correct": "✓" if is_correct else "✗",
        "confidence": selection.confidence,
    })
    print(f"  {'✓' if is_correct else '✗'} {query[:40]:<42} expected={expected:<15} got={selection.selected_tool}")

print(f"\\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
"""),
        md("## What You Learned\n- **Tool selection benchmarking** with ground truth\n- **Structured output** for reliable tool routing\n- **Accuracy measurement** for agent capabilities"),
    ]))

    # ── Project 65: Local Hallucination Audit ───────────────────────────
    paths.append(write_nb(7, "65_Local_Hallucination_Audit", [
        md("# Project 65 — Local Hallucination Audit\n## Claim Extraction → Source Verification → Scoring\n\n**Stack:** LangChain · Ollama · Jupyter"),
        code("# !pip install -q langchain langchain-ollama"),
        md("## Step 1 — Setup"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

# Source documents (ground truth)
sources = {
    "company_report": "Acme Corp revenue was $2.3M in Q3 2024, up 15% from Q2. "
        "The company has 150 employees across 3 offices. CEO is Jane Smith.",
    "product_specs": "The Widget X weighs 2.5 kg, costs $49.99, and comes in blue and red. "
        "Battery life is 8 hours. It was launched in March 2024.",
}

# LLM-generated answers to audit
answers_to_audit = [
    {"question": "Tell me about Acme Corp",
     "answer": "Acme Corp made $2.3M in Q3 2024, a 20% increase. They have 200 employees "
               "and their CEO is Jane Smith. They're headquartered in San Francisco.",
     "source_key": "company_report"},
    {"question": "What are Widget X specs?",
     "answer": "Widget X weighs 2.5 kg and costs $49.99. It comes in blue, red, and green. "
               "Battery lasts 8 hours. It launched in March 2024 and has sold 1M units.",
     "source_key": "product_specs"},
]
print(f"Auditing {len(answers_to_audit)} answers against {len(sources)} sources")
"""),
        md("## Step 2 — Extract and Verify Claims"),
        code("""
class ClaimVerification(BaseModel):
    claim: str
    supported: str = Field(description="supported, contradicted, or unverifiable")
    evidence: str = Field(description="Quote from source or 'not found'")
    explanation: str

class AuditResult(BaseModel):
    claims: list[ClaimVerification]
    hallucination_score: float = Field(description="0.0=no hallucination, 1.0=all hallucinated")

auditor = llm.with_structured_output(AuditResult)

for item in answers_to_audit:
    source = sources[item["source_key"]]
    result = auditor.invoke(
        f"SOURCE (ground truth):\\n{source}\\n\\n"
        f"ANSWER TO AUDIT:\\n{item['answer']}\\n\\n"
        f"Extract each factual claim from the answer. For each claim, check if it's "
        f"supported, contradicted, or unverifiable based on the source."
    )

    print(f"\\nQ: {item['question']}")
    print(f"Hallucination score: {result.hallucination_score:.0%}")
    for c in result.claims:
        icon = {"supported":"✓","contradicted":"✗","unverifiable":"?"}[c.supported]
        print(f"  {icon} [{c.supported}] {c.claim}")
        if c.supported != "supported":
            print(f"    Evidence: {c.evidence}")
"""),
        md("## What You Learned\n- **Claim extraction** from generated text\n- **Source verification** against ground truth\n- **Hallucination scoring** for RAG quality assurance"),
    ]))

    # ── Projects 66-70: Template-based eval projects ────────────────────
    eval_projects = [
        (66, "66_Local_Groundedness_Checker", "Groundedness Checker",
         "Score how well answers are grounded in retrieved context",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

class GroundednessScore(BaseModel):
    score: float = Field(ge=0, le=1, description="0=not grounded, 1=fully grounded")
    grounded_sentences: list[str]
    ungrounded_sentences: list[str]
    explanation: str

scorer = llm.with_structured_output(GroundednessScore)

test_pairs = [
    {"context": "Python was created by Guido van Rossum in 1991. It emphasizes code readability.",
     "answer": "Python was created by Guido van Rossum in 1991. It's the most popular language and emphasizes readability."},
    {"context": "Docker containers share the host OS kernel. They're lighter than VMs.",
     "answer": "Docker uses containers that share the OS kernel, making them lighter than VMs. Docker was created in 2013 by Solomon Hykes."},
]

for pair in test_pairs:
    result = scorer.invoke(
        f"Context:\\n{pair['context']}\\n\\nAnswer:\\n{pair['answer']}\\n\\n"
        f"Score how well the answer is grounded in the context."
    )
    print(f"Score: {result.score:.0%}")
    print(f"  Grounded: {result.grounded_sentences}")
    print(f"  Ungrounded: {result.ungrounded_sentences}")
    print()
"""),

        (67, "67_Local_Structured_Output_Test", "Structured Output Reliability Test",
         "Test JSON schema adherence across different output structures",
         """
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

# Define test schemas
class SimpleOutput(BaseModel):
    name: str
    age: int
    city: str

class NestedOutput(BaseModel):
    title: str
    tags: list[str]
    metadata: dict[str, str]

class ComplexOutput(BaseModel):
    analysis: str
    scores: list[float]
    recommendations: list[str]
    confidence: float = Field(ge=0, le=1)

schemas = [
    ("simple", SimpleOutput, "Extract: John Doe, 30, lives in NYC"),
    ("nested", NestedOutput, "Create a blog post entry about machine learning with tags"),
    ("complex", ComplexOutput, "Analyze the performance of a sales team last quarter"),
]

results = []
for name, schema, prompt in schemas:
    structured = llm.with_structured_output(schema)
    try:
        output = structured.invoke(prompt)
        valid = True
        result_dict = output.model_dump()
    except Exception as e:
        valid = False
        result_dict = {"error": str(e)}

    results.append({"schema": name, "valid": valid, "output": result_dict})
    print(f"{name}: {'✓ PASS' if valid else '✗ FAIL'}")
    print(f"  Output: {json.dumps(result_dict, indent=2, default=str)[:200]}")
    print()

pass_rate = sum(1 for r in results if r["valid"]) / len(results)
print(f"Pass rate: {pass_rate:.0%}")
"""),

        (68, "68_Local_Cost_Latency_Benchmark", "Cost & Latency Notebook",
         "Compare model speed and quality for different task types",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time, pandas as pd

# Test with different temperature settings (simulating different "models")
configs = [
    {"name": "precise", "temperature": 0.0},
    {"name": "balanced", "temperature": 0.3},
    {"name": "creative", "temperature": 0.8},
]

tasks = [
    ("classification", "Classify this as positive or negative: 'Great product, love it!'"),
    ("summarization", "Summarize: Machine learning is a branch of AI that uses data to learn patterns."),
    ("generation", "Write a haiku about programming."),
    ("extraction", "Extract the name and date: 'Meeting with Alice on January 15th.'"),
]

results = []
for config in configs:
    llm = ChatOllama(model="qwen3:8b", temperature=config["temperature"])
    chain = ChatPromptTemplate.from_template("{task}") | llm | StrOutputParser()
    for task_type, prompt in tasks:
        start = time.time()
        output = chain.invoke({"task": prompt})
        elapsed = time.time() - start
        results.append({
            "config": config["name"],
            "task": task_type,
            "latency_s": round(elapsed, 2),
            "output_tokens": len(output.split()),
            "output_len": len(output),
        })

df = pd.DataFrame(results)
pivot = df.pivot_table(index="task", columns="config", values="latency_s", aggfunc="mean")
print("LATENCY BY TASK × CONFIG (seconds)")
print(pivot.round(2).to_string())
print(f"\\nOverall avg latency: {df['latency_s'].mean():.2f}s")
"""),

        (69, "69_Local_Memory_Strategy_Benchmark", "Memory Strategy Benchmark",
         "Compare buffer, summary, and vector memory approaches",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

# Simulate different memory strategies
conversation = [
    ("user", "Hi, I'm Alice and I work at TechCorp as a data scientist."),
    ("assistant", "Hello Alice! Nice to meet you. Data science at TechCorp sounds exciting."),
    ("user", "I'm working on a churn prediction model using XGBoost."),
    ("assistant", "XGBoost is great for churn prediction. What features are you using?"),
    ("user", "Usage frequency, last login, and support ticket count."),
    ("user", "What model am I working on and what's my name?"),
]

# Strategy 1: Full buffer (all messages)
def buffer_memory(messages):
    context = "\\n".join([f"{role}: {msg}" for role, msg in messages[:-1]])
    return context

# Strategy 2: Summary memory
def summary_memory(messages, llm):
    full = "\\n".join([f"{role}: {msg}" for role, msg in messages[:-1]])
    chain = ChatPromptTemplate.from_template(
        "Summarize this conversation in 2 sentences: {conv}"
    ) | llm | StrOutputParser()
    return chain.invoke({"conv": full})

# Strategy 3: Recent window only (last 2 exchanges)
def window_memory(messages, window=4):
    recent = messages[max(0, len(messages)-window-1):-1]
    return "\\n".join([f"{role}: {msg}" for role, msg in recent])

strategies = {
    "full_buffer": buffer_memory(conversation),
    "summary": summary_memory(conversation, llm),
    "window_2": window_memory(conversation, 4),
}

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the conversation context."),
    ("human", "Context: {context}\\n\\nQuestion: {question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

question = conversation[-1][1]
print(f"Test question: {question}\\n")

for name, context in strategies.items():
    start = time.time()
    answer = qa_chain.invoke({"context": context, "question": question})
    elapsed = time.time() - start
    print(f"[{name}] ({elapsed:.1f}s)")
    print(f"  Context length: {len(context)} chars")
    print(f"  Answer: {answer[:200]}")
    print()
"""),

        (70, "70_Local_Agent_Trace_Analyzer", "Agent Trace Analyzer",
         "Log and inspect agent execution traces for debugging",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, time
from datetime import datetime

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# Trace logger
class TraceLogger:
    def __init__(self):
        self.traces = []
        self.current_trace = None

    def start_trace(self, task):
        self.current_trace = {
            "task": task, "start": time.time(),
            "steps": [], "errors": [], "status": "running"
        }

    def log_step(self, step_name, input_data, output_data, duration):
        self.current_trace["steps"].append({
            "name": step_name, "input_len": len(str(input_data)),
            "output_len": len(str(output_data)), "duration_s": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
        })

    def log_error(self, error):
        self.current_trace["errors"].append(str(error))

    def end_trace(self, status="success"):
        self.current_trace["status"] = status
        self.current_trace["total_duration"] = round(time.time() - self.current_trace["start"], 3)
        self.traces.append(self.current_trace)
        self.current_trace = None

    def summary(self):
        return {
            "total_traces": len(self.traces),
            "success": sum(1 for t in self.traces if t["status"] == "success"),
            "failed": sum(1 for t in self.traces if t["status"] == "failed"),
            "avg_duration": sum(t["total_duration"] for t in self.traces) / max(len(self.traces), 1),
            "avg_steps": sum(len(t["steps"]) for t in self.traces) / max(len(self.traces), 1),
        }

logger = TraceLogger()

# Run traced tasks
tasks = [
    "Summarize the benefits of Python",
    "List 3 sorting algorithms",
    "Explain what an API is",
]

chain = ChatPromptTemplate.from_template("{task}") | llm | StrOutputParser()

for task in tasks:
    logger.start_trace(task)
    try:
        start = time.time()
        result = chain.invoke({"task": task})
        logger.log_step("llm_call", task, result, time.time() - start)
        logger.end_trace("success")
    except Exception as e:
        logger.log_error(e)
        logger.end_trace("failed")

# Analyze traces
summary = logger.summary()
print("TRACE ANALYSIS")
print("="*50)
print(json.dumps(summary, indent=2))

print("\\nDetailed traces:")
for t in logger.traces:
    print(f"  [{t['status']}] {t['task'][:40]} — {t['total_duration']:.2f}s, {len(t['steps'])} steps")
"""),
    ]

    for proj_num, folder, title, desc, main_code in eval_projects:
        paths.append(write_nb(7, folder, [
            md(f"# Project {proj_num} — {title}\n## {desc}\n\n**Stack:** LangChain · Ollama · Jupyter"),
            code("# !pip install -q langchain langchain-ollama pandas pydantic"),
            md("## Implementation"),
            code(main_code),
            md(f"## What You Learned\n- **{title}** — {desc.lower()}\n- **Local evaluation** patterns for LLM applications\n- **Quantitative metrics** for quality assurance"),
        ]))

    print(f"Group 7 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
