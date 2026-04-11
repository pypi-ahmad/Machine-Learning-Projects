"""Group 8 — Projects 71-80: Fine-Tuning-Adjacent Learning Projects."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 71: Fine-Tuning Dataset Builder ─────────────────────────
    paths.append(write_nb(8, "71_Fine_Tuning_Dataset_Builder", [
        md("# Project 71 — Fine-Tuning Dataset Builder\n## Generate Instruction–Response Training Pairs\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Define Domain & Seed Examples"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd, json

llm = ChatOllama(model="qwen3:8b", temperature=0.7)

domain = "customer support for a SaaS product"

seed_examples = [
    {"instruction": "How do I reset my password?",
     "response": "Go to Settings > Security > Reset Password. You'll receive a confirmation email."},
    {"instruction": "Can I upgrade my plan mid-billing cycle?",
     "response": "Yes! Go to Billing > Change Plan. You'll be prorated for the remaining days."},
    {"instruction": "Why is my dashboard loading slowly?",
     "response": "Try clearing your browser cache. If it persists, check our status page for outages."},
]
print(f"Domain: {domain}")
print(f"Seed examples: {len(seed_examples)}")
"""),
        md("## Step 2 — Generate New Training Pairs"),
        code("""
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", f\"\"\"You are generating training data for a {domain} chatbot.

Given seed examples, generate 5 NEW diverse instruction-response pairs.
Cover different topics: billing, technical issues, features, account management.

Return a JSON array of objects with "instruction" and "response" keys.
Return ONLY valid JSON, no extra text.\"\"\"),
    ("human", "Seed examples:\\n{seeds}\\n\\nGenerate 5 new pairs:")
])
gen_chain = gen_prompt | llm | StrOutputParser()

seeds_text = json.dumps(seed_examples, indent=2)
raw = gen_chain.invoke({"seeds": seeds_text})

# Parse generated pairs
start = raw.find("[")
end = raw.rfind("]") + 1
if start >= 0 and end > start:
    generated = json.loads(raw[start:end])
else:
    generated = []
    print("Warning: Could not parse JSON, using fallback")

all_pairs = seed_examples + generated
print(f"Generated {len(generated)} new pairs (total: {len(all_pairs)})")
for pair in generated[:3]:
    print(f"  Q: {pair['instruction'][:60]}")
    print(f"  A: {pair['response'][:60]}...")
    print()
"""),
        md("## Step 3 — Quality Check & Format"),
        code("""
quality_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Rate this training pair on a 1-5 scale for:
- relevance: Is it on-topic for customer support?
- clarity: Is the response clear and helpful?
- correctness: Is the response accurate?

Return JSON: {{"relevance": N, "clarity": N, "correctness": N, "issues": "..."}}\"\"\"),
    ("human", "Instruction: {instruction}\\nResponse: {response}")
])
quality_chain = quality_prompt | llm | StrOutputParser()

checked_pairs = []
for pair in all_pairs:
    try:
        raw = quality_chain.invoke(pair)
        s = raw.find("{"); e = raw.rfind("}") + 1
        scores = json.loads(raw[s:e]) if s >= 0 else {"relevance":3,"clarity":3,"correctness":3}
    except Exception:
        scores = {"relevance": 3, "clarity": 3, "correctness": 3}

    pair["quality_score"] = (scores.get("relevance",3) + scores.get("clarity",3) + scores.get("correctness",3)) / 3
    checked_pairs.append(pair)

df = pd.DataFrame(checked_pairs)
print(f"Quality scores:")
print(f"  Mean: {df['quality_score'].mean():.2f}")
print(f"  Min:  {df['quality_score'].min():.2f}")
print(f"  Max:  {df['quality_score'].max():.2f}")
"""),
        md("## Step 4 — Export in Multiple Formats"),
        code("""
from pathlib import Path

Path("sample_data").mkdir(exist_ok=True)

# Alpaca format
alpaca = [{"instruction": p["instruction"], "input": "", "output": p["response"]} for p in checked_pairs]
with open("sample_data/training_alpaca.json", "w") as f:
    json.dump(alpaca, f, indent=2)

# ShareGPT format
sharegpt = [{"conversations": [
    {"from": "human", "value": p["instruction"]},
    {"from": "gpt", "value": p["response"]}
]} for p in checked_pairs]
with open("sample_data/training_sharegpt.json", "w") as f:
    json.dump(sharegpt, f, indent=2)

# JSONL format
with open("sample_data/training.jsonl", "w") as f:
    for p in checked_pairs:
        json.dump({"prompt": p["instruction"], "completion": p["response"]}, f)
        f.write("\\n")

print(f"Exported {len(checked_pairs)} pairs in 3 formats:")
print("  ✓ sample_data/training_alpaca.json")
print("  ✓ sample_data/training_sharegpt.json")
print("  ✓ sample_data/training.jsonl")
"""),
        md("## What You Learned\n- **Seed-based data augmentation** for training pairs\n- **Quality scoring** with LLM-as-judge\n- **Export formats:** Alpaca, ShareGPT, JSONL"),
    ]))

    # ── Project 72: Synthetic Data Generator ────────────────────────────
    paths.append(write_nb(8, "72_Synthetic_Data_Generator", [
        md("# Project 72 — Synthetic Data Generator for Classification\n## Generate Labeled Examples for Text Classification\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Define Classification Task"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd, json

llm = ChatOllama(model="qwen3:8b", temperature=0.8)

task = "email intent classification"
labels = ["inquiry", "complaint", "feedback", "request", "spam"]

print(f"Task: {task}")
print(f"Labels: {labels}")
"""),
        md("## Step 2 — Generate Synthetic Examples Per Label"),
        code("""
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Generate 5 realistic email subject+body examples for the label: {label}

Context: Corporate email classification system.
Make each example distinct in topic and writing style.

Return a JSON array: [{{"subject": "...", "body": "...", "label": "{label}"}}]\"\"\"),
    ("human", "Generate 5 {label} emails:")
])
gen_chain = gen_prompt | llm | StrOutputParser()

all_examples = []
for label in labels:
    raw = gen_chain.invoke({"label": label})
    try:
        s = raw.find("["); e = raw.rfind("]") + 1
        examples = json.loads(raw[s:e]) if s >= 0 else []
    except Exception:
        examples = [{"subject": f"Example {label}", "body": f"Sample {label} email", "label": label}]
    all_examples.extend(examples)
    print(f"  {label}: generated {len(examples)} examples")

df = pd.DataFrame(all_examples)
print(f"\\nTotal: {len(df)} examples across {df['label'].nunique()} labels")
print(df['label'].value_counts().to_string())
"""),
        md("## Step 3 — Validate with Cross-Check"),
        code("""
verify_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify this email into one of: {labels}. Return ONLY the label."),
    ("human", "Subject: {subject}\\nBody: {body}")
])
verify_chain = verify_prompt | llm | StrOutputParser()

correct = 0
for _, row in df.iterrows():
    predicted = verify_chain.invoke({
        "labels": ", ".join(labels),
        "subject": row.get("subject", ""),
        "body": row.get("body", ""),
    }).strip().lower()
    if row["label"].lower() in predicted:
        correct += 1

accuracy = correct / len(df) if len(df) > 0 else 0
print(f"Cross-validation accuracy: {accuracy:.0%}")
print(f"Consistent examples: {correct}/{len(df)}")

# Save
df.to_csv("sample_data/synthetic_classification.csv", index=False)
print("\\n✓ Saved to sample_data/synthetic_classification.csv")
"""),
        md("## What You Learned\n- **Synthetic data generation** with label-conditioned prompts\n- **Cross-validation** using LLM to verify label consistency\n- **Dataset export** for downstream classification training"),
    ]))

    # ── Projects 73-80: Template-based fine-tuning projects ─────────────
    ft_projects = [
        (73, "73_Prompt_vs_FineTune_Comparison", "Prompt vs Fine-Tune Comparison Lab",
         "Compare prompt engineering against simulated fine-tuning performance",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd, time

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# Task: Sentiment classification
test_cases = [
    ("I absolutely love this product!", "positive"),
    ("Terrible experience, never again.", "negative"),
    ("It's okay, nothing special.", "neutral"),
    ("Best purchase I've made this year!", "positive"),
    ("The quality has really gone downhill.", "negative"),
    ("It works as expected.", "neutral"),
    ("Wow, exceeded all my expectations!", "positive"),
    ("Worst customer service ever.", "negative"),
]

# Strategy 1: Zero-shot
zero_shot = ChatPromptTemplate.from_template(
    "Classify sentiment as positive/negative/neutral: {text}\\nLabel:"
) | llm | StrOutputParser()

# Strategy 2: Few-shot
few_shot = ChatPromptTemplate.from_template(
    \"\"\"Classify sentiment. Examples:
"Great product!" → positive
"Horrible quality" → negative  
"It's fine" → neutral

Text: {text}
Label:\"\"\"
) | llm | StrOutputParser()

# Strategy 3: Detailed instruction (simulating fine-tune style)
detailed = ChatPromptTemplate.from_template(
    \"\"\"You are a sentiment classifier trained on product reviews.
Rules: 
- "positive" = satisfaction, praise, excitement
- "negative" = dissatisfaction, complaint, frustration
- "neutral" = factual statements, indifference

Respond with ONLY the label.
Text: {text}
Label:\"\"\"
) | llm | StrOutputParser()

strategies = {"zero_shot": zero_shot, "few_shot": few_shot, "detailed": detailed}

results = []
for name, chain in strategies.items():
    correct = 0
    start = time.time()
    for text, expected in test_cases:
        predicted = chain.invoke({"text": text}).strip().lower()
        match = expected in predicted
        correct += match
    elapsed = time.time() - start
    results.append({
        "strategy": name,
        "accuracy": f"{correct}/{len(test_cases)}",
        "pct": round(correct/len(test_cases)*100),
        "latency": round(elapsed, 2),
    })

rdf = pd.DataFrame(results)
print("STRATEGY COMPARISON")
print("="*50)
print(rdf.to_string(index=False))
print(f"\\nBest strategy: {rdf.loc[rdf['pct'].idxmax(), 'strategy']}")
"""),

        (74, "74_Style_Dataset_Creator", "Style Dataset Creator",
         "Extract and replicate writing style for training data",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.7)

# Sample texts in different styles
style_samples = {
    "technical": "The system utilizes a microservices architecture with gRPC communication. "
                 "Each service maintains its own PostgreSQL instance, ensuring data isolation.",
    "casual": "So basically we split everything into tiny services that talk to each other. "
              "Each one gets its own database so nothing gets mixed up. Pretty clean setup!",
    "executive": "Our platform leverages a modern distributed architecture to ensure scalability "
                 "and reliability. This strategic approach reduces operational risk by 40%.",
}

# Extract style characteristics
analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the writing style. Return JSON with: tone, vocabulary_level, "
     "sentence_length, formality (1-5), distinguishing_features"),
    ("human", "Text: {text}")
])
analyze_chain = analyze_prompt | llm | StrOutputParser()

style_profiles = {}
for name, sample in style_samples.items():
    raw = analyze_chain.invoke({"text": sample})
    try:
        s = raw.find("{"); e = raw.rfind("}") + 1
        profile = json.loads(raw[s:e])
    except Exception:
        profile = {"tone": name, "formality": 3}
    style_profiles[name] = profile
    print(f"\\n{name} style:")
    print(f"  {json.dumps(profile, indent=2)[:200]}")

# Generate style-matched training data
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write 3 sentences about cloud computing in the {style} style. "
     "Style characteristics: {profile}"),
    ("human", "Generate text in {style} style:")
])
gen_chain = gen_prompt | llm | StrOutputParser()

dataset = []
for style, profile in style_profiles.items():
    text = gen_chain.invoke({"style": style, "profile": json.dumps(profile)})
    dataset.append({"style": style, "text": text, "profile": profile})
    print(f"\\n{style} generated: {text[:100]}...")

print(f"\\n✓ Generated {len(dataset)} style-matched samples")
"""),

        (75, "75_Instruction_Quality_Checker", "Instruction Dataset Quality Checker",
         "Detect duplicates, contradictions, and low-quality training pairs",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

# Sample instruction dataset (with intentional issues)
dataset = [
    {"instruction": "How do I reset my password?",
     "response": "Go to Settings > Reset Password."},
    {"instruction": "How can I change my password?",
     "response": "Navigate to Account > Security > Change Password."},  # Near-duplicate
    {"instruction": "What are your business hours?",
     "response": "We're open Monday-Friday, 9am-5pm EST."},
    {"instruction": "When is customer support available?",
     "response": "Our support team is available 24/7."},  # Contradicts above
    {"instruction": "Delete my account",
     "response": "ok"},  # Low quality
    {"instruction": "How do I export my data?",
     "response": "Go to Settings > Data > Export. Choose your format (CSV/JSON) and click Download."},
]

class QualityIssue(BaseModel):
    pair_index: int
    issue_type: str = Field(description="duplicate, contradiction, low_quality, ambiguous")
    severity: str = Field(description="low, medium, high")
    description: str
    suggested_fix: str

class QualityReport(BaseModel):
    issues: list[QualityIssue]
    overall_score: float = Field(ge=0, le=1)

checker = llm.with_structured_output(QualityReport)

report = checker.invoke(
    f"Analyze this instruction dataset for quality issues "
    f"(duplicates, contradictions, low-quality responses):\\n\\n"
    f"{json.dumps(dataset, indent=2)}"
)

print("QUALITY AUDIT REPORT")
print("="*50)
print(f"Overall score: {report.overall_score:.0%}")
print(f"Issues found: {len(report.issues)}\\n")
for issue in report.issues:
    print(f"  [{issue.severity.upper()}] Pair #{issue.pair_index}: {issue.issue_type}")
    print(f"    {issue.description}")
    print(f"    Fix: {issue.suggested_fix}")
    print()
"""),

        (76, "76_Local_Distillation_Lab", "Local Distillation Lab",
         "Generate teacher-student training data for model compression",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, time

# Teacher: verbose, high-quality answers
teacher = ChatOllama(model="qwen3:8b", temperature=0.3)
# Student target: concise, efficient answers

prompts = [
    "Explain how a hash table works",
    "What is the difference between TCP and UDP?",
    "How does garbage collection work in programming?",
    "What is a deadlock in concurrent programming?",
    "Explain the CAP theorem",
]

teacher_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher. Give thorough, detailed explanations with examples."),
    ("human", "{question}")
])
teacher_chain = teacher_prompt | teacher | StrOutputParser()

compress_prompt = ChatPromptTemplate.from_messages([
    ("system", "Compress this teacher explanation into a concise student-friendly version. "
     "Keep all key facts but reduce length by 70%. No examples, just core concepts."),
    ("human", "Teacher explanation:\\n{teacher_output}")
])
compress_chain = compress_prompt | teacher | StrOutputParser()

distillation_data = []
for q in prompts:
    t_start = time.time()
    teacher_output = teacher_chain.invoke({"question": q})
    student_output = compress_chain.invoke({"teacher_output": teacher_output})
    elapsed = time.time() - t_start

    distillation_data.append({
        "instruction": q,
        "teacher_response": teacher_output,
        "student_response": student_output,
        "compression_ratio": round(len(student_output) / max(len(teacher_output), 1), 2),
    })
    print(f"  {q[:40]}... | teacher={len(teacher_output)} chars → student={len(student_output)} chars | {elapsed:.1f}s")

avg_compression = sum(d["compression_ratio"] for d in distillation_data) / len(distillation_data)
print(f"\\nAvg compression ratio: {avg_compression:.0%}")
print(f"Total distillation pairs: {len(distillation_data)}")

with open("sample_data/distillation_data.json", "w") as f:
    json.dump(distillation_data, f, indent=2)
print("✓ Saved to sample_data/distillation_data.json")
"""),

        (77, "77_Preference_Pair_Builder", "Preference Pair Builder",
         "Create chosen/rejected pairs for RLHF-style training",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.7)
judge = ChatOllama(model="qwen3:8b", temperature=0.0)

prompts = [
    "Write a product description for wireless earbuds",
    "Explain recursion to a beginner",
    "Draft a LinkedIn post about starting a new job",
    "Write a bug report for a login failure",
]

# Generate pairs with different temperatures
pairs = []
for prompt in prompts:
    outputs = []
    for temp in [0.1, 0.5, 0.9]:
        gen = ChatOllama(model="qwen3:8b", temperature=temp)
        chain = ChatPromptTemplate.from_template("{p}") | gen | StrOutputParser()
        outputs.append(chain.invoke({"p": prompt}))

    # Judge ranks them
    class Ranking(BaseModel):
        best_index: int = Field(description="Index of best response (0, 1, or 2)")
        worst_index: int = Field(description="Index of worst response (0, 1, or 2)")
        reasoning: str

    ranking_judge = judge.with_structured_output(Ranking)
    ranking = ranking_judge.invoke(
        f"Prompt: {prompt}\\n\\n"
        + "\\n\\n".join([f"Response {i}: {o[:200]}" for i, o in enumerate(outputs)])
        + "\\n\\nRank the responses by quality."
    )

    pairs.append({
        "prompt": prompt,
        "chosen": outputs[ranking.best_index][:300],
        "rejected": outputs[ranking.worst_index][:300],
        "reasoning": ranking.reasoning,
    })
    print(f"  {prompt[:40]}... best={ranking.best_index}, worst={ranking.worst_index}")

with open("sample_data/preference_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)
print(f"\\n✓ Generated {len(pairs)} preference pairs")
print("  Saved to sample_data/preference_pairs.json")
"""),

        (78, "78_JSON_Extraction_Dataset_Builder", "JSON Extraction Dataset Builder",
         "Build training data for document-to-JSON extraction",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.7)

# Source documents
documents = [
    "John Smith, Senior Engineer at Google, graduated from MIT in 2015. Email: john@example.com",
    "Invoice #INV-2024-001, Date: Jan 15 2024, Amount: $1,250.00, Vendor: Acme Corp, Due: Feb 15",
    "Meeting: Q1 Planning, Attendees: Alice, Bob, Carol. Date: Monday 10am. Room: 301-B",
    "Product: Widget Pro X, SKU: WPX-100, Price: $99.99, Weight: 0.5 kg, Color: Blue",
]

schemas = [
    {"name": "str", "role": "str", "company": "str", "education": "str", "email": "str"},
    {"invoice_id": "str", "date": "str", "amount": "float", "vendor": "str", "due_date": "str"},
    {"title": "str", "attendees": "list[str]", "date": "str", "room": "str"},
    {"product": "str", "sku": "str", "price": "float", "weight": "str", "color": "str"},
]

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured data from the text. Return valid JSON matching the schema."),
    ("human", "Schema: {schema}\\nText: {text}\\n\\nExtracted JSON:")
])
chain = extract_prompt | llm | StrOutputParser()

dataset = []
for doc, schema in zip(documents, schemas):
    raw = chain.invoke({"schema": json.dumps(schema), "text": doc})
    try:
        s = raw.find("{"); e = raw.rfind("}") + 1
        extracted = json.loads(raw[s:e]) if s >= 0 else {}
    except Exception:
        extracted = {"error": "parse failed"}

    dataset.append({
        "input": doc,
        "schema": schema,
        "output": extracted,
    })
    print(f"  Doc: {doc[:50]}...")
    print(f"  Extracted: {json.dumps(extracted)[:100]}")
    print()

with open("sample_data/extraction_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
print(f"✓ Built {len(dataset)} extraction training pairs")
"""),

        (79, "79_Classification_FineTune_Readiness", "Classification Fine-Tune Readiness Audit",
         "Check dataset quality before investing in fine-tuning",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

# Sample classification dataset to audit
dataset = [
    {"text": "Server is down, critical issue", "label": "urgent"},
    {"text": "When do you close today?", "label": "inquiry"},
    {"text": "URGENT: system failure!!!", "label": "urgent"},
    {"text": "I want to cancel my subscription", "label": "churn"},
    {"text": "What time do you open?", "label": "inquiry"},
    {"text": "The app crashed again", "label": "bug"},
    {"text": "Love your new feature!", "label": "feedback"},
    {"text": "Everything is broken", "label": "urgent"},  # or bug?
    {"text": "Nice update", "label": "feedback"},
    {"text": "Cancel immediately", "label": "churn"},
]

class ReadinessReport(BaseModel):
    total_samples: int
    unique_labels: list[str]
    min_samples_per_label: int
    label_balance_score: float = Field(description="0=imbalanced, 1=perfectly balanced")
    ambiguous_samples: list[int] = Field(description="Indices of ambiguous samples")
    duplicate_risk: list[int] = Field(description="Indices of near-duplicate samples")
    readiness: str = Field(description="ready, needs_work, not_ready")
    recommendations: list[str]

auditor = llm.with_structured_output(ReadinessReport)

report = auditor.invoke(
    f"Audit this classification dataset for fine-tuning readiness:\\n\\n"
    f"{json.dumps(dataset, indent=2)}\\n\\n"
    f"Check for: class imbalance, ambiguous labels, duplicates, insufficient data."
)

print("FINE-TUNE READINESS AUDIT")
print("="*50)
print(f"Total samples:    {report.total_samples}")
print(f"Labels:           {report.unique_labels}")
print(f"Min per label:    {report.min_samples_per_label}")
print(f"Balance score:    {report.label_balance_score:.0%}")
print(f"Ambiguous:        indices {report.ambiguous_samples}")
print(f"Duplicate risk:   indices {report.duplicate_risk}")
print(f"Readiness:        {report.readiness.upper()}")
print(f"\\nRecommendations:")
for r in report.recommendations:
    print(f"  → {r}")
"""),

        (80, "80_Local_FineTuning_Evals_Harness", "Local Fine-Tuning Evals Harness",
         "Build an evaluation framework for fine-tuning experiments",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd, time, json

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# Evaluation test suite
eval_suite = {
    "classification": [
        {"input": "This product is amazing!", "expected": "positive"},
        {"input": "Terrible, waste of money.", "expected": "negative"},
        {"input": "It's decent, could be better.", "expected": "neutral"},
    ],
    "extraction": [
        {"input": "Meeting with Alice on Jan 15 at 3pm", "expected_fields": ["person", "date", "time"]},
        {"input": "Order #123 for $99.99 from Acme Corp", "expected_fields": ["order_id", "amount", "vendor"]},
    ],
    "summarization": [
        {"input": "Machine learning uses data to train models. Models learn patterns. "
                  "They can then make predictions on new data.",
         "expected_contains": ["machine learning", "data", "predictions"]},
    ],
}

# Run evaluations
results = []
for task, cases in eval_suite.items():
    for case in cases:
        start = time.time()
        prompt = ChatPromptTemplate.from_template(f"{{task}}: {{input}}")
        chain = prompt | llm | StrOutputParser()
        output = chain.invoke({"task": task.replace("_"," ").title(), "input": case["input"]})
        elapsed = time.time() - start

        # Score based on task type
        if task == "classification":
            score = 1.0 if case["expected"] in output.lower() else 0.0
        elif task == "extraction":
            found = sum(1 for f in case["expected_fields"] if f.lower() in output.lower())
            score = found / len(case["expected_fields"])
        elif task == "summarization":
            found = sum(1 for w in case["expected_contains"] if w.lower() in output.lower())
            score = found / len(case["expected_contains"])
        else:
            score = 0.5

        results.append({
            "task": task, "input": case["input"][:40],
            "score": round(score, 2), "latency": round(elapsed, 2),
            "output_len": len(output)
        })

rdf = pd.DataFrame(results)
print("EVALUATION RESULTS")
print("="*50)
print(rdf.to_string(index=False))
print(f"\\nOverall score: {rdf['score'].mean():.0%}")
print(f"By task:")
print(rdf.groupby("task")["score"].mean().round(2).to_string())
"""),
    ]

    for proj_num, folder, title, desc, main_code in ft_projects:
        from pathlib import Path
        Path("sample_data").mkdir(exist_ok=True)
        paths.append(write_nb(8, folder, [
            md(f"# Project {proj_num} — {title}\n## {desc}\n\n**Stack:** LangChain · Ollama · Jupyter"),
            code("# !pip install -q langchain langchain-ollama pandas pydantic"),
            md("## Implementation"),
            code(main_code),
            md(f"## What You Learned\n- **{title}** — {desc.lower()}\n- **Dataset engineering** for model improvement\n- **Quality metrics** for training data curation"),
        ]))

    print(f"Group 8 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
