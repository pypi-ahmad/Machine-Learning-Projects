"""Enrich batch C — Projects 73-80 (Fine-Tuning Adjacent) to 10+ cells each."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── 73 — Prompt vs Fine-Tune Comparison ─────────────────────────────
    paths.append(write_nb(8, "73_Prompt_vs_FineTune_Comparison", [
        md("# Project 73 — Prompt vs Fine-Tune Comparison Lab\n## Zero-Shot → Few-Shot → Detailed Instruction → Benchmark\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Define Evaluation Tasks"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd, time

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# Ground-truth labeled test set
test_data = [
    ("I absolutely love this product — best purchase ever!", "positive"),
    ("Terrible experience. Waste of money, would not recommend.", "negative"),
    ("It's okay, does what it's supposed to do, nothing special.", "neutral"),
    ("Exceeded my expectations! Five stars!", "positive"),
    ("Broke after two days. Horrible quality.", "negative"),
    ("Decent value for the price. Not amazing, not bad.", "neutral"),
    ("Amazing customer service and fast shipping!", "positive"),
    ("Worst product I've ever used. Total garbage.", "negative"),
    ("Works fine. Meets basic requirements.", "neutral"),
    ("This changed my life — I can't go back!", "positive"),
    ("Don't waste your money. Completely useless.", "negative"),
    ("Average product. You get what you pay for.", "neutral"),
]
print(f"Test set: {len(test_data)} labeled samples")
print(f"Distribution: {pd.Series([l for _,l in test_data]).value_counts().to_dict()}")
"""),
        md("## Step 2 — Define Four Prompt Strategies"),
        code("""
strategies = {
    "zero_shot": ChatPromptTemplate.from_template(
        "Classify the sentiment as positive, negative, or neutral: {text}\\nLabel:"
    ),
    "few_shot": ChatPromptTemplate.from_template(
        \"\"\"Classify sentiment. Examples:
"Great product!" → positive
"Horrible quality" → negative
"It's fine" → neutral
"Love it!" → positive
"Terrible" → negative

Text: {text}
Label:\"\"\"
    ),
    "cot": ChatPromptTemplate.from_template(
        \"\"\"Analyze the sentiment step by step:
1. Identify emotional words
2. Determine overall tone
3. Classify as positive/negative/neutral

Text: {text}
Analysis and Label:\"\"\"
    ),
    "detailed_instruction": ChatPromptTemplate.from_template(
        \"\"\"You are a sentiment classification model trained on product reviews.
Classification rules:
- "positive" = satisfaction, praise, excitement, recommendation, happiness
- "negative" = dissatisfaction, complaint, frustration, warning against buying
- "neutral" = factual, indifferent, balanced pros/cons, acceptable

CRITICAL: Respond with ONLY ONE WORD: positive, negative, or neutral.

Text: {text}
Label:\"\"\"
    ),
}
print(f"Strategies: {list(strategies.keys())}")
"""),
        md("## Step 3 — Run All Strategies"),
        code("""
results = []
for strat_name, prompt in strategies.items():
    chain = prompt | llm | StrOutputParser()
    correct = 0
    start = time.time()
    for text, expected in test_data:
        predicted = chain.invoke({"text": text}).strip().lower()
        # Extract label from response
        for label in ["positive", "negative", "neutral"]:
            if label in predicted:
                predicted = label
                break
        match = predicted == expected
        correct += match
        results.append({
            "strategy": strat_name, "text": text[:40],
            "expected": expected, "predicted": predicted,
            "correct": match,
        })
    elapsed = time.time() - start
    acc = correct / len(test_data)
    print(f"  {strat_name:<25} accuracy={acc:.0%} ({correct}/{len(test_data)}) in {elapsed:.1f}s")
"""),
        md("## Step 4 — Detailed Analysis"),
        code("""
df = pd.DataFrame(results)

# Per-strategy metrics
print("STRATEGY COMPARISON")
print("=" * 60)
for strat in strategies:
    sub = df[df["strategy"] == strat]
    acc = sub["correct"].mean()
    errors = sub[~sub["correct"]]
    print(f"\\n{strat}: {acc:.0%} accuracy")
    if len(errors) > 0:
        print(f"  Errors:")
        for _, e in errors.iterrows():
            print(f"    '{e['text']}' → expected={e['expected']}, got={e['predicted']}")

# Per-label accuracy
print("\\nPER-LABEL ACCURACY:")
for label in ["positive", "negative", "neutral"]:
    sub = df[df["expected"] == label]
    by_strat = sub.groupby("strategy")["correct"].mean().round(2)
    print(f"  {label}: {by_strat.to_dict()}")
"""),
        md("## Step 5 — Winner & Recommendations"),
        code("""
summary = df.groupby("strategy")["correct"].mean().sort_values(ascending=False)
print("LEADERBOARD")
print("=" * 40)
for i, (strat, acc) in enumerate(summary.items(), 1):
    medal = ["🥇","🥈","🥉","  "][min(i-1,3)]
    print(f"  {medal} {strat:<25} {acc:.0%}")

winner = summary.index[0]
print(f"\\nBest strategy: {winner}")
print(f"\\nRecommendation: {'Fine-tuning likely unnecessary' if summary.iloc[0] > 0.85 else 'Consider fine-tuning'}")
"""),
        md("## What You Learned\n- **Four prompt strategies** compared on same data\n- **Per-label accuracy** to spot systematic weaknesses\n- **Fine-tuning necessity assessment** based on prompt ceiling\n- **Error analysis** for targeted improvement"),
    ]))

    # ── 74 — Style Dataset Creator ──────────────────────────────────────
    paths.append(write_nb(8, "74_Style_Dataset_Creator", [
        md("# Project 74 — Style Dataset Creator\n## Extract Writing Style → Generate Style-Matched Training Data\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Style Samples"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.7)

style_samples = {
    "technical": [
        "The microservices architecture employs gRPC for inter-service communication with "
        "Protocol Buffers serialization, achieving sub-millisecond latency.",
        "Database sharding distributes data across partitions using consistent hashing, "
        "ensuring horizontal scalability with minimal rebalancing overhead.",
    ],
    "casual": [
        "So basically we split everything into tiny services that talk to each other. "
        "It's super fast and way easier to debug than the old monolith.",
        "We spread the data across multiple databases so nothing gets too big. "
        "Pretty slick setup honestly!",
    ],
    "executive": [
        "Our platform modernization reduced operational costs by 35% while improving "
        "system reliability to 99.95% uptime, positioning us for 10x scale.",
        "Strategic investment in AI capabilities has yielded 40% faster time-to-market "
        "and a 25% improvement in customer satisfaction scores.",
    ],
}
print(f"Style categories: {list(style_samples.keys())}")
"""),
        md("## Step 2 — Analyze Style DNA"),
        code("""
class StyleDNA(BaseModel):
    tone: str
    formality: int = Field(ge=1, le=5, description="1=casual, 5=formal")
    avg_sentence_length: str = Field(description="short, medium, long")
    vocabulary_level: str = Field(description="simple, intermediate, advanced, technical")
    uses_jargon: bool
    uses_metrics: bool
    distinctive_patterns: list[str]

analyzer = llm.with_structured_output(StyleDNA)

profiles = {}
for style, samples in style_samples.items():
    combined = "\\n".join(samples)
    profile = analyzer.invoke(f"Analyze the writing style of these samples:\\n{combined}")
    profiles[style] = profile
    print(f"\\n{style.upper()} STYLE DNA:")
    print(f"  Tone: {profile.tone}")
    print(f"  Formality: {profile.formality}/5")
    print(f"  Vocabulary: {profile.vocabulary_level}")
    print(f"  Jargon: {profile.uses_jargon} | Metrics: {profile.uses_metrics}")
    print(f"  Patterns: {profile.distinctive_patterns}")
"""),
        md("## Step 3 — Generate Style-Matched Training Data"),
        code("""
topics = [
    "cloud computing benefits",
    "machine learning model deployment",
    "team productivity improvement",
    "data security best practices",
]

gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write about the topic in the specified style. "
     "Match the style DNA precisely: tone, formality, vocabulary."),
    ("human", "Style: {style}\\nStyle DNA: {dna}\\nTopic: {topic}\\n\\nGenerate 2 paragraphs:")
])
gen_chain = gen_prompt | llm | StrOutputParser()

dataset = []
for style, profile in profiles.items():
    for topic in topics:
        text = gen_chain.invoke({
            "style": style,
            "dna": json.dumps(profile.model_dump()),
            "topic": topic,
        })
        dataset.append({
            "style": style, "topic": topic,
            "text": text, "formality": profile.formality,
        })
        print(f"  {style}/{topic}: {len(text)} chars")

df = pd.DataFrame(dataset)
print(f"\\nDataset: {len(df)} style-matched samples")
print(df.groupby("style").size().to_string())
"""),
        md("## Step 4 — Style Consistency Verification"),
        code("""
# Cross-check: can the LLM correctly identify the style?
verify_prompt = ChatPromptTemplate.from_template(
    "What writing style is this: technical, casual, or executive?\\n\\n{text}\\n\\nStyle:"
)
verify_chain = verify_prompt | llm | StrOutputParser()

correct = 0
for _, row in df.iterrows():
    predicted = verify_chain.invoke({"text": row["text"][:300]}).strip().lower()
    match = row["style"] in predicted
    correct += match

consistency = correct / len(df) if len(df) > 0 else 0
print(f"Style consistency: {consistency:.0%} ({correct}/{len(df)})")

# Save dataset
df.to_json("sample_data/style_training_data.json", orient="records", indent=2)
print("✓ Saved to sample_data/style_training_data.json")
"""),
        md("## What You Learned\n- **Style DNA extraction** with quantified attributes\n- **Style-conditioned text generation** for training data\n- **Consistency verification** via round-trip classification\n- **Training dataset export** for style fine-tuning"),
    ]))

    # ── 75 — Instruction Quality Checker ────────────────────────────────
    paths.append(write_nb(8, "75_Instruction_Quality_Checker", [
        md("# Project 75 — Instruction Dataset Quality Checker\n## Detect Duplicates, Contradictions & Low-Quality Pairs\n\n**Stack:** LangChain · Ollama · Pydantic · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Sample Dataset with Intentional Issues"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

dataset = [
    {"id": 0, "instruction": "How do I reset my password?",
     "response": "Go to Settings > Security > Reset Password. Click 'Reset' and check your email."},
    {"id": 1, "instruction": "How can I change my password?",
     "response": "Navigate to Account > Security > Change Password. Enter old and new password."},
    {"id": 2, "instruction": "What are your business hours?",
     "response": "We're open Monday-Friday, 9am-5pm EST. Closed weekends and holidays."},
    {"id": 3, "instruction": "When is customer support available?",
     "response": "Our support team is available 24/7 via chat and email."},
    {"id": 4, "instruction": "Delete my account",
     "response": "ok"},
    {"id": 5, "instruction": "How do I export my data?",
     "response": "Go to Settings > Data > Export. Choose CSV or JSON format and click Download."},
    {"id": 6, "instruction": "Can I cancel my subscription?",
     "response": "Yes, go to Billing > Manage Subscription > Cancel. You'll keep access until period end."},
    {"id": 7, "instruction": "What's the refund policy?",
     "response": ""},
    {"id": 8, "instruction": "How to reset password",
     "response": "Click forgot password on login page."},
    {"id": 9, "instruction": "Tell me about the pricing plans",
     "response": "We offer three tiers: Basic ($9/mo), Pro ($29/mo), Enterprise (custom). "
                  "All plans include 14-day free trial."},
]
print(f"Dataset: {len(dataset)} instruction-response pairs")
"""),
        md("## Step 2 — Duplicate Detection"),
        code("""
from langchain_ollama import OllamaEmbeddings
import numpy as np

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Embed all instructions
instructions = [d["instruction"] for d in dataset]
vectors = embeddings.embed_documents(instructions)

# Find near-duplicates via cosine similarity
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

duplicates = []
for i in range(len(vectors)):
    for j in range(i+1, len(vectors)):
        sim = cosine_sim(vectors[i], vectors[j])
        if sim > 0.85:
            duplicates.append({
                "pair": f"{i}-{j}",
                "similarity": round(sim, 3),
                "inst_a": dataset[i]["instruction"],
                "inst_b": dataset[j]["instruction"],
            })

print(f"Near-duplicates found: {len(duplicates)}")
for d in duplicates:
    print(f"  [{d['pair']}] sim={d['similarity']}")
    print(f"    A: {d['inst_a']}")
    print(f"    B: {d['inst_b']}")
"""),
        md("## Step 3 — Quality & Contradiction Audit"),
        code("""
class PairQuality(BaseModel):
    pair_id: int
    issues: list[str]
    quality_score: float = Field(ge=0, le=1)
    category: str = Field(description="good, low_quality, empty_response, duplicate, contradiction")

class DatasetAudit(BaseModel):
    total_pairs: int
    good_count: int
    issues_found: list[PairQuality]
    contradictions: list[str]
    overall_score: float = Field(ge=0, le=1)

auditor = llm.with_structured_output(DatasetAudit)

audit = auditor.invoke(
    f"Audit this instruction dataset for quality issues:\\n\\n"
    f"{json.dumps(dataset, indent=2)}\\n\\n"
    f"Check for: empty responses, very short responses, near-duplicates, "
    f"contradictory information, ambiguous instructions."
)

print("QUALITY AUDIT")
print("=" * 50)
print(f"Overall score: {audit.overall_score:.0%}")
print(f"Good pairs: {audit.good_count}/{audit.total_pairs}")

if audit.contradictions:
    print(f"\\nContradictions:")
    for c in audit.contradictions:
        print(f"  ⚠ {c}")

print(f"\\nIssues by pair:")
for issue in audit.issues_found:
    print(f"  [{issue.category}] Pair {issue.pair_id}: score={issue.quality_score:.0%}")
    for i in issue.issues:
        print(f"    → {i}")
"""),
        md("## Step 4 — Auto-Fix Suggestions"),
        code("""
fix_prompt = ChatPromptTemplate.from_messages([
    ("system", "Fix this low-quality training pair. Provide an improved response "
     "that is helpful, complete, and professional. Return ONLY the fixed response."),
    ("human", "Instruction: {instruction}\\nCurrent response: {response}\\nFix:")
])
fix_chain = fix_prompt | llm | StrOutputParser()

fixed = []
for issue in audit.issues_found:
    if issue.quality_score < 0.7:
        pair = dataset[issue.pair_id]
        improved = fix_chain.invoke({
            "instruction": pair["instruction"],
            "response": pair["response"],
        })
        fixed.append({
            "id": issue.pair_id,
            "instruction": pair["instruction"],
            "original": pair["response"][:60],
            "fixed": improved[:120],
        })
        print(f"  Fixed #{issue.pair_id}: '{pair['response'][:30]}' → '{improved[:60]}...'")

print(f"\\nFixed {len(fixed)} low-quality pairs")
"""),
        md("## What You Learned\n- **Embedding-based duplicate detection** with cosine similarity\n- **Automated quality scoring** for each training pair\n- **Contradiction detection** across the dataset\n- **Auto-fix pipeline** for low-quality responses"),
    ]))

    # ── 76 — Distillation Lab ───────────────────────────────────────────
    paths.append(write_nb(8, "76_Local_Distillation_Lab", [
        md("# Project 76 — Local Distillation Lab\n## Teacher → Compressed Student → Quality Comparison\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Define Knowledge Domains"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time, pandas as pd, json
from pathlib import Path

teacher = ChatOllama(model="qwen3:8b", temperature=0.3)

domains = [
    "Explain how a hash table works, including collision resolution strategies",
    "Describe the difference between TCP and UDP with real-world examples",
    "Explain garbage collection in programming languages",
    "What is a deadlock and how do you prevent it?",
    "Explain the CAP theorem with database examples",
    "How does TLS/SSL encryption work?",
    "Explain ACID properties in databases",
    "What is eventual consistency and when is it acceptable?",
]
print(f"Knowledge domains: {len(domains)} topics")
"""),
        md("## Step 2 — Generate Teacher Explanations"),
        code("""
teacher_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher. Provide a thorough explanation with examples, "
     "analogies, and edge cases. Be comprehensive."),
    ("human", "{topic}")
])
teacher_chain = teacher_prompt | teacher | StrOutputParser()

teacher_outputs = []
for topic in domains:
    start = time.time()
    explanation = teacher_chain.invoke({"topic": topic})
    elapsed = time.time() - start
    teacher_outputs.append({
        "topic": topic[:50], "output": explanation,
        "chars": len(explanation), "words": len(explanation.split()),
        "time_s": round(elapsed, 2),
    })
    print(f"  {topic[:45]}... → {len(explanation)} chars ({elapsed:.1f}s)")

print(f"\\nTeacher avg: {sum(t['chars'] for t in teacher_outputs)//len(teacher_outputs)} chars/topic")
"""),
        md("## Step 3 — Distill to Student Format"),
        code("""
compress_prompt = ChatPromptTemplate.from_messages([
    ("system", "Compress this teacher explanation into a concise student version. "
     "Rules:\\n- Keep ALL key facts, definitions, and numbers\\n"
     "- Remove examples, analogies, and verbose explanations\\n"
     "- Target 30% of original length\\n- Use bullet points"),
    ("human", "Teacher explanation:\\n{teacher}")
])
compress_chain = compress_prompt | teacher | StrOutputParser()

distilled = []
for t_out in teacher_outputs:
    start = time.time()
    student = compress_chain.invoke({"teacher": t_out["output"]})
    elapsed = time.time() - start
    ratio = len(student) / max(len(t_out["output"]), 1)
    distilled.append({
        "topic": t_out["topic"],
        "teacher_output": t_out["output"],
        "student_output": student,
        "teacher_chars": t_out["chars"],
        "student_chars": len(student),
        "compression_ratio": round(ratio, 2),
        "time_s": round(elapsed, 2),
    })
    print(f"  {t_out['topic'][:40]}... {t_out['chars']}→{len(student)} chars (ratio={ratio:.0%})")

Path("sample_data").mkdir(exist_ok=True)
with open("sample_data/distillation_data.json", "w") as f:
    json.dump(distilled, f, indent=2, default=str)
"""),
        md("## Step 4 — Quality Comparison"),
        code("""
from pydantic import BaseModel, Field

class QualityScore(BaseModel):
    completeness: float = Field(ge=0, le=1, description="Are all key facts preserved?")
    accuracy: float = Field(ge=0, le=1, description="Is everything correct?")
    clarity: float = Field(ge=0, le=1, description="Is it easy to understand?")

judge = teacher.with_structured_output(QualityScore)

rows = []
for d in distilled:
    score = judge.invoke(
        f"Compare student vs teacher explanation.\\n\\n"
        f"Teacher: {d['teacher_output'][:500]}\\n\\n"
        f"Student: {d['student_output'][:500]}\\n\\n"
        f"Rate the student version."
    )
    rows.append({
        "topic": d["topic"],
        "compression": d["compression_ratio"],
        "completeness": score.completeness,
        "accuracy": score.accuracy,
        "clarity": score.clarity,
        "avg_score": round((score.completeness + score.accuracy + score.clarity) / 3, 2),
    })

qdf = pd.DataFrame(rows)
print("DISTILLATION QUALITY")
print("=" * 60)
print(qdf.to_string(index=False))
print(f"\\nAvg compression: {qdf['compression'].mean():.0%}")
print(f"Avg quality:     {qdf['avg_score'].mean():.0%}")
"""),
        md("## What You Learned\n- **Teacher-student distillation** pipeline\n- **Compression with fact preservation** constraints\n- **Quality scoring** of compressed outputs\n- **Dataset generation** for student model training"),
    ]))

    # ── 77 — Preference Pair Builder ────────────────────────────────────
    paths.append(write_nb(8, "77_Preference_Pair_Builder", [
        md("# Project 77 — Preference Pair Builder\n## Generate Chosen/Rejected Pairs for RLHF-Style Training\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Define Tasks & Generate Candidates"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd
from pathlib import Path

judge = ChatOllama(model="qwen3:8b", temperature=0.0)

prompts = [
    "Write a product description for wireless earbuds",
    "Explain recursion to a beginner",
    "Draft a LinkedIn post about starting a new job",
    "Write a bug report for a login failure",
    "Create a meeting agenda for a sprint retrospective",
    "Explain why unit tests are important",
]

# Generate 3 candidates per prompt at different temperatures
all_candidates = {}
for prompt in prompts:
    outputs = []
    for temp in [0.1, 0.5, 0.9]:
        gen = ChatOllama(model="qwen3:8b", temperature=temp)
        chain = ChatPromptTemplate.from_template("{p}") | gen | StrOutputParser()
        outputs.append(chain.invoke({"p": prompt}))
    all_candidates[prompt] = outputs
    print(f"  Generated 3 candidates for: {prompt[:40]}...")

print(f"\\nTotal: {sum(len(v) for v in all_candidates.values())} candidate outputs")
"""),
        md("## Step 2 — Pairwise Ranking"),
        code("""
class PairwiseJudgment(BaseModel):
    winner: str = Field(description="A or B")
    score_a: int = Field(ge=1, le=5)
    score_b: int = Field(ge=1, le=5)
    reasoning: str
    criteria_used: list[str]

ranker = judge.with_structured_output(PairwiseJudgment)

preference_pairs = []
for prompt, outputs in all_candidates.items():
    # Compare all pairs
    best_idx, worst_idx = 0, 0
    best_score, worst_score = 0, 10

    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            judgment = ranker.invoke(
                f"Prompt: {prompt}\\n\\n"
                f"Response A:\\n{outputs[i][:400]}\\n\\n"
                f"Response B:\\n{outputs[j][:400]}\\n\\n"
                f"Which response is better? Evaluate: relevance, clarity, completeness."
            )
            if judgment.winner == "A" and judgment.score_a > best_score:
                best_idx, best_score = i, judgment.score_a
            elif judgment.winner == "B" and judgment.score_b > best_score:
                best_idx, best_score = j, judgment.score_b

    # Set worst as the other extreme
    scores = []
    for i in range(len(outputs)):
        if i != best_idx:
            j_check = ranker.invoke(
                f"Rate this response to '{prompt[:40]}...':\\n{outputs[i][:300]}\\n\\n"
                f"Score 1-5:"
            )
            scores.append((i, j_check.score_a))
    worst_idx = min(scores, key=lambda x: x[1])[0] if scores else (1 if best_idx == 0 else 0)

    preference_pairs.append({
        "prompt": prompt,
        "chosen": outputs[best_idx][:500],
        "rejected": outputs[worst_idx][:500],
        "best_temp": [0.1, 0.5, 0.9][best_idx],
        "worst_temp": [0.1, 0.5, 0.9][worst_idx],
    })
    print(f"  {prompt[:40]}... best=T{[0.1,0.5,0.9][best_idx]}, worst=T{[0.1,0.5,0.9][worst_idx]}")
"""),
        md("## Step 3 — Export in DPO Format"),
        code("""
# DPO format
dpo_data = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
            for p in preference_pairs]

Path("sample_data").mkdir(exist_ok=True)
with open("sample_data/preference_pairs.json", "w") as f:
    json.dump(dpo_data, f, indent=2)

# RLHF format
rlhf_data = []
for p in preference_pairs:
    rlhf_data.append({"prompt": p["prompt"], "response": p["chosen"], "label": 1})
    rlhf_data.append({"prompt": p["prompt"], "response": p["rejected"], "label": 0})

with open("sample_data/rlhf_pairs.jsonl", "w") as f:
    for item in rlhf_data:
        f.write(json.dumps(item) + "\\n")

print(f"Exported {len(dpo_data)} DPO pairs → sample_data/preference_pairs.json")
print(f"Exported {len(rlhf_data)} RLHF entries → sample_data/rlhf_pairs.jsonl")
"""),
        md("## Step 4 — Temperature Analysis"),
        code("""
temp_results = pd.DataFrame([{
    "prompt": p["prompt"][:30],
    "best_temp": p["best_temp"],
    "worst_temp": p["worst_temp"],
} for p in preference_pairs])

print("TEMPERATURE PREFERENCE ANALYSIS")
print("=" * 50)
print(f"Best temperature distribution: {temp_results['best_temp'].value_counts().to_dict()}")
print(f"Worst temperature distribution: {temp_results['worst_temp'].value_counts().to_dict()}")
print(f"\\nInsight: Temperature {temp_results['best_temp'].mode().iloc[0]} most often produces best results")
"""),
        md("## What You Learned\n- **Pairwise preference ranking** for RLHF\n- **Multi-temperature candidate generation**\n- **DPO & RLHF format export**\n- **Temperature impact analysis** on output quality"),
    ]))

    # ── 78 — JSON Extraction Dataset Builder ────────────────────────────
    paths.append(write_nb(8, "78_JSON_Extraction_Dataset_Builder", [
        md("# Project 78 — JSON Extraction Dataset Builder\n## Document → Schema → Extract → Validate → Export\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Define Extraction Schemas"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd
from pathlib import Path

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

# Define multiple target schemas
class PersonExtract(BaseModel):
    name: str
    role: str
    organization: str
    email: str = ""

class InvoiceExtract(BaseModel):
    invoice_id: str
    date: str
    vendor: str
    total: float
    items: list[str]

class EventExtract(BaseModel):
    title: str
    date: str
    time: str
    location: str
    attendees: list[str]

class ProductExtract(BaseModel):
    name: str
    sku: str
    price: float
    category: str
    features: list[str]

schemas = {
    "person": PersonExtract,
    "invoice": InvoiceExtract,
    "event": EventExtract,
    "product": ProductExtract,
}
print(f"Extraction schemas: {list(schemas.keys())}")
"""),
        md("## Step 2 — Source Documents"),
        code("""
documents = [
    {"type": "person", "text": "Dr. Sarah Chen, Chief Data Officer at DataFlow Inc, "
     "has been leading the analytics transformation since 2022. Contact: sarah@dataflow.io"},
    {"type": "person", "text": "Meeting with John Smith (VP Engineering, TechCorp). "
     "Email: j.smith@techcorp.com. Discussed Q2 roadmap."},
    {"type": "invoice", "text": "Invoice #INV-2025-0042 from CloudServ Corp, dated Feb 1 2025. "
     "Items: API Gateway ($800), Storage addon ($200), Support ($500). Total: $1,500."},
    {"type": "invoice", "text": "Bill #B-9901 — DataPipe Solutions — March 10 2025. "
     "Professional Services (40hrs × $150 = $6000), License ($2000). Grand total: $8,000."},
    {"type": "event", "text": "Q1 All-Hands Meeting on March 25, 2025 at 2:00 PM in Conference Room A. "
     "Attendees: Alice, Bob, Carol, Dave, Eve. Agenda: roadmap review, team updates."},
    {"type": "product", "text": "ProMax Headphones (SKU: PMH-500), $79.99, Audio category. "
     "Features: active noise cancelling, 30hr battery, USB-C, Bluetooth 5.3, foldable design."},
]
print(f"Documents to process: {len(documents)}")
"""),
        md("## Step 3 — Extract & Validate"),
        code("""
dataset = []
for doc in documents:
    schema_class = schemas[doc["type"]]
    extractor = llm.with_structured_output(schema_class)
    try:
        extracted = extractor.invoke(f"Extract structured data:\\n{doc['text']}")
        extracted_dict = extracted.model_dump()
        valid = True
        error = ""
    except Exception as e:
        extracted_dict = {}
        valid = False
        error = str(e)[:80]

    dataset.append({
        "input": doc["text"],
        "type": doc["type"],
        "output": extracted_dict,
        "valid": valid,
        "error": error,
    })
    icon = "✓" if valid else "✗"
    print(f"  {icon} [{doc['type']}] {json.dumps(extracted_dict, default=str)[:100]}")

pass_rate = sum(1 for d in dataset if d["valid"]) / len(dataset)
print(f"\\nExtraction pass rate: {pass_rate:.0%}")
"""),
        md("## Step 4 — Generate Augmented Variants"),
        code("""
aug_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 new documents similar to the example but with different data. "
     "Vary names, numbers, dates, and details. Return only the text, one per line."),
    ("human", "Example: {text}\\n\\nGenerate 3 variants:")
])
aug_chain = aug_prompt | llm | StrOutputParser()

augmented = []
for doc in documents[:4]:
    variants = aug_chain.invoke({"text": doc["text"]})
    for line in variants.strip().split("\\n"):
        line = line.strip().lstrip("0123456789.-) ")
        if len(line) > 20:
            augmented.append({"type": doc["type"], "text": line})

print(f"Original: {len(documents)} | Augmented: {len(augmented)} | Total: {len(documents) + len(augmented)}")

# Export
Path("sample_data").mkdir(exist_ok=True)
with open("sample_data/extraction_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2, default=str)
print("✓ Saved to sample_data/extraction_dataset.json")
"""),
        md("## What You Learned\n- **Multi-schema extraction** from unstructured text\n- **Pydantic validation** for output correctness\n- **Data augmentation** to expand training sets\n- **Training data export** for extraction fine-tuning"),
    ]))

    # ── 79 — Classification Fine-Tune Readiness ────────────────────────
    paths.append(write_nb(8, "79_Classification_FineTune_Readiness", [
        md("# Project 79 — Classification Fine-Tune Readiness Audit\n## Dataset Quality Gates Before Investing in Training\n\n**Stack:** LangChain · Ollama · Pydantic · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Sample Classification Dataset"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

dataset = [
    {"text": "Server is down! Production outage!", "label": "urgent"},
    {"text": "URGENT: payment system failure", "label": "urgent"},
    {"text": "Everything is broken!!!!", "label": "urgent"},
    {"text": "When do you close today?", "label": "inquiry"},
    {"text": "What time do you open?", "label": "inquiry"},
    {"text": "Do you have a student discount?", "label": "inquiry"},
    {"text": "I want to cancel my subscription", "label": "churn"},
    {"text": "Cancel immediately, terrible service", "label": "churn"},
    {"text": "The app crashed again", "label": "bug"},
    {"text": "Login button not working on mobile", "label": "bug"},
    {"text": "Love your new feature!", "label": "feedback"},
    {"text": "Nice update, the UI looks great", "label": "feedback"},
    {"text": "Everything broke after the latest update", "label": "urgent"},  # or bug?
    {"text": "How much does the premium plan cost?", "label": "inquiry"},
    {"text": "Great customer service!", "label": "feedback"},
]

df = pd.DataFrame(dataset)
print(f"Dataset: {len(df)} samples")
print(f"Labels: {df['label'].value_counts().to_dict()}")
"""),
        md("## Step 2 — Statistical Quality Checks"),
        code("""
# Check 1: Class balance
counts = df["label"].value_counts()
min_count = counts.min()
max_count = counts.max()
balance_ratio = min_count / max_count
print("CHECK 1: Class Balance")
print(f"  Counts: {counts.to_dict()}")
print(f"  Balance ratio: {balance_ratio:.2f} (target > 0.5)")
print(f"  {'✓ PASS' if balance_ratio > 0.3 else '✗ FAIL'}")

# Check 2: Minimum samples per class
min_threshold = 5
print(f"\\nCHECK 2: Minimum Samples (threshold={min_threshold})")
for label, count in counts.items():
    print(f"  {label}: {count} {'✓' if count >= min_threshold else '✗ INSUFFICIENT'}")

# Check 3: Text length distribution
df["text_len"] = df["text"].str.len()
print(f"\\nCHECK 3: Text Length")
print(f"  Mean: {df['text_len'].mean():.0f} chars")
print(f"  Min:  {df['text_len'].min()} | Max: {df['text_len'].max()}")
print(f"  Very short (<10): {(df['text_len'] < 10).sum()}")

# Check 4: Total dataset size
print(f"\\nCHECK 4: Dataset Size")
print(f"  Total: {len(df)} (recommended: >100 for fine-tuning)")
print(f"  {'✓ OK' if len(df) > 50 else '⚠ Too small for reliable fine-tuning'}")
"""),
        md("## Step 3 — LLM-Powered Ambiguity Detection"),
        code("""
class AmbiguityCheck(BaseModel):
    sample_id: int
    assigned_label: str
    could_be: list[str] = Field(description="Other plausible labels")
    ambiguity_score: float = Field(ge=0, le=1)
    reasoning: str

class ReadinessReport(BaseModel):
    readiness: str = Field(description="ready, needs_work, not_ready")
    overall_score: float = Field(ge=0, le=1)
    ambiguous_samples: list[AmbiguityCheck]
    recommendations: list[str]
    estimated_accuracy_ceiling: float = Field(ge=0, le=1)

auditor = llm.with_structured_output(ReadinessReport)

report = auditor.invoke(
    f"Audit this text classification dataset for fine-tuning readiness.\\n"
    f"Labels: {list(counts.index)}\\n\\n{json.dumps(dataset, indent=2)}\\n\\n"
    f"Check for: ambiguous labels, overlapping categories, inconsistent labeling."
)

print("READINESS AUDIT")
print("=" * 50)
print(f"Readiness: {report.readiness.upper()}")
print(f"Overall score: {report.overall_score:.0%}")
print(f"Estimated accuracy ceiling: {report.estimated_accuracy_ceiling:.0%}")

print(f"\\nAmbiguous samples:")
for a in report.ambiguous_samples:
    print(f"  #{a.sample_id} [{a.assigned_label}] could be {a.could_be} (ambiguity={a.ambiguity_score:.0%})")
"""),
        md("## Step 4 — Recommendations"),
        code("""
print("RECOMMENDATIONS")
print("=" * 50)
for i, rec in enumerate(report.recommendations, 1):
    print(f"  {i}. {rec}")

# Generate action items
action_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a prioritized action plan to make this dataset ready for fine-tuning."),
    ("human", "Current state: {len} samples, {labels} labels, readiness={readiness}, "
     "score={score:.0%}, issues: {issues}")
])
action_chain = action_prompt | llm | StrOutputParser()

action_plan = action_chain.invoke({
    "len": len(df),
    "labels": len(counts),
    "readiness": report.readiness,
    "score": report.overall_score,
    "issues": "; ".join(report.recommendations[:3]),
})
print(f"\\nACTION PLAN:")
print(action_plan[:500])
"""),
        md("## What You Learned\n- **Statistical quality gates** for training data\n- **Ambiguity detection** with LLM audit\n- **Readiness scoring** before fine-tuning investment\n- **Actionable recommendations** for dataset improvement"),
    ]))

    # ── 80 — Fine-Tuning Evals Harness ─────────────────────────────────
    paths.append(write_nb(8, "80_Local_FineTuning_Evals_Harness", [
        md("# Project 80 — Local Fine-Tuning Evals Harness\n## Multi-Task Evaluation Framework for Model Assessment\n\n**Stack:** LangChain · Ollama · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pandas"),
        md("## Step 1 — Define Eval Suite"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd, time, json

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

eval_suite = {
    "classification": {
        "prompt": "Classify as positive/negative/neutral: {text}\\nLabel:",
        "cases": [
            {"text": "Amazing product!", "expected": "positive"},
            {"text": "Terrible service.", "expected": "negative"},
            {"text": "It works fine.", "expected": "neutral"},
            {"text": "Love it!", "expected": "positive"},
            {"text": "Total waste.", "expected": "negative"},
        ],
        "scorer": lambda exp, pred: 1.0 if exp.lower() in pred.lower() else 0.0,
    },
    "extraction": {
        "prompt": "Extract the person's name and date from: {text}\\nJSON:",
        "cases": [
            {"text": "Meeting with Alice on Jan 15", "expected": ["alice", "jan"]},
            {"text": "Call from Dr. Bob Chen, March 3", "expected": ["bob", "march"]},
            {"text": "Lunch with Carol — February 28", "expected": ["carol", "february"]},
        ],
        "scorer": lambda exp, pred: sum(1 for e in exp if e in pred.lower()) / len(exp),
    },
    "summarization": {
        "prompt": "Summarize in one sentence: {text}",
        "cases": [
            {"text": "Machine learning models learn from data. They find patterns. "
             "They generalize to new inputs. Overfitting is a common problem.",
             "expected": ["learn", "data", "pattern"]},
            {"text": "Python is a programming language. It's easy to read. "
             "It supports multiple paradigms. It has a large ecosystem.",
             "expected": ["python", "language"]},
        ],
        "scorer": lambda exp, pred: sum(1 for w in exp if w in pred.lower()) / len(exp),
    },
    "reasoning": {
        "prompt": "Answer: {text}",
        "cases": [
            {"text": "If all roses are flowers and all flowers need water, do roses need water?",
             "expected": ["yes"]},
            {"text": "A bat and ball cost $1.10. The bat costs $1 more than the ball. "
             "What does the ball cost?",
             "expected": ["0.05", "5 cent", "five cent"]},
        ],
        "scorer": lambda exp, pred: 1.0 if any(e in pred.lower() for e in exp) else 0.0,
    },
}
print(f"Eval suite: {len(eval_suite)} tasks, "
      f"{sum(len(t['cases']) for t in eval_suite.values())} total cases")
"""),
        md("## Step 2 — Run Full Evaluation"),
        code("""
results = []
for task_name, task_config in eval_suite.items():
    prompt = ChatPromptTemplate.from_template(task_config["prompt"])
    chain = prompt | llm | StrOutputParser()
    scorer = task_config["scorer"]

    for case in task_config["cases"]:
        start = time.time()
        output = chain.invoke({"text": case["text"]})
        elapsed = time.time() - start
        score = scorer(case["expected"], output)

        results.append({
            "task": task_name,
            "input": case["text"][:40],
            "score": round(score, 2),
            "latency_s": round(elapsed, 2),
            "output_len": len(output),
        })

df = pd.DataFrame(results)
print("EVALUATION RESULTS")
print("=" * 60)
print(df.to_string(index=False))
"""),
        md("## Step 3 — Task-Level Analysis"),
        code("""
print("TASK-LEVEL SUMMARY")
print("=" * 50)
task_summary = df.groupby("task").agg({
    "score": ["mean", "min", "max", "std"],
    "latency_s": "mean",
}).round(3)
print(task_summary.to_string())

print(f"\\nOVERALL METRICS:")
print(f"  Mean score:    {df['score'].mean():.0%}")
print(f"  Median score:  {df['score'].median():.0%}")
print(f"  Mean latency:  {df['latency_s'].mean():.2f}s")

# Weakest area
weakest = df.groupby("task")["score"].mean().idxmin()
print(f"\\nWeakest task: {weakest} ({df.groupby('task')['score'].mean()[weakest]:.0%})")
print(f"Strongest task: {df.groupby('task')['score'].mean().idxmax()} ({df.groupby('task')['score'].mean().max():.0%})")
"""),
        md("## Step 4 — Pass/Fail Report Card"),
        code("""
thresholds = {"classification": 0.8, "extraction": 0.7, "summarization": 0.6, "reasoning": 0.5}

print("REPORT CARD")
print("=" * 50)
all_pass = True
for task, threshold in thresholds.items():
    avg = df[df["task"] == task]["score"].mean()
    passed = avg >= threshold
    all_pass = all_pass and passed
    icon = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {icon} {task:<20} {avg:.0%} (threshold={threshold:.0%})")

print(f"\\nOverall: {'✓ ALL PASSED' if all_pass else '✗ NEEDS IMPROVEMENT'}")

if not all_pass:
    print("\\nFailed tasks need attention before fine-tuning is worthwhile.")
    print("Consider improving prompt strategy or collecting more training data.")
"""),
        md("## What You Learned\n- **Multi-task evaluation harness** with custom scorers\n- **Task-specific thresholds** for pass/fail gates\n- **Performance profiling** per task category\n- **Report card generation** for model readiness assessment"),
    ]))

    print(f"\\nEnriched {len(paths)} notebooks (73-80)")
    for p in paths:
        print(f"  ✓ {p}")

if __name__ == "__main__":
    build()
