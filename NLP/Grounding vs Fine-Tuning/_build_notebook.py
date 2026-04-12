"""Build the Grounding vs Fine-Tuning notebook."""
import json, pathlib

cells = []

def _lines(text: str):
    parts = text.split("\n")
    for i in range(len(parts) - 1):
        parts[i] += "\n"
    return parts

def md(text: str):
    cells.append({"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": _lines(text)})

def code(text: str):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {"language": "python"}, "outputs": [], "source": _lines(text)})


# ══════════════════════════════════════════════
# 1  Title
# ══════════════════════════════════════════════
md(r"""# Grounding vs Fine-Tuning: Two Paths to Factual Answers

**Goal:** compare two strategies for making a model give factual, grounded answers — retrieval-only grounding (RAG) vs fine-tuning on corrected grounded answers — and explain the tradeoffs honestly.

---

## The Problem

Language models hallucinate. When asked domain-specific questions, they confidently generate plausible-sounding answers that may be factually wrong.

Two popular strategies address this:

| Strategy | How it works | Core idea |
|---|---|---|
| **Retrieval grounding (RAG)** | Retrieve relevant passages at inference time, include them in the prompt | Give the model the facts just-in-time |
| **Fine-tuning on grounded answers** | Train on (question, retrieved context, corrected answer) triples | Teach the model to use retrieved context better |

This notebook compares them side-by-side on the same task and evaluation set.

## The Honest Starting Point

Neither strategy is universally better. Each has strengths the other lacks, and combining them is sometimes — but not always — worth the added complexity.

## Pipeline Overview

```text
Knowledge base (documents)
     │
     ├──── Strategy A: Retrieval Only ──────────────┐
     │     retrieve → prompt with context → generate │
     │                                               │
     ├──── Strategy B: Fine-Tuned + Retrieval ──────┐│
     │     retrieve → fine-tuned model → generate    ││
     │                                               ││
     └── Same evaluation on both ◄───────────────────┘│
                                  ◄────────────────────┘
```
""")

# ══════════════════════════════════════════════
# 2  Setup
# ══════════════════════════════════════════════
md("## 1. Environment Setup")

code("""!pip install -q pandas numpy scikit-learn datasets transformers peft trl accelerate seaborn matplotlib""")

code("""import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path.cwd()
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = ARTIFACT_DIR / "grounded-lora"

RUN_RAG_EVAL = False
RUN_TRAINING = False
RUN_FT_EVAL = False

print(f"Project dir:  {PROJECT_DIR}")
print(f"Base model:   {BASE_MODEL}")
print(f"RAG eval:     {RUN_RAG_EVAL}")
print(f"Training:     {RUN_TRAINING}")
print(f"FT eval:      {RUN_FT_EVAL}")""")

# ══════════════════════════════════════════════
# 3  Knowledge base
# ══════════════════════════════════════════════
md(r"""## 2. The Knowledge Base

We create a small synthetic knowledge base of technical product documentation. In a real system, this would be an indexed corpus of company docs, wikis, or manuals.

Each document has an ID, a title, and content. The retriever returns the relevant document(s) for each question.
""")

code("""KNOWLEDGE_BASE = {
    "DOC-001": {
        "title": "API Rate Limits",
        "content": (
            "All API plans enforce rate limits. The Free plan allows 100 requests per minute. "
            "The Pro plan allows 1,000 requests per minute. The Enterprise plan allows 10,000 "
            "requests per minute with burst capacity up to 15,000 for 30-second windows. "
            "Rate limit responses return HTTP 429 with a Retry-After header in seconds. "
            "Exceeding the burst limit on Enterprise triggers a 60-second cooldown."
        ),
    },
    "DOC-002": {
        "title": "Authentication Methods",
        "content": (
            "The platform supports three authentication methods: API key (header X-API-Key), "
            "OAuth 2.0 (authorization code flow with PKCE), and service account tokens "
            "(JWT signed with RS256, valid for 1 hour). API keys do not expire but can be "
            "revoked. OAuth tokens expire after 30 minutes and can be refreshed. "
            "Service account tokens cannot be refreshed; a new token must be issued."
        ),
    },
    "DOC-003": {
        "title": "Webhook Configuration",
        "content": (
            "Webhooks deliver event payloads via HTTP POST to a user-configured URL. "
            "Payloads are signed with HMAC-SHA256 using the webhook secret. "
            "Delivery is retried up to 5 times with exponential backoff starting at 10 seconds. "
            "After 5 failures, the webhook is automatically disabled and the account owner "
            "is notified by email. Events are buffered for 72 hours and can be replayed "
            "from the dashboard."
        ),
    },
    "DOC-004": {
        "title": "Data Retention Policy",
        "content": (
            "User data is retained for the lifetime of the account. After account deletion, "
            "all personal data is purged within 30 days. Audit logs are retained for 1 year "
            "regardless of account status. Backups are kept for 90 days on a rolling basis. "
            "Customers on the Enterprise plan can request extended retention up to 7 years "
            "for compliance purposes."
        ),
    },
    "DOC-005": {
        "title": "Deployment Environments",
        "content": (
            "Each project has three environments: development, staging, and production. "
            "Development uses shared infrastructure and resets nightly. Staging mirrors "
            "the production architecture but with reduced capacity (2 replicas vs 6). "
            "Production deployments require approval from at least one reviewer and pass "
            "all CI checks. Rollbacks are instantaneous via the previously cached container image."
        ),
    },
    "DOC-006": {
        "title": "Billing and Invoicing",
        "content": (
            "Billing is usage-based, calculated at the end of each calendar month. "
            "Invoices are generated on the 1st of the following month and due within 30 days. "
            "Accepted payment methods: credit card (Visa, Mastercard, Amex), wire transfer "
            "(Enterprise only, minimum $1,000), and ACH direct debit (US accounts only). "
            "Late payments incur a 1.5% monthly fee. Annual prepayment gets a 15% discount."
        ),
    },
    "DOC-007": {
        "title": "SLA and Uptime",
        "content": (
            "The platform SLA guarantees 99.95% monthly uptime for production environments "
            "on Pro and Enterprise plans. Free plan has no SLA. Uptime is measured excluding "
            "planned maintenance windows, which are announced 72 hours in advance. "
            "SLA credits: 10% for 99.9-99.95%, 25% for 99.0-99.9%, 50% for below 99.0%. "
            "Credits are applied automatically to the next invoice."
        ),
    },
    "DOC-008": {
        "title": "File Upload Limits",
        "content": (
            "Single file uploads are limited to 100 MB on all plans. Multipart uploads "
            "support files up to 5 GB with a minimum part size of 5 MB. Concurrent uploads "
            "are limited to 10 per account. Files are scanned for malware before processing. "
            "Supported formats: PDF, PNG, JPEG, CSV, JSON, XLSX. Unsupported formats are "
            "rejected with HTTP 415."
        ),
    },
}

print(f"Knowledge base: {len(KNOWLEDGE_BASE)} documents")
for doc_id, doc in KNOWLEDGE_BASE.items():
    print(f"  {doc_id}: {doc['title']} ({len(doc['content'])} chars)")""")

# ══════════════════════════════════════════════
# 4  QA dataset
# ══════════════════════════════════════════════
md(r"""## 3. QA Dataset: Questions, Retrieved Context, and Ground-Truth Answers

Each record contains:
- `question`: what the user asks
- `doc_id`: which document is relevant (simulates the retriever)
- `ground_truth`: the factually correct answer grounded in the document
- `difficulty`: how tricky the question is for a model without retrieval

### Difficulty Levels

| Level | Description |
|---|---|
| `direct` | Answer is stated almost verbatim in the document |
| `inference` | Answer requires light reasoning over the document |
| `multi-fact` | Answer combines multiple facts from the document |
| `boundary` | Answer involves an edge case or limit not explicitly stated |
""")

code("""qa_data = [
    # ── DOC-001: Rate Limits ──
    {"question": "How many API requests per minute does the Pro plan allow?",
     "doc_id": "DOC-001", "difficulty": "direct",
     "ground_truth": "The Pro plan allows 1,000 requests per minute."},
    {"question": "What happens if an Enterprise customer exceeds the burst limit?",
     "doc_id": "DOC-001", "difficulty": "inference",
     "ground_truth": "Exceeding the burst limit on Enterprise triggers a 60-second cooldown."},
    {"question": "What HTTP status code is returned when rate limits are hit and what header is included?",
     "doc_id": "DOC-001", "difficulty": "multi-fact",
     "ground_truth": "HTTP 429 is returned with a Retry-After header indicating the wait time in seconds."},
    {"question": "Can a Free plan user burst to 200 requests per minute temporarily?",
     "doc_id": "DOC-001", "difficulty": "boundary",
     "ground_truth": "No. Only the Enterprise plan mentions burst capacity. The Free plan is limited to 100 requests per minute with no burst mentioned."},

    # ── DOC-002: Authentication ──
    {"question": "How long are OAuth tokens valid?",
     "doc_id": "DOC-002", "difficulty": "direct",
     "ground_truth": "OAuth tokens expire after 30 minutes and can be refreshed."},
    {"question": "Can service account tokens be refreshed?",
     "doc_id": "DOC-002", "difficulty": "direct",
     "ground_truth": "No. Service account tokens cannot be refreshed; a new token must be issued."},
    {"question": "Which authentication method uses RS256 signing?",
     "doc_id": "DOC-002", "difficulty": "inference",
     "ground_truth": "Service account tokens use JWT signed with RS256."},
    {"question": "If an API key is compromised, does it expire on its own?",
     "doc_id": "DOC-002", "difficulty": "boundary",
     "ground_truth": "No. API keys do not expire but can be revoked. A compromised key must be manually revoked."},

    # ── DOC-003: Webhooks ──
    {"question": "How many times are webhook deliveries retried?",
     "doc_id": "DOC-003", "difficulty": "direct",
     "ground_truth": "Webhook deliveries are retried up to 5 times with exponential backoff starting at 10 seconds."},
    {"question": "What happens after all webhook retry attempts fail?",
     "doc_id": "DOC-003", "difficulty": "inference",
     "ground_truth": "After 5 failures, the webhook is automatically disabled and the account owner is notified by email."},
    {"question": "How long are webhook events buffered and can they be replayed?",
     "doc_id": "DOC-003", "difficulty": "multi-fact",
     "ground_truth": "Events are buffered for 72 hours and can be replayed from the dashboard."},

    # ── DOC-004: Data Retention ──
    {"question": "How long after account deletion is personal data purged?",
     "doc_id": "DOC-004", "difficulty": "direct",
     "ground_truth": "All personal data is purged within 30 days after account deletion."},
    {"question": "Are audit logs deleted when an account is deleted?",
     "doc_id": "DOC-004", "difficulty": "inference",
     "ground_truth": "No. Audit logs are retained for 1 year regardless of account status."},
    {"question": "Can a Pro-plan customer request 7-year data retention?",
     "doc_id": "DOC-004", "difficulty": "boundary",
     "ground_truth": "No. Extended retention up to 7 years is only available for Enterprise customers."},

    # ── DOC-005: Deployment ──
    {"question": "How many replicas does the staging environment use?",
     "doc_id": "DOC-005", "difficulty": "direct",
     "ground_truth": "Staging uses 2 replicas, compared to 6 in production."},
    {"question": "What is required before deploying to production?",
     "doc_id": "DOC-005", "difficulty": "multi-fact",
     "ground_truth": "Production deployments require approval from at least one reviewer and must pass all CI checks."},

    # ── DOC-006: Billing ──
    {"question": "What discount do you get for annual prepayment?",
     "doc_id": "DOC-006", "difficulty": "direct",
     "ground_truth": "Annual prepayment gets a 15% discount."},
    {"question": "What is the late payment fee?",
     "doc_id": "DOC-006", "difficulty": "direct",
     "ground_truth": "Late payments incur a 1.5% monthly fee."},
    {"question": "Can a small business use wire transfer for a $500 payment?",
     "doc_id": "DOC-006", "difficulty": "boundary",
     "ground_truth": "No. Wire transfer is Enterprise only with a minimum of $1,000."},

    # ── DOC-007: SLA ──
    {"question": "What is the uptime SLA for production?",
     "doc_id": "DOC-007", "difficulty": "direct",
     "ground_truth": "The SLA guarantees 99.95% monthly uptime for production environments on Pro and Enterprise plans."},
    {"question": "What SLA credit do you get if uptime drops to 99.5%?",
     "doc_id": "DOC-007", "difficulty": "inference",
     "ground_truth": "Uptime of 99.5% falls in the 99.0-99.9% bracket, which gets a 25% SLA credit."},

    # ── DOC-008: File Uploads ──
    {"question": "What is the maximum size for a single file upload?",
     "doc_id": "DOC-008", "difficulty": "direct",
     "ground_truth": "Single file uploads are limited to 100 MB on all plans."},
    {"question": "What happens if someone uploads a .exe file?",
     "doc_id": "DOC-008", "difficulty": "boundary",
     "ground_truth": ".exe is not in the list of supported formats (PDF, PNG, JPEG, CSV, JSON, XLSX). Unsupported formats are rejected with HTTP 415."},
    {"question": "How large can a multipart upload be and what is the minimum part size?",
     "doc_id": "DOC-008", "difficulty": "multi-fact",
     "ground_truth": "Multipart uploads support files up to 5 GB with a minimum part size of 5 MB."},
]

qa_df = pd.DataFrame(qa_data)
print(f"QA pairs: {len(qa_df)}")
print(f"\\nDifficulty distribution:")
print(qa_df["difficulty"].value_counts().to_string())
print(f"\\nDoc coverage:")
print(qa_df["doc_id"].value_counts().sort_index().to_string())""")

# ══════════════════════════════════════════════
# 5  Split
# ══════════════════════════════════════════════
md("## 4. Train / Eval Split")

code("""from sklearn.model_selection import train_test_split

train_df, eval_df = train_test_split(
    qa_df, test_size=0.33, random_state=SEED, stratify=qa_df["difficulty"]
)
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)

print(f"Train: {len(train_df)}")
print(f"Eval:  {len(eval_df)}")
print(f"\\nEval difficulty counts: {dict(Counter(eval_df['difficulty']))}")""")

# ══════════════════════════════════════════════
# 6  Evaluation framework
# ══════════════════════════════════════════════
md(r"""## 5. Evaluation Framework

### Metrics

| Metric | What it measures | Why it matters |
|---|---|---|
| **Factual accuracy** | Does the answer match the ground truth? | Core correctness |
| **Groundedness** | Is every claim traceable to the retrieved context? | Hallucination detection |
| **Completeness** | Does the answer cover all facts in the ground truth? | Missing info detection |
| **Hallucination rate** | Does the answer contain facts NOT in the context? | The key failure mode |
| **Refusal when uncertain** | Does the model say "I don't know" when the context does not support an answer? | Safety check |

We use heuristic scorers here. In production, use an LLM-as-judge or human reviewers.
""")

code("""def factual_accuracy(answer, ground_truth):
    '''Heuristic: what fraction of key facts from ground_truth appear in the answer?'''
    gt_lower = ground_truth.lower()
    answer_lower = answer.lower()
    # Extract numbers and key phrases
    gt_numbers = set(re.findall(r'\\b\\d[\\d,.]*\\b', gt_lower))
    ans_numbers = set(re.findall(r'\\b\\d[\\d,.]*\\b', answer_lower))
    number_match = len(gt_numbers & ans_numbers) / len(gt_numbers) if gt_numbers else 1.0

    gt_words = set(re.findall(r'\\b[a-z]{4,}\\b', gt_lower))
    ans_words = set(re.findall(r'\\b[a-z]{4,}\\b', answer_lower))
    word_overlap = len(gt_words & ans_words) / len(gt_words) if gt_words else 1.0

    return round(0.6 * number_match + 0.4 * word_overlap, 3)


def groundedness_score(answer, context):
    '''Heuristic: what fraction of claims in the answer appear in the context?'''
    answer_sents = [s.strip() for s in re.split(r'[.!?\\n]', answer) if s.strip()]
    if not answer_sents:
        return 1.0
    ctx_lower = context.lower()
    grounded = 0
    for sent in answer_sents:
        keywords = set(re.findall(r'\\b[a-z]{4,}\\b', sent.lower()))
        if not keywords:
            grounded += 1
            continue
        hit_rate = sum(1 for k in keywords if k in ctx_lower) / len(keywords)
        if hit_rate >= 0.5:
            grounded += 1
    return round(grounded / len(answer_sents), 3)


def hallucination_check(answer, context):
    '''Heuristic: does the answer introduce numbers not in the context?'''
    ctx_numbers = set(re.findall(r'\\b\\d[\\d,.]*\\b', context.lower()))
    ans_numbers = set(re.findall(r'\\b\\d[\\d,.]*\\b', answer.lower()))
    novel = ans_numbers - ctx_numbers
    return {"hallucinated_numbers": list(novel), "has_hallucination": len(novel) > 0}


def completeness_score(answer, ground_truth):
    '''Same as factual_accuracy from the ground_truth side.'''
    return factual_accuracy(answer, ground_truth)


def evaluate_qa(answers, eval_frame, strategy_name):
    rows = []
    for i, (_, qrow) in enumerate(eval_frame.iterrows()):
        answer = answers[i]
        ctx = KNOWLEDGE_BASE[qrow["doc_id"]]["content"]
        gt = qrow["ground_truth"]

        fa = factual_accuracy(answer, gt)
        gs = groundedness_score(answer, ctx)
        cs = completeness_score(answer, gt)
        hc = hallucination_check(answer, ctx)

        rows.append({
            "question": qrow["question"][:60],
            "difficulty": qrow["difficulty"],
            "factual_accuracy": fa,
            "groundedness": gs,
            "completeness": cs,
            "has_hallucination": hc["has_hallucination"],
        })

    result_df = pd.DataFrame(rows)
    summary = {
        "strategy": strategy_name,
        "factual_accuracy": round(result_df["factual_accuracy"].mean(), 3),
        "groundedness": round(result_df["groundedness"].mean(), 3),
        "completeness": round(result_df["completeness"].mean(), 3),
        "hallucination_rate": round(result_df["has_hallucination"].mean(), 3),
    }
    return result_df, summary


print("Evaluation framework defined.")
print("Metrics: factual_accuracy, groundedness, completeness, hallucination_rate")""")

# ══════════════════════════════════════════════
# 7  Strategy A — RAG
# ══════════════════════════════════════════════
md(r"""## 6. Strategy A — Retrieval-Only Grounding (RAG)

### How It Works

1. User asks a question
2. Retriever finds the most relevant document
3. The model receives: system prompt + retrieved context + question
4. The model generates an answer grounded in the context

### No Training Required

This is purely an inference-time strategy. The model's weights are unchanged. All grounding comes from the retrieved text in the prompt.

### Prompt Design

The prompt explicitly instructs the model to:
- answer ONLY from the provided context
- say "I don't know" if the context does not cover the question
- not add information from its own training data
""")

code("""RAG_SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the user's question using ONLY "
    "the provided context. Do not add information from your own knowledge.\\n\\n"
    "Rules:\\n"
    "- If the context contains the answer, provide it concisely.\\n"
    "- If the context does not contain the answer, say 'I don't have enough "
    "information in the provided documentation to answer this.'\\n"
    "- Do not speculate or infer beyond what the context states.\\n"
    "- Cite specific details (numbers, limits, timeframes) from the context."
)


def build_rag_prompt(question, doc_id):
    doc = KNOWLEDGE_BASE[doc_id]
    context_block = f"--- Context: {doc['title']} ---\\n{doc['content']}\\n---"
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context_block}\\n\\nQuestion: {question}"},
    ]


print("RAG prompt template:")
example = build_rag_prompt("How many requests per minute on Pro?", "DOC-001")
print(json.dumps(example, indent=2))""")

code("""if RUN_RAG_EVAL:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model_rag = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    gen = hf_pipeline("text-generation", model=model_rag, tokenizer=tokenizer)

    rag_answers = []
    for _, row in eval_df.iterrows():
        messages = build_rag_prompt(row["question"], row["doc_id"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = gen(prompt, max_new_tokens=150, do_sample=False)
        output = result[0]["generated_text"][len(prompt):].strip()
        rag_answers.append(output)

    rag_detail, rag_summary = evaluate_qa(rag_answers, eval_df, "RAG_only")
    print(json.dumps(rag_summary, indent=2))
else:
    # Simulate plausible RAG results
    rng = np.random.default_rng(SEED)
    rag_answers = []
    for _, row in eval_df.iterrows():
        gt = row["ground_truth"]
        diff = row["difficulty"]
        if diff == "direct":
            answer = gt  # RAG handles direct lookups well
        elif diff == "inference":
            if rng.random() < 0.75:
                answer = gt
            else:
                answer = gt + " Additionally, this may vary by configuration."  # slight over-generation
        elif diff == "multi-fact":
            if rng.random() < 0.65:
                answer = gt
            else:
                # Miss one fact
                parts = gt.split(". ")
                answer = parts[0] + "." if parts else gt
        else:  # boundary
            if rng.random() < 0.45:
                answer = gt
            else:
                doc = KNOWLEDGE_BASE[row["doc_id"]]["content"]
                answer = "Based on the documentation, " + gt.split(".")[0] + ". However, I'm not entirely certain about the edge case details."
        rag_answers.append(answer)

    rag_detail, rag_summary = evaluate_qa(rag_answers, eval_df, "RAG_only")
    print("RAG (simulated) summary:")
    print(json.dumps(rag_summary, indent=2))
    print("\\nSet RUN_RAG_EVAL = True for real evaluation.")""")

# ══════════════════════════════════════════════
# 8  Strategy B — Build grounded SFT data
# ══════════════════════════════════════════════
md(r"""## 7. Strategy B — Fine-Tuning on Corrected Grounded Answers

### How It Works

1. Build a dataset of (context, question, **corrected** answer) triples
2. The "corrected" answer is a human-reviewed, ground-truth response that demonstrates how to use the context properly
3. Fine-tune the model on these triples so it learns both the grounding behavior AND the domain-specific reasoning

### What "Corrected" Means

The ground-truth answers in our dataset are carefully written to:
- include only facts from the document
- cover all relevant facts (completeness)
- handle boundary cases correctly (say "not mentioned" when appropriate)
- use the right level of specificity

This is what distinguishes fine-tuning from RAG alone: the model learns **how to reason** about the context, not just **that it should look at** the context.
""")

code("""from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


GROUNDED_SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer questions using only the provided context. "
    "If the context does not contain enough information, say so. "
    "Be precise with numbers, limits, and conditions."
)


def to_sft_record(row):
    doc = KNOWLEDGE_BASE[row["doc_id"]]
    context_block = f"--- Context: {doc['title']} ---\\n{doc['content']}\\n---"
    messages = [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context_block}\\n\\nQuestion: {row['question']}"},
        {"role": "assistant", "content": row["ground_truth"]},
    ]
    return {
        "messages": messages,
        "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
    }


from datasets import Dataset
train_records = [to_sft_record(row) for _, row in train_df.iterrows()]
train_dataset = Dataset.from_list(train_records)

train_lengths = [len(tokenizer(r["text"]).input_ids) for r in train_records]

print(f"Fine-tuning examples: {len(train_records)}")
print(f"Token lengths: mean={np.mean(train_lengths):.0f}  max={max(train_lengths)}  min={min(train_lengths)}")

jsonl_path = ARTIFACT_DIR / "grounded_sft.jsonl"
with jsonl_path.open("w", encoding="utf-8") as f:
    for rec in train_records:
        f.write(json.dumps({"messages": rec["messages"]}, ensure_ascii=False) + "\\n")
print(f"Saved: {jsonl_path}")""")

# ══════════════════════════════════════════════
# 9  Fine-tune
# ══════════════════════════════════════════════
md("## 8. Fine-Tune with LoRA")

code("""import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

MAX_SEQ_LENGTH = max(train_lengths) + 32

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    report_to="none",
    seed=SEED,
)

model_ft = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, trust_remote_code=True,
    torch_dtype=(
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    ),
    device_map="auto" if torch.cuda.is_available() else None,
)
model_ft.config.use_cache = False

trainer = SFTTrainer(
    model=model_ft, args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)
print(f"Trainer ready.  Max seq: {MAX_SEQ_LENGTH}  Epochs: {training_args.num_train_epochs}")""")

code("""if RUN_TRAINING:
    result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(result)
    print(f"Adapter saved: {OUTPUT_DIR}")
else:
    print("Training skipped. Set RUN_TRAINING = True to fine-tune.")""")

# ══════════════════════════════════════════════
# 10  Evaluate fine-tuned
# ══════════════════════════════════════════════
md("## 9. Evaluate the Fine-Tuned Model")

code("""if RUN_FT_EVAL:
    from peft import PeftModel
    from transformers import pipeline as hf_pipeline

    ft_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    ft_model = PeftModel.from_pretrained(ft_base, str(OUTPUT_DIR))
    ft_gen = hf_pipeline("text-generation", model=ft_model, tokenizer=tokenizer)

    ft_answers = []
    for _, row in eval_df.iterrows():
        doc = KNOWLEDGE_BASE[row["doc_id"]]
        context_block = f"--- Context: {doc['title']} ---\\n{doc['content']}\\n---"
        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context_block}\\n\\nQuestion: {row['question']}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = ft_gen(prompt, max_new_tokens=150, do_sample=False)
        output = result[0]["generated_text"][len(prompt):].strip()
        ft_answers.append(output)

    ft_detail, ft_summary = evaluate_qa(ft_answers, eval_df, "fine_tuned_grounded")
    print(json.dumps(ft_summary, indent=2))
else:
    # Simulate fine-tuned results: better accuracy and fewer hallucinations
    rng_ft = np.random.default_rng(SEED + 50)
    ft_answers = []
    for _, row in eval_df.iterrows():
        gt = row["ground_truth"]
        diff = row["difficulty"]
        if diff == "direct":
            ft_answers.append(gt)
        elif diff == "inference":
            ft_answers.append(gt if rng_ft.random() < 0.90 else gt.split(".")[0] + ".")
        elif diff == "multi-fact":
            ft_answers.append(gt if rng_ft.random() < 0.85 else gt.rsplit(",", 1)[0] + ".")
        else:  # boundary
            ft_answers.append(gt if rng_ft.random() < 0.75 else "Based on the documentation, " + gt)

    ft_detail, ft_summary = evaluate_qa(ft_answers, eval_df, "fine_tuned_grounded")
    print("Fine-tuned (simulated) summary:")
    print(json.dumps(ft_summary, indent=2))
    print("\\nSet RUN_FT_EVAL = True for real evaluation.")""")

# ══════════════════════════════════════════════
# 11  Comparison
# ══════════════════════════════════════════════
md("## 10. Side-by-Side Comparison")

code("""comparison = pd.DataFrame([rag_summary, ft_summary]).set_index("strategy").T
comparison["delta"] = comparison["fine_tuned_grounded"] - comparison["RAG_only"]
comparison["direction"] = comparison["delta"].apply(
    lambda d: "better" if d > 0 else ("worse" if d < 0 else "same")
)
# Hallucination rate: lower is better
if "hallucination_rate" in comparison.index:
    idx = comparison.index.get_loc("hallucination_rate")
    comparison.iloc[idx, -1] = "better" if comparison.iloc[idx, -2] < 0 else "worse" if comparison.iloc[idx, -2] > 0 else "same"

print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)
print(comparison.to_string())""")

code("""# Visual comparison
metrics = ["factual_accuracy", "groundedness", "completeness", "hallucination_rate"]
rag_vals = [rag_summary[m] for m in metrics]
ft_vals = [ft_summary[m] for m in metrics]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width / 2, rag_vals, width, label="RAG Only", color="#d62728", alpha=0.8)
ax.bar(x + width / 2, ft_vals, width, label="Fine-Tuned + Retrieval", color="#2ca02c", alpha=0.8)

ax.set_ylabel("Score")
ax.set_title("RAG Only vs Fine-Tuned + Retrieval")
ax.set_xticks(x)
ax.set_xticklabels([m.replace("_", "\\n") for m in metrics], fontsize=9)
ax.set_ylim([0, 1.1])
ax.legend()

for i, (rv, fv) in enumerate(zip(rag_vals, ft_vals)):
    ax.text(i - width / 2, rv + 0.02, f"{rv:.0%}", ha="center", fontsize=8)
    ax.text(i + width / 2, fv + 0.02, f"{fv:.0%}", ha="center", fontsize=8)

plt.tight_layout()
plt.show()""")

code("""# Breakdown by difficulty
rag_detail["strategy"] = "RAG_only"
ft_detail["strategy"] = "fine_tuned"
combined = pd.concat([rag_detail, ft_detail], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, metric, title in [
    (axes[0], "factual_accuracy", "Factual Accuracy by Difficulty"),
    (axes[1], "groundedness", "Groundedness by Difficulty"),
]:
    pivot = combined.pivot_table(index="difficulty", columns="strategy", values=metric, aggfunc="mean")
    pivot = pivot.reindex(["direct", "inference", "multi-fact", "boundary"])
    pivot.plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"], alpha=0.8, rot=0)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()""")

# ══════════════════════════════════════════════
# 12  Honest tradeoffs
# ══════════════════════════════════════════════
md(r"""## 11. Honest Tradeoffs

### Where RAG-Only Wins

| Advantage | Why it matters |
|---|---|
| **No training required** | Ship in hours, not days. No GPU budget for training. |
| **Always up-to-date** | Update the knowledge base → answers update immediately. No retraining needed. |
| **Transparent reasoning** | The retrieved context is visible. Debugging is straightforward. |
| **Works with any model** | Swap the LLM without retraining. OpenAI today, local model tomorrow. |
| **Scales to new domains** | Add documents to the index. No new training data required. |

### Where Fine-Tuning Wins

| Advantage | Why it matters |
|---|---|
| **Better boundary-case reasoning** | The model has seen corrected examples of edge-case logic. |
| **More consistent output format** | Trained on well-structured answers → outputs are more predictable. |
| **Reduced hallucination on trained patterns** | The model learns which facts to include and which to omit. |
| **Lower inference cost at scale** | The fine-tuned model can sometimes work with shorter prompts (less context injection needed). |
| **Encodes domain reasoning rules** | "Only Enterprise can do X" is learned from examples, not just read from context. |

### Where Both Struggle

| Problem | RAG-only | Fine-tuned |
|---|---|---|
| **Novel questions** (not in KB) | Retrieves irrelevant docs → bad answer | Still relies on retrieval; no magic for missing info |
| **Contradictory documents** | Presents conflicting info | May overfit to one version |
| **Very long documents** | Context window limits | Same limit at inference time |
| **Rapidly changing facts** | Good — update KB | Bad — needs retraining |
| **Multi-hop reasoning** | Weak unless retriever returns all pieces | Better IF training data included multi-hop examples |

### The Uncomfortable Truth

**Fine-tuning does not eliminate the need for retrieval.** The fine-tuned model still hallucinates without context. Fine-tuning teaches it to USE context better, not to REPLACE context.

**RAG does not eliminate the need for good prompting.** A poorly prompted RAG system still hallucinates or ignores the context.

**Neither solves bad retrieval.** If the retriever returns the wrong document, both strategies fail.
""")

# ══════════════════════════════════════════════
# 13  Decision framework
# ══════════════════════════════════════════════
md(r"""## 12. Decision Framework: Which to Use?

### Start with RAG-Only

RAG-only is the correct default. It works immediately, requires no training data, and stays current with the knowledge base.

### Add Fine-Tuning When

You should consider adding fine-tuning ON TOP of retrieval when:

| Signal | Details |
|---|---|
| **Boundary cases matter** | Your domain has nuanced rules (plan-specific limits, conditional eligibility) that the model gets wrong even with the right context |
| **Format consistency matters** | You need the model to output in a specific structure reliably |
| **You have corrected examples** | You have human-reviewed (question, context, answer) triples — at least 50–100 |
| **Hallucination rate is still too high** | RAG alone gives > 10% hallucination on your eval set |
| **Reasoning patterns repeat** | The same type of inference ("Only Enterprise customers can...") comes up frequently |

### Do Not Fine-Tune When

| Signal | Why not |
|---|---|
| **Knowledge base changes weekly** | Retraining lag means the model is constantly stale |
| **You don't have correction data** | Training on uncorrected model outputs just reinforces errors |
| **RAG already meets your accuracy target** | Fine-tuning adds complexity without value |
| **The retriever is the bottleneck** | Fix retrieval quality first; fine-tuning won't help if the wrong docs arrive |

### The Practical Default

```text
Start here: RAG-only with a good prompt
    │
    Is accuracy sufficient on your eval set?
    ├── YES → Ship it. Monitor. Done.
    └── NO  → Where do errors come from?
              ├── Wrong docs retrieved → Fix retrieval (better embeddings, chunking, re-ranking)
              ├── Right docs but wrong answer → Fine-tune on corrected examples
              └── Missing docs → Add to knowledge base
```
""")

# ══════════════════════════════════════════════
# 14  Combining both
# ══════════════════════════════════════════════
md(r"""## 13. Combining Both: RAG + Fine-Tuning

The production sweet spot is often **fine-tuned model WITH retrieval** — the model is better at using context because of training, and retrieval keeps it grounded in current facts.

### Architecture

```text
User question
    │
    ▼
┌──────────────┐
│  Retriever    │  Find relevant docs (embeddings, BM25, hybrid)
└──────┬───────┘
       │ context
       ▼
┌──────────────────────┐
│  Fine-tuned model    │  Better at: using context, boundary cases,
│  (with LoRA adapter) │  format consistency, avoiding hallucination
└──────┬───────────────┘
       │
       ▼
   Grounded answer
```

### What Each Component Contributes

| Component | Contribution |
|---|---|
| **Retriever** | Current facts, source traceability |
| **Fine-tuning** | Reasoning patterns, format, calibrated uncertainty |
| **Prompt** | Task framing, output constraints |

All three work together. None is sufficient alone.
""")

# ══════════════════════════════════════════════
# 15  Limitations
# ══════════════════════════════════════════════
md(r"""## 14. Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Heuristic evaluation** | Automated metrics are proxies, not ground truth | Use human review or LLM-as-judge for production |
| **Small dataset** | 24 QA pairs is far below production scale | Target 200+ pairs with difficulty diversity |
| **Simulated results** | Demo curves do not reflect real model behavior | Run with `RUN_*_EVAL = True` for actual numbers |
| **Perfect retrieval assumed** | We always retrieve the correct document | Real systems have retrieval errors — evaluate end-to-end |
| **No multi-document questions** | Every question maps to one document | Production systems need multi-hop retrieval and reasoning |
| **No latency comparison** | RAG adds retrieval latency; fine-tuning adds training cost | Benchmark both in your infrastructure |

### The Evaluation Gap

The biggest gap in practice is between **automated grounding scores** and **real user satisfaction**. A response can score 100% on groundedness and still be unhelpful if it is:
- too verbose
- missing the specific detail the user needed
- technically correct but confusingly phrased

**Human evaluation remains necessary for deployment decisions.**
""")

# ══════════════════════════════════════════════
# 16  Save
# ══════════════════════════════════════════════
md("## 15. Save Experiment Log")

code("""log = {
    "timestamp": datetime.now().isoformat(),
    "task": "grounding_vs_fine_tuning",
    "base_model": BASE_MODEL,
    "knowledge_base_docs": len(KNOWLEDGE_BASE),
    "qa_pairs": len(qa_df),
    "train_size": len(train_df),
    "eval_size": len(eval_df),
    "rag_summary": rag_summary,
    "ft_summary": ft_summary,
}

log_path = ARTIFACT_DIR / "grounding_vs_finetuning_log.json"
log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
print(f"Saved: {log_path}")""")

# ══════════════════════════════════════════════
# 17  Key takeaways
# ══════════════════════════════════════════════
md(r"""## 16. Key Takeaways

### What We Compared

- **RAG-only**: retrieve document, inject into prompt, generate answer. No training.
- **Fine-tuned + retrieval**: train on (context, question, corrected answer) triples, then use with retrieval at inference.

### What We Found (General Patterns)

- RAG-only handles **direct lookups** well but struggles with **boundary cases** and **multi-fact reasoning**
- Fine-tuning improves performance on **inference, boundary, and multi-fact** questions most
- Fine-tuning reduces **hallucination rate** because the model learns which facts to include and which to avoid
- Neither strategy eliminates the need for **good retrieval** — wrong docs break both

### Honest Recommendations

1. **Start with RAG-only.** It works immediately and stays current.
2. **Measure before fine-tuning.** Build your eval set first. If RAG already meets your target, ship it.
3. **Fine-tune on corrected examples, not raw model outputs.** The whole point is to teach better behavior.
4. **Fine-tuning teaches context-use, not facts.** The model still needs retrieval. Do not expect it to memorize the knowledge base.
5. **Fix retrieval before fine-tuning.** If the wrong documents arrive, no amount of fine-tuning helps.
6. **Monitor both in production.** Knowledge bases change. Retrain when accuracy drifts.
7. **Human evaluation is not optional.** Automated scores are for iteration; human review is for deployment.
""")


# ══════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).parent / "grounding_vs_fine_tuning.ipynb"
out.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Notebook written: {out}")
print(f"Cells: {len(cells)}")
print(f"Code:  {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"Size: {out.stat().st_size:,} bytes")
