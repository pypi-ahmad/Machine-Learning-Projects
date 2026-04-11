"""Generate Projects 1-10: Beginner Local LLM Apps — complete notebook implementations."""
import json, os

BASE = r"E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\Beginner_Local_LLM_Apps"

def nb(cells):
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.10.0", "file_extension": ".py", "mimetype": "text/x-python"}},
        "nbformat": 4, "nbformat_minor": 4
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.split("\n")}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.split("\n")}

def save(folder, cells):
    d = os.path.join(BASE, folder)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "notebook.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"  ✓ {folder}")

# ─────────────────────────────────────────────
# Project 1 — Local PDF Q&A Tutor
# ─────────────────────────────────────────────
save("01_Local_PDF_QA_Tutor", [
md("""# Project 1 — Local PDF Q&A Tutor
## Chat with PDFs Using Local Embeddings and Ollama

**What you'll learn:**
- Load and parse PDF documents with PyPDF
- Split text into overlapping chunks for better retrieval
- Create embeddings with a local Ollama model
- Store vectors in ChromaDB
- Build a RetrievalQA chain that answers questions with citations

**Stack:** Ollama · LangChain · ChromaDB · PyPDF · Jupyter

**Prerequisites:** `ollama pull nomic-embed-text` and `ollama pull qwen3:8b`

```
PDF → Split into chunks → Embed each chunk → Store in Chroma
                                                       ↓
User question → Embed question → Find similar chunks → Feed to LLM → Answer + citation
```"""),

code("""# Cell 1 — Install dependencies
# !pip install -q langchain langchain-ollama langchain-community chromadb pypdf"""),

md("""## Step 1 — Configure LLM and Embeddings
We use Ollama running at `localhost:11434`. No API keys needed."""),

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings

# Local LLM — adjust model name to whatever you have pulled
llm = ChatOllama(model="qwen3:8b", temperature=0.3)

# Local embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Quick smoke test
vec = embeddings.embed_query("test")
print(f"Embedding dim: {len(vec)} — model is working!")"""),

md("""## Step 2 — Create a Sample PDF (for demo)
We generate a small text file so the notebook is self-contained.
In practice, replace this with any PDF file path."""),

code("""from pathlib import Path
import textwrap

SAMPLE_DIR = Path("sample_data")
SAMPLE_DIR.mkdir(exist_ok=True)

sample_text = textwrap.dedent(\"\"\"
    Machine Learning Fundamentals

    Supervised learning uses labeled data to train models that predict outcomes.
    Common algorithms include linear regression for continuous targets and
    logistic regression for classification. Decision trees split data based on
    feature thresholds, while random forests combine many trees to reduce overfitting.

    Deep Learning Overview

    Neural networks consist of layers of interconnected nodes. Convolutional
    Neural Networks (CNNs) excel at image recognition by learning spatial
    hierarchies of features. Recurrent Neural Networks (RNNs) and their
    variants like LSTMs handle sequential data such as text and time series.

    Transformers and Attention

    The transformer architecture, introduced in "Attention Is All You Need"
    (Vaswani et al., 2017), replaced recurrence with self-attention mechanisms.
    This enables parallel processing of sequences, leading to models like BERT
    for understanding and GPT for generation. Large Language Models (LLMs) are
    scaled-up transformers trained on massive text corpora.

    Retrieval-Augmented Generation (RAG)

    RAG combines a retriever (which finds relevant documents) with a generator
    (which produces answers). This reduces hallucination by grounding the LLM
    in actual source material. Key components include: document chunking,
    embedding models, vector databases, and prompt engineering.
\"\"\")

sample_path = SAMPLE_DIR / "ml_fundamentals.txt"
sample_path.write_text(sample_text, encoding="utf-8")
print(f"Sample document saved to: {sample_path}")"""),

md("""## Step 3 — Load and Chunk the Document
**Chunking strategy:** `RecursiveCharacterTextSplitter` tries to split on
paragraph boundaries first, then sentences, then words.

- `chunk_size=500` — each chunk ≈ 500 chars (roughly a paragraph)
- `chunk_overlap=50` — overlap prevents cutting important context"""),

code("""from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader(str(SAMPLE_DIR / "ml_fundamentals.txt"), encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\\n\\n", "\\n", ". ", " "],
)
chunks = splitter.split_documents(docs)

print(f"Original doc length: {len(docs[0].page_content)} chars")
print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}: {len(chunk.page_content)} chars — {chunk.page_content[:60]}...")"""),

md("""## Step 4 — Create the Vector Store
ChromaDB stores chunks as vectors and lets us do similarity search."""),

code("""from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(SAMPLE_DIR / "chroma_db"),
    collection_name="pdf_qa_tutor",
)

results = vectorstore.similarity_search("What is RAG?", k=2)
print("Top 2 results for 'What is RAG?':")
for i, r in enumerate(results):
    print(f"  [{i+1}] {r.page_content[:100]}...")"""),

md("""## Step 5 — Build the QA Chain"""),

code("""from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"Use the following context to answer the question. If you cannot
find the answer in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer (cite relevant details from the context):\"\"\"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
)
print("QA chain ready!")"""),

md("""## Step 6 — Ask Questions"""),

code("""questions = [
    "What is the difference between CNNs and RNNs?",
    "How does RAG reduce hallucination?",
    "What did the transformer architecture replace?",
]

for q in questions:
    print(f"\\n{'='*60}")
    print(f"Q: {q}")
    result = qa_chain.invoke({"query": q})
    print(f"A: {result['result']}")
    print(f"Sources: {len(result['source_documents'])} chunks used")"""),

md("""## Step 7 — Interactive Q&A Helper"""),

code("""def ask(question: str) -> str:
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    output = f"Answer: {answer}\\n\\nSources ({len(sources)}):\\n"
    for i, src in enumerate(sources):
        output += f"  [{i+1}] {src.page_content[:80]}...\\n"
    return output

print(ask("What are the key components of RAG?"))"""),

md("""## What You Learned
- **PDF loading** — parsing documents into text
- **Chunking** — splitting text with overlap to avoid info loss
- **Embeddings** — converting text to vectors locally
- **Vector store** — ChromaDB for similarity search
- **RetrievalQA** — end-to-end Q&A with citations

## Next Steps
- Try with real PDFs using `PyPDFLoader`
- Experiment with chunk sizes (250 vs 500 vs 1000)
- Add conversation memory (→ Project 18)
- Compare FAISS vs Chroma"""),
])

# ─────────────────────────────────────────────
# Project 2 — Local Markdown Knowledge Bot
# ─────────────────────────────────────────────
save("02_Local_Markdown_Knowledge_Bot", [
md("""# Project 2 — Local Markdown Knowledge Bot
## Query Your Markdown Notes Using LlamaIndex + Ollama

**What you'll learn:**
- Index markdown files with LlamaIndex's `SimpleDirectoryReader`
- Build a `VectorStoreIndex` with local Ollama embeddings
- Query your personal knowledge base in natural language
- Add chat mode with conversation memory

**Stack:** Ollama · LlamaIndex · Jupyter

**Prerequisites:** `ollama pull nomic-embed-text` and `ollama pull qwen3:8b`"""),

code("""# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"""),

md("""## Step 1 — Create Sample Markdown Notes"""),

code("""from pathlib import Path

NOTES_DIR = Path("sample_notes")
NOTES_DIR.mkdir(exist_ok=True)

notes = {
    "python_basics.md": "# Python Basics\\n## Variables\\nPython is dynamically typed. Common types: int, float, str, bool, list, dict.\\n\\n## List Comprehensions\\nConcise syntax: `[x**2 for x in range(10)]`\\n\\n## Functions\\nDefined with `def`. Support default args, *args, **kwargs.\\n",
    "git_cheatsheet.md": "# Git Cheatsheet\\n## Common Commands\\n- `git init` — init new repo\\n- `git add .` — stage all\\n- `git commit -m 'msg'` — commit\\n- `git push origin main` — push\\n\\n## Branching\\n- `git branch feat` — create\\n- `git checkout feat` — switch\\n- `git merge feat` — merge\\n",
    "docker_notes.md": "# Docker Notes\\n## Concepts\\n- **Image** — read-only template\\n- **Container** — running instance\\n- **Dockerfile** — build script\\n- **Volume** — persistent storage\\n\\n## Commands\\n- `docker build -t app .`\\n- `docker run -p 8080:80 app`\\n- `docker compose up`\\n",
    "ml_pipeline.md": "# ML Pipeline\\n## Steps\\n1. Data Collection\\n2. EDA — distributions, missing values\\n3. Feature Engineering\\n4. Model Training\\n5. Evaluation — accuracy, F1, RMSE\\n6. Deployment — API or batch\\n\\n## Pitfalls\\n- Data leakage\\n- Not tracking experiments\\n- Overfitting on small data\\n",
}

for name, content in notes.items():
    (NOTES_DIR / name).write_text(content, encoding="utf-8")
print(f"Created {len(notes)} sample markdown notes in {NOTES_DIR}/")"""),

md("""## Step 2 — Configure LlamaIndex with Ollama"""),

code("""from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
print("LlamaIndex configured with local Ollama.")"""),

md("""## Step 3 — Load and Index"""),

code("""documents = SimpleDirectoryReader(
    input_dir=str(NOTES_DIR), required_exts=[".md"], recursive=True
).load_data()

print(f"Loaded {len(documents)} documents:")
for doc in documents:
    print(f"  - {doc.metadata.get('file_name', '?')}: {len(doc.text)} chars")

index = VectorStoreIndex.from_documents(documents, show_progress=True)
print("\\nVector index built!")"""),

md("""## Step 4 — Query the Knowledge Base"""),

code("""query_engine = index.as_query_engine(similarity_top_k=3)

queries = [
    "How do I create a new Git branch?",
    "What are the steps in a typical ML pipeline?",
    "What is a Docker volume?",
    "How do list comprehensions work in Python?",
]

for q in queries:
    print(f"\\nQ: {q}")
    response = query_engine.query(q)
    print(f"A: {response}")
    print(f"Sources: {[n.metadata.get('file_name', '?') for n in response.source_nodes]}")
    print("-" * 50)"""),

md("""## Step 5 — Chat Mode with Memory"""),

code("""from llama_index.core.chat_engine import CondenseQuestionChatEngine

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine, verbose=True
)

conversation = [
    "What version control tool is covered in my notes?",
    "How do I push changes to the remote?",
    "What about branching?",
]

for msg in conversation:
    print(f"\\nUser: {msg}")
    response = chat_engine.chat(msg)
    print(f"Bot: {response}")"""),

md("""## What You Learned
- **SimpleDirectoryReader** → loading docs from a folder
- **VectorStoreIndex** → automatic chunking, embedding, indexing
- **Query engine** → semantic search + LLM answer generation
- **Chat engine** → multi-turn conversation with context

## Next Steps
- Point to your real Obsidian/Notion export folder
- Add metadata (tags, dates) for filtering
- Try Project 17 (full wiki copilot)"""),
])

# ─────────────────────────────────────────────
# Project 3 — Local Meeting Notes Summarizer
# ─────────────────────────────────────────────
save("03_Local_Meeting_Notes_Summarizer", [
md("""# Project 3 — Local Meeting Notes Summarizer
## Extract Actions, Decisions & Blockers from Transcripts

**What you'll learn:**
- Structured prompting to extract meeting components
- Pydantic models for validated LLM output
- Output parsing and JSON extraction

**Stack:** Ollama · LangChain · Pydantic · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama pydantic"""),

md("""## Step 1 — Sample Transcript"""),

code("""TRANSCRIPT = \"\"\"
Meeting: Q4 Planning Review
Date: 2024-12-15
Attendees: Sarah (PM), Mike (Engineering), Lisa (Design), Tom (QA)

Sarah: Let's start with status. Mike, where are we on the API migration?

Mike: We've completed 70% of the endpoints. Auth module is done, but payments
API is pending — we need DevOps to approve the encryption library by Friday.
That's a blocker.

Sarah: Got it. Lisa, dashboard redesign?

Lisa: Wireframes approved. High-fidelity mockups ready by next Wednesday.
One concern — data viz component needs Mike's new API endpoints.

Tom: QA side — test cases written for completed endpoints. Need staging
deploy by end of week for integration testing.

Sarah: Decisions — prioritize payments API this sprint. Mike, work with
DevOps on library approval. Tom, start testing auth module on staging now.
Reconvene Friday for sync.

Mike: One more — we should add rate limiting. I'll draft a proposal.
Sarah: Good idea, add to backlog for next sprint.
\"\"\"
print(f"Transcript: {len(TRANSCRIPT)} chars")"""),

md("""## Step 2 — Define Output Schema"""),

code("""from pydantic import BaseModel, Field
from typing import List

class ActionItem(BaseModel):
    owner: str = Field(description="Person responsible")
    task: str = Field(description="What needs to be done")
    deadline: str = Field(description="When it's due")

class MeetingSummary(BaseModel):
    title: str
    date: str
    attendees: List[str]
    summary: str = Field(description="2-3 sentence summary")
    key_decisions: List[str]
    action_items: List[ActionItem]
    blockers: List[str]
    follow_up_date: str

print("Schema fields:", list(MeetingSummary.model_fields.keys()))"""),

md("""## Step 3 — Build the Chain"""),

code("""from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import json, re

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are an expert meeting note-taker. Extract structured info.
Return a JSON object with: title, date, attendees (list), summary, key_decisions (list),
action_items (list of {{owner, task, deadline}}), blockers (list), follow_up_date.
Return ONLY valid JSON.\"\"\"),
    ("human", "Transcript:\\n{transcript}"),
])

chain = prompt | llm
print("Chain ready.")"""),

md("""## Step 4 — Run & Parse"""),

code("""result = chain.invoke({"transcript": TRANSCRIPT})

json_match = re.search(r'\\{[\\s\\S]*\\}', result.content)
if json_match:
    parsed = json.loads(json_match.group())
    summary = MeetingSummary(**parsed)

    print(f"Title: {summary.title} — {summary.date}")
    print(f"Attendees: {', '.join(summary.attendees)}")
    print(f"\\nSummary: {summary.summary}")
    print(f"\\nDecisions:")
    for d in summary.key_decisions:
        print(f"  • {d}")
    print(f"\\nAction Items:")
    for a in summary.action_items:
        print(f"  [{a.owner}] {a.task} (by: {a.deadline})")
    print(f"\\nBlockers:")
    for b in summary.blockers:
        print(f"  ⚠ {b}")
    print(f"\\nFollow-up: {summary.follow_up_date}")
else:
    print("Could not parse JSON. Raw:", result.content)"""),

md("""## Step 5 — Reusable Summarizer Function"""),

code("""def summarize_meeting(transcript: str) -> dict:
    result = chain.invoke({"transcript": transcript})
    match = re.search(r'\\{[\\s\\S]*\\}', result.content)
    if match:
        return json.loads(match.group())
    return {"error": "Parse failed", "raw": result.content}

output = summarize_meeting(TRANSCRIPT)
print(json.dumps(output, indent=2))"""),

md("""## What You Learned
- **Structured prompting** — guiding LLM to produce specific format
- **Pydantic validation** — enforcing schema on LLM outputs
- **JSON extraction** — regex parsing from free-text responses
- **Chain composition** — prompt | LLM pipeline

## Next Steps
- Connect to Whisper transcription (→ Project 88)
- Auto-email action items to owners
- Store summaries in a searchable index"""),
])

# ─────────────────────────────────────────────
# Project 4 — Local Resume Rewriter
# ─────────────────────────────────────────────
save("04_Local_Resume_Rewriter", [
md("""# Project 4 — Local Resume Rewriter
## Improve Resume Bullets with Few-Shot Prompting & STAR Method

**What you'll learn:**
- Few-shot prompting to teach style by example
- STAR method (Situation, Task, Action, Result) for impact writing
- Generating multiple rewrite variants

**Stack:** Ollama · LangChain · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama"""),

code("""from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
llm = ChatOllama(model="qwen3:8b", temperature=0.7)"""),

md("""## Step 1 — Few-Shot Examples"""),

code("""examples = [
    {"weak": "Managed a team of developers",
     "strong": "Led cross-functional team of 8 engineers, delivering 3 features on time, reducing sprint overruns by 30%"},
    {"weak": "Improved website performance",
     "strong": "Optimized front-end bundle by 45% and added lazy loading, cutting load time from 4.2s to 1.8s"},
    {"weak": "Worked on data analysis projects",
     "strong": "Built automated ETL pipelines processing 2M+ daily records with Python/Airflow, reducing manual reporting from 8h to 15min"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Weak: {weak}"), ("ai", "Strong: {strong}"),
])

few_shot = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
print(f"Loaded {len(examples)} few-shot examples")"""),

md("""## Step 2 — Build Rewriter Chain"""),

code("""rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"You are an expert resume coach. Rewrite bullets using STAR method.
Rules: Start with action verb, include metrics, show impact not just duties,
keep each bullet to 1-2 lines. Use realistic placeholders (X%, N+) for numbers.\"\"\"),
    few_shot,
    ("human", "Weak: {bullet}"),
])

rewrite_chain = rewriter_prompt | llm"""),

md("""## Step 3 — Rewrite Bullets"""),

code("""weak_bullets = [
    "Responsible for customer support tickets",
    "Did code reviews for the team",
    "Helped with database migration",
    "Participated in agile ceremonies",
    "Created documentation for the API",
]

for bullet in weak_bullets:
    result = rewrite_chain.invoke({"bullet": bullet})
    print(f"❌ Before: {bullet}")
    print(f"✅ After:  {result.content}")
    print("-" * 50)"""),

md("""## Step 4 — Multiple Variants"""),

code("""variant_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 rewrite options with different angles. Number them 1-3. Use strong action verbs and metrics."),
    ("human", "{bullet}"),
])

result = (variant_prompt | llm).invoke({"bullet": "Managed social media accounts for the company"})
print(result.content)"""),

md("""## Step 5 — Full Section Rewriter"""),

code("""def rewrite_section(title: str, bullets: list[str]) -> str:
    output = f"### {title}\\n"
    for b in bullets:
        result = rewrite_chain.invoke({"bullet": b})
        output += f"• {result.content.strip()}\\n"
    return output

experience = {
    "Software Engineer — Acme Corp (2022-2024)": [
        "Wrote Python scripts for data processing",
        "Fixed bugs in production codebase",
        "Attended daily standups",
    ],
}

for title, bullets in experience.items():
    print(rewrite_section(title, bullets))"""),

md("""## What You Learned
- **Few-shot prompting** — teaching by example for consistent style
- **STAR method** — structured impact-driven writing
- **Variant generation** — multiple options for selection

## Next Steps
- Chain with Project 5 (cover letters) for full job app toolkit
- Add JD keyword matching
- Score bullets for ATS density"""),
])

# ─────────────────────────────────────────────
# Project 5 — Local Cover Letter Generator
# ─────────────────────────────────────────────
save("05_Local_Cover_Letter_Generator", [
md("""# Project 5 — Local Cover Letter Generator
## Generate Tailored Cover Letters from JD + Resume

**What you'll learn:**
- Multi-input prompt design (JD + resume → letter)
- Tone control via prompt parameters
- Keyword match analysis for ATS awareness

**Stack:** Ollama · LangChain · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama"""),

code("""from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
llm = ChatOllama(model="qwen3:8b", temperature=0.6)"""),

md("""## Step 1 — Sample Inputs"""),

code("""JOB_DESC = \"\"\"Senior Data Engineer — TechCorp
Looking for a Senior Data Engineer to build scalable data pipelines.
Requirements: 5+ years, Python, SQL, Spark, Airflow, CI/CD, communication skills.\"\"\"

RESUME = \"\"\"Alex Chen | Data Engineer at DataFlow Inc (3 years)
Skills: Python, SQL, Spark, Airflow, Docker, PostgreSQL, AWS
Achievements: Built ETL processing 5M daily records, cut costs 35% via Spark optimization,
designed data quality monitoring, led cloud migration.\"\"\"

print("Inputs loaded.")"""),

md("""## Step 2 — Cover Letter Chain"""),

code("""prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Write a compelling cover letter. 3-4 paragraphs: hook, experience match,
achievement spotlight, closing. Match tone to company. Reference specific JD requirements.
Highlight 2-3 achievements. No generic fillers. Under 350 words. Sound human.\"\"\"),
    ("human", "Job Description:\\n{jd}\\n\\nResume:\\n{resume}\\n\\nTone: {tone}"),
])

chain = prompt | llm"""),

md("""## Step 3 — Generate"""),

code("""result = chain.invoke({"jd": JOB_DESC, "resume": RESUME, "tone": "professional but approachable"})
print(result.content)"""),

md("""## Step 4 — Multiple Tones"""),

code("""for tone in ["formal", "confident and direct", "conversational"]:
    result = chain.invoke({"jd": JOB_DESC, "resume": RESUME, "tone": tone})
    print(f"\\n{'='*50}\\nTone: {tone}\\n{'='*50}")
    print(result.content[:400] + "...\\n")"""),

md("""## Step 5 — Keyword Match Analysis"""),

code("""import re

def keyword_match(jd: str, letter: str) -> dict:
    stops = {'the','and','for','with','you','are','will','have','this','that','from'}
    jd_kw = {w for w in re.findall(r'\\b[a-zA-Z]{3,}\\b', jd.lower()) if w not in stops}
    lt_kw = {w for w in re.findall(r'\\b[a-zA-Z]{3,}\\b', letter.lower()) if w not in stops}
    return {"matched": sorted(jd_kw & lt_kw), "missed": sorted(jd_kw - lt_kw),
            "coverage": f"{len(jd_kw & lt_kw)}/{len(jd_kw)}"}

analysis = keyword_match(JOB_DESC, result.content)
print(f"Coverage: {analysis['coverage']}")
print(f"Matched: {', '.join(analysis['matched'][:10])}")
print(f"Missed: {', '.join(analysis['missed'][:10])}")"""),

md("""## What You Learned
- **Multi-input prompting** — combining JD + resume
- **Tone control** — varying style via parameters
- **Keyword analysis** — basic ATS awareness

## Next Steps: Chain with Project 4 for full application prep"""),
])

# ─────────────────────────────────────────────
# Project 6 — Local Email Reply Assistant
# ─────────────────────────────────────────────
save("06_Local_Email_Reply_Assistant", [
md("""# Project 6 — Local Email Reply Assistant
## Classify Intent and Draft Replies with Structured Output

**What you'll learn:**
- Classify text into categories using an LLM
- Pydantic models for structured output parsing
- Two-step chains: classify → generate

**Stack:** Ollama · LangChain · Pydantic · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama pydantic"""),

code("""from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
import json, re

llm = ChatOllama(model="qwen3:8b", temperature=0.3)"""),

md("""## Step 1 — Classification Schema"""),

code("""class EmailClassification(BaseModel):
    intent: Literal["question","complaint","request","info","meeting","urgent"] = Field(description="Primary intent")
    sentiment: Literal["positive","neutral","negative"] = Field(description="Tone")
    priority: Literal["low","medium","high"] = Field(description="Priority level")
    summary: str = Field(description="One-sentence summary")
    key_points: list[str] = Field(description="Main points to address")

print("Schema:", list(EmailClassification.model_fields.keys()))"""),

md("""## Step 2 — Sample Emails"""),

code("""EMAILS = [
    {"from": "jane@client.com", "subject": "Urgent: API 500 errors",
     "body": "Since this morning our integration returns 500 errors on /orders. Blocking checkout. Tried clearing cache and checking keys. Issue persists."},
    {"from": "tom@partner.com", "subject": "Pricing question",
     "body": "Confused about Pro vs Enterprise. Does Pro include API? Rate limit difference? Enterprise trial available?"},
    {"from": "sarah@bigcorp.com", "subject": "Partnership meeting",
     "body": "Would like to schedule 30min call about integrating your doc processing into our platform. Next Tue/Wed afternoon work?"},
]"""),

md("""## Step 3 — Classify"""),

code("""classify_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify email. Return JSON with: intent, sentiment, priority, summary, key_points. ONLY valid JSON."),
    ("human", "From: {sender}\\nSubject: {subject}\\n\\n{body}"),
])

classify_chain = classify_prompt | llm

for email in EMAILS:
    result = classify_chain.invoke({"sender": email["from"], "subject": email["subject"], "body": email["body"]})
    match = re.search(r'\\{[\\s\\S]*\\}', result.content)
    if match:
        cls = EmailClassification(**json.loads(match.group()))
        print(f"\\n{email['subject']}: intent={cls.intent}, priority={cls.priority}, sentiment={cls.sentiment}")
        print(f"  Summary: {cls.summary}")
    else:
        print(f"  Parse error for {email['subject']}")"""),

md("""## Step 4 — Draft Replies"""),

code("""reply_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Draft professional reply based on classification.
- complaint/urgent: Acknowledge, empathize, promise action, give timeline
- question: Answer directly, offer clarification
- meeting: Suggest times, confirm interest
Keep under 150 words. Professional but warm.\"\"\"),
    ("human", "From: {sender}\\nSubject: {subject}\\n{body}\\n\\nClassification: {classification}"),
])

reply_chain = reply_prompt | llm

for email in EMAILS:
    cls_result = classify_chain.invoke({"sender": email["from"], "subject": email["subject"], "body": email["body"]})
    match = re.search(r'\\{[\\s\\S]*\\}', cls_result.content)
    if match:
        cls = EmailClassification(**json.loads(match.group()))
        reply = reply_chain.invoke({**email, "sender": email["from"], "classification": cls.model_dump_json()})
        print(f"\\n{'='*50}\\nReply to: {email['subject']} [{cls.intent}/{cls.priority}]\\n{'='*50}")
        print(reply.content)"""),

md("""## What You Learned
- **Intent classification** — categorizing text via LLM
- **Pydantic structured output** — enforcing schema
- **Two-step chains** — classify then generate
- **Tone-appropriate replies** — context-aware generation"""),
])

# ─────────────────────────────────────────────
# Project 7 — Local Research Paper Explainer
# ─────────────────────────────────────────────
save("07_Local_Research_Paper_Explainer", [
md("""# Project 7 — Local Research Paper Explainer
## Explain Research Papers in Plain English

**What you'll learn:**
- Section-aware document summarization
- Tree summarization for long documents
- Multi-perspective explanations (ELI5, technical, blog)

**Stack:** Ollama · LlamaIndex · Jupyter"""),

code("""# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"""),

code("""from llama_index.core import Settings, Document, VectorStoreIndex, SummaryIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")"""),

md("""## Step 1 — Simulated Paper Sections"""),

code("""PAPER = {
    "title": "Attention Is All You Need — Simplified",
    "abstract": "We propose the Transformer, relying entirely on attention mechanisms instead of recurrence. Allows significantly more parallelization and reaches new SOTA in translation after 12h on 8 GPUs.",
    "introduction": "RNNs (LSTM, GRU) dominate sequence modeling but prevent parallelization. Attention mechanisms model dependencies regardless of distance. We propose the Transformer — no recurrence, all attention.",
    "methodology": "Encoder-decoder with multi-head attention. Three attention types: encoder self-attention, masked decoder self-attention, encoder-decoder cross-attention. Positional encoding via sinusoidal functions.",
    "results": "WMT 2014 En-De: 28.4 BLEU (new SOTA, +2 over previous best). En-Fr: 41.0 BLEU. Training: 3.5 days on 8 P100 GPUs — fraction of competing methods.",
    "conclusion": "First sequence model based entirely on attention. Trains faster than recurrent/convolutional models. Plan to extend to images, audio, video.",
}

documents = [Document(text=v, metadata={"section": k}) for k, v in PAPER.items() if k != "title"]
print(f"Paper: {PAPER['title']} — {len(documents)} sections")"""),

md("""## Step 2 — Multi-Perspective Explanations"""),

code("""index = VectorStoreIndex.from_documents(documents)
engine = index.as_query_engine(similarity_top_k=3)

perspectives = [
    ("ELI5", "Explain this paper as if I'm 5 years old."),
    ("Key Innovation", "What is the single most important innovation?"),
    ("How It Works", "Explain the methodology step by step simply."),
    ("Why It Matters", "Why are the results significant?"),
]

for label, query in perspectives:
    print(f"\\n[{label}]")
    print(engine.query(query))
    print("-" * 40)"""),

md("""## Step 3 — Full Paper Summary"""),

code("""summary_index = SummaryIndex.from_documents(documents)
summary_engine = summary_index.as_query_engine(response_mode="tree_summarize")

response = summary_engine.query(
    "Structured summary: 1) Main contribution, 2) How it works (3 bullets), "
    "3) Key results with numbers, 4) Why this matters"
)
print(response)"""),

md("""## Step 4 — Blog Post Version"""),

code("""from langchain_ollama import ChatOllama as LC_Chat
from langchain.prompts import ChatPromptTemplate

blog_prompt = ChatPromptTemplate.from_messages([
    ("system", "Turn academic papers into 300-word accessible blog posts."),
    ("human", "Title: {title}\\nAbstract: {abstract}\\nMethod: {method}\\nResults: {results}"),
])

result = (blog_prompt | LC_Chat(model="qwen3:8b", temperature=0.5)).invoke({
    "title": PAPER["title"], "abstract": PAPER["abstract"],
    "method": PAPER["methodology"], "results": PAPER["results"],
})
print(result.content)"""),

md("""## What You Learned
- **Section-aware indexing** — metadata for document structure
- **VectorStoreIndex vs SummaryIndex** — retrieval vs full summarization
- **Multi-perspective prompting** — ELI5, technical, blog formats

## Next Steps: Load real PDFs, index paper collections (→ Project 13)"""),
])

# ─────────────────────────────────────────────
# Project 8 — Local Blog-to-Thread Converter
# ─────────────────────────────────────────────
save("08_Local_Blog_to_Thread_Converter", [
md("""# Project 8 — Local Blog-to-Thread Converter
## Repurpose Blog Posts into Threads, LinkedIn Posts & Newsletters

**What you'll learn:**
- Multi-format content generation from one source
- Platform-specific constraints (character limits, tone, formatting)
- Batch repurposing pipeline

**Stack:** Ollama · LangChain · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama"""),

code("""from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
llm = ChatOllama(model="qwen3:8b", temperature=0.6)"""),

md("""## Step 1 — Sample Blog Post"""),

code("""BLOG = \"\"\"# Why RAG Is Eating the AI World

RAG has become the default architecture for production AI apps.

## The Problem with Pure LLMs
LLMs hallucinate, have knowledge cutoffs, can't access your data. Fine-tuning is expensive.

## Enter RAG
1. Index documents (chunk, embed, store)
2. Find relevant chunks for user questions
3. Feed chunks as context to LLM
4. Generate grounded answers

## Why It's Winning
- Cheaper than fine-tuning
- Always fresh — update index anytime
- Auditable — see which sources informed answers
- Domain-flexible

## Challenges
Chunking matters. Retrieval quality determines answer quality. Must evaluate groundedness.

## What's Next
Agentic RAG, multi-hop reasoning, hybrid retrieval.
\"\"\"
print(f"Blog: {len(BLOG.split())} words")"""),

md("""## Step 2 — Twitter/X Thread"""),

code("""thread_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert to Twitter thread. 6-10 tweets, each <280 chars. Hook first, CTA last. Number: 1/, 2/, etc."),
    ("human", "{blog}"),
])
print((thread_prompt | llm).invoke({"blog": BLOG}).content)"""),

md("""## Step 3 — LinkedIn Post"""),

code("""linkedin_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert to LinkedIn post. 150-250 words. Hook opener, short paragraphs, end with question. 3-5 hashtags."),
    ("human", "{blog}"),
])
print((linkedin_prompt | llm).invoke({"blog": BLOG}).content)"""),

md("""## Step 4 — Email Newsletter"""),

code("""email_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert to email newsletter. Include SUBJECT, PREVIEW, BODY. 200-300 words, 2-3 takeaways as bullets, CTA at end."),
    ("human", "{blog}"),
])
print((email_prompt | llm).invoke({"blog": BLOG}).content)"""),

md("""## Step 5 — All-in-One Repurposer"""),

code("""def repurpose(blog: str) -> dict:
    prompts = {"twitter": thread_prompt, "linkedin": linkedin_prompt, "email": email_prompt}
    results = {}
    for fmt, p in prompts.items():
        results[fmt] = (p | llm).invoke({"blog": blog}).content
        print(f"  ✓ {fmt}")
    return results

all_content = repurpose(BLOG)
print(f"Generated {len(all_content)} formats from 1 blog post.")"""),

md("""## What You Learned
- **Multi-format generation** — one input, multiple outputs
- **Platform constraints** — character limits, tone, structure
- **Batch pipelines** — reusable content repurposing"""),
])

# ─────────────────────────────────────────────
# Project 9 — Local Study Notes Generator
# ─────────────────────────────────────────────
save("09_Local_Study_Notes_Generator", [
md("""# Project 9 — Local Study Notes Generator
## Turn Raw Text into Notes, Flashcards & Quizzes

**What you'll learn:**
- Multi-format educational content generation
- Flashcard creation from source text
- Quiz generation with answer keys
- Progressive summarization

**Stack:** Ollama · LangChain · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama"""),

code("""from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import json, re
llm = ChatOllama(model="qwen3:8b", temperature=0.4)"""),

md("""## Step 1 — Source Material"""),

code("""SOURCE = \"\"\"Operating Systems Concepts

A process is a program in execution with its own memory space and program counter.
The OS manages processes via scheduling: Round Robin, Priority Scheduling, FCFS.

Memory management: allocating/deallocating memory. Virtual memory uses disk as RAM extension.
Paging divides memory into fixed frames. Segmentation uses variable-size logical segments.

File systems organize storage data. Types: FAT32, NTFS (Win), ext4 (Linux), APFS (Mac).
Operations: create, read, write, delete — managed through system calls.

Concurrency: multiple processes/threads executing simultaneously. Challenges: race conditions
(outcome depends on timing), deadlocks (processes wait indefinitely), synchronization via
mutexes, semaphores, monitors.
\"\"\"
print(f"Source: {len(SOURCE.split())} words")"""),

md("""## Step 2 — Structured Notes"""),

code("""notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create study notes: ## Topic, bullet points with **bold** key terms, Key Terms glossary at end."),
    ("human", "{text}"),
])
print((notes_prompt | llm).invoke({"text": SOURCE}).content)"""),

md("""## Step 3 — Flashcards"""),

code("""fc_prompt = ChatPromptTemplate.from_messages([
    ("system", 'Create 8 flashcards as JSON array. Each: {"front":"question","back":"answer","difficulty":"easy|medium|hard"}. ONLY JSON.'),
    ("human", "{text}"),
])

result = (fc_prompt | llm).invoke({"text": SOURCE}).content
match = re.search(r'\\[[\\s\\S]*\\]', result)
if match:
    cards = json.loads(match.group())
    for i, c in enumerate(cards, 1):
        print(f"Card {i} [{c.get('difficulty','?')}] Q: {c['front']}")
        print(f"  A: {c['back']}\\n")"""),

md("""## Step 4 — Multiple Choice Quiz"""),

code("""quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", 'Create 5 MCQ as JSON array. Each: {"question":"...","options":["A)...","B)...","C)...","D)..."],"correct":"A","explanation":"..."}. ONLY JSON.'),
    ("human", "{text}"),
])

result = (quiz_prompt | llm).invoke({"text": SOURCE}).content
match = re.search(r'\\[[\\s\\S]*\\]', result)
if match:
    quiz = json.loads(match.group())
    for i, q in enumerate(quiz, 1):
        print(f"Q{i}: {q['question']}")
        for o in q['options']: print(f"   {o}")
        print(f"   ✓ {q['correct']}: {q['explanation']}\\n")"""),

md("""## Step 5 — Progressive Summarization"""),

code("""for level, instr in [("Detailed","150-200 words, all key concepts"), ("Concise","50-75 words, critical points only"), ("One-liner","One sentence under 25 words")]:
    p = ChatPromptTemplate.from_messages([("system", instr), ("human", "{text}")])
    print(f"[{level}] {(p | llm).invoke({'text': SOURCE}).content}\\n")"""),

md("""## What You Learned
- **Multi-format generation** — notes, flashcards, quizzes from one source
- **JSON structured output** — extracting structured data
- **Progressive summarization** — adjusting detail level"""),
])

# ─────────────────────────────────────────────
# Project 10 — Local Code Explainer
# ─────────────────────────────────────────────
save("10_Local_Code_Explainer", [
md("""# Project 10 — Local Code Explainer
## Explain Code Snippets and Detect Issues

**What you'll learn:**
- Code analysis with LLMs (explanation, review, improvement)
- Multi-language code understanding
- Basic security and performance issue detection

**Stack:** Ollama · LangChain · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama"""),

code("""from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
llm = ChatOllama(model="qwen3:8b", temperature=0.2)"""),

md("""## Step 1 — Code Explanation Engine"""),

code("""explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "Explain code: 1) Purpose (1 sentence), 2) Line-by-line walkthrough, 3) Key concepts, 4) Example output."),
    ("human", "Explain this {language} code:\\n```{language}\\n{code}\\n```"),
])

explain_chain = explain_prompt | llm"""),

md("""## Step 2 — Explain Different Snippets"""),

code("""snippets = [
    {"language": "python", "code": \"\"\"from collections import Counter
def most_common_words(text, n=5):
    words = text.lower().split()
    words = [w.strip('.,!?') for w in words]
    return Counter(words).most_common(n)\"\"\"},
    {"language": "python", "code": \"\"\"def fibonacci(n, memo={}):
    if n in memo: return memo[n]
    if n <= 1: return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]\"\"\"},
    {"language": "sql", "code": \"\"\"SELECT department, COUNT(*) as cnt, AVG(salary) as avg_sal
FROM employees WHERE hire_date >= '2023-01-01'
GROUP BY department HAVING COUNT(*) > 5
ORDER BY avg_sal DESC;\"\"\"},
]

for s in snippets:
    print(f"\\n{'='*50}\\n[{s['language']}]\\n{'='*50}")
    print(explain_chain.invoke(s).content)"""),

md("""## Step 3 — Issue Detector"""),

code("""review_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Review for: Bugs, Security issues, Performance problems, Style issues.
For each: severity (Critical/Warning/Info), what's wrong, how to fix.\"\"\"),
    ("human", "Review this {language} code:\\n```{language}\\n{code}\\n```"),
])

buggy = {"language": "python", "code": \"\"\"import os
def read_user_file(filename):
    path = "/data/" + filename  # path traversal risk
    with open(path) as f: return f.read()

def process_items(items):
    total = 0
    for i in range(len(items)):
        total += items[i]['price']
    avg = total / len(items)  # ZeroDivisionError if empty
    return avg

DB_PASSWORD = "admin123"  # hardcoded secret
\"\"\"}

print((review_prompt | llm).invoke(buggy).content)"""),

md("""## Step 4 — Code Improver"""),

code("""improve_prompt = ChatPromptTemplate.from_messages([
    ("system", "Show improved version with comments explaining each change. Focus on readability, Pythonic patterns, error handling."),
    ("human", "Improve:\\n```{language}\\n{code}\\n```"),
])

print((improve_prompt | llm).invoke(buggy).content)"""),

md("""## Step 5 — All-in-One Analysis"""),

code("""def analyze_code(code: str, lang: str = "python") -> dict:
    d = {"code": code, "language": lang}
    return {k: (p | llm).invoke(d).content for k, p in
            [("explain", explain_prompt), ("review", review_prompt), ("improve", improve_prompt)]}

result = analyze_code("def search(data, t):\\n    for i in range(len(data)):\\n        if data[i]==t: return i\\n    return -1")
for section, content in result.items():
    print(f"\\n[{section.upper()}]\\n{content[:300]}...")"""),

md("""## What You Learned
- **Code explanation** — line-by-line walkthroughs
- **Issue detection** — security, bugs, performance
- **Code improvement** — suggesting better patterns
- **Multi-language** — same prompts across languages

## Next Steps: Build a coding copilot (→ Project 91), test generator (→ 92), PR reviewer (→ 93)"""),
])

print("\n✅ Batch 1 complete — Projects 1-10 (Beginner Local LLM Apps)")
