"""Build notebook 11 — Local Website FAQ Bot."""
import json, os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "100_Local_AI_Projects", "Local_RAG", "11_Local_Website_FAQ_Bot", "notebook.ipynb")

cells = []

def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})

def code(source: str):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.split("\n")})


# ── Title ────────────────────────────────────────────────────────────────────
md("""# Project 11 — Local Website FAQ Bot

## Ingest One Website and Answer Questions Over It

**Goal:** Build a local RAG-powered FAQ bot that ingests website pages,
indexes them in a vector store, and answers customer questions with
source-page citations — all running locally via Ollama.

**Stack:** Ollama · LangChain · ChromaDB · BeautifulSoup · Jupyter

```
Website pages ──► BeautifulSoup ──► clean text ──► TextSplitter ──► chunks
                                                                      │
                                                            Ollama Embeddings
                                                                      │
                                                                      ▼
                                                                  ChromaDB
                                                                      │
User question ──► embed ──► similarity search ──► top chunks ──► LLM ──► Answer
```

### What You'll Learn

1. Simulate website content ingestion (pages with metadata)
2. Clean noisy web content (nav bars, footers, boilerplate)
3. Chunk HTML-extracted text intelligently with overlap
4. Build a ChromaDB vector store with page-level metadata
5. Create a domain-specific FAQ prompt with graceful refusal
6. Test retrieval accuracy and out-of-scope handling
7. Add metadata-filtered search for targeted retrieval

### Prerequisites

- **Ollama** installed and running (`ollama serve`)
- Models pulled: `ollama pull nomic-embed-text` and `ollama pull qwen3:8b`
- Python 3.9+""")

# ── Install ──────────────────────────────────────────────────────────────────
code("""# Install dependencies (uncomment and run once)
# !pip install -q langchain langchain-ollama langchain-community langchain-text-splitters chromadb beautifulsoup4""")

# ── Step 1 ───────────────────────────────────────────────────────────────────
md("""## Step 1 — Verify Ollama Is Running

Before we do anything, let's confirm Ollama is reachable and the required
models are available. If this cell fails, start Ollama with `ollama serve`.""")

code("""import requests, sys

OLLAMA_BASE = "http://localhost:11434"

try:
    r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"✅ Ollama is running — {len(models)} model(s) available")
    for m in models:
        print(f"   • {m}")
except Exception as e:
    print(f"❌ Cannot reach Ollama at {OLLAMA_BASE}: {e}")
    print("\\nFix: start Ollama and run:")
    print("  ollama pull qwen3:8b")
    print("  ollama pull nomic-embed-text")""")

# ── Step 2 ───────────────────────────────────────────────────────────────────
md("""## Step 2 — Configure LLM and Embeddings

We use two Ollama models:
- **nomic-embed-text** — converts text into dense vectors for similarity search
- **qwen3:8b** — instruction-following LLM that generates grounded FAQ answers

Both run entirely on your local machine. No API keys, no cloud calls.""")

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"

llm = ChatOllama(model=LLM_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# Quick smoke test
vec = embeddings.embed_query("test")
print(f"✅ Embedding model ready — dimension: {len(vec)}")

resp = llm.invoke("Say 'hello' in one word.")
print(f"✅ LLM ready — response: {resp.content[:120]}")""")

# ── Step 3 ───────────────────────────────────────────────────────────────────
md("""## Step 3 — Create Simulated Website Content

In a real project you would use a web crawler (like `WebBaseLoader` from LangChain,
or `requests` + `BeautifulSoup`) to fetch actual web pages. Here we simulate a
complete SaaS company website with **6 pages** of realistic content.

Each page has:
- `url` — the page path on the site
- `title` — the page title
- `content` — the extracted body text
- `category` — a semantic tag for filtering

This simulates what you'd get after cleaning HTML with BeautifulSoup.""")

code("""from pathlib import Path
import json

data_dir = Path("sample_data")
data_dir.mkdir(exist_ok=True)

pages = [
    {
        "url": "/pricing",
        "title": "Pricing",
        "category": "billing",
        "content": (
            "Our Pricing Plans:\\n"
            "- Starter: $9/mo — 1 user, 5GB storage, email support\\n"
            "- Professional: $29/mo — 5 users, 50GB storage, priority support, API access\\n"
            "- Enterprise: $99/mo — Unlimited users, 500GB, dedicated support, SSO, audit logs\\n\\n"
            "All plans include a 14-day free trial. Annual billing saves 20%.\\n"
            "Students and non-profits get a 50% discount on all plans.\\n"
            "Custom plans are available for teams larger than 100 users."
        ),
    },
    {
        "url": "/features",
        "title": "Features",
        "category": "product",
        "content": (
            "Key Features:\\n"
            "- Real-time Collaboration: Edit documents simultaneously with your team.\\n"
            "- AI-Powered Search: Find any document in seconds with semantic search.\\n"
            "- Version History: Track all changes with unlimited version history.\\n"
            "- Integrations: Connect with Slack, Jira, GitHub, Google Drive, and 50+ other tools.\\n"
            "- Security: SOC 2 Type II certified, end-to-end encryption at rest and in transit.\\n"
            "- API Access: RESTful API with comprehensive documentation and SDKs in Python, JS, and Go.\\n"
            "- Offline Mode: Work without internet and sync changes when reconnected.\\n"
            "- Mobile Apps: Native iOS and Android apps with full editing capability."
        ),
    },
    {
        "url": "/faq",
        "title": "FAQ",
        "category": "support",
        "content": (
            "Frequently Asked Questions:\\n\\n"
            "Q: Can I cancel anytime?\\n"
            "A: Yes, you can cancel your subscription at any time. No cancellation fees.\\n\\n"
            "Q: Do you offer refunds?\\n"
            "A: We offer a full refund within the first 30 days of any paid plan.\\n\\n"
            "Q: Is my data secure?\\n"
            "A: Yes. We use AES-256 encryption and are SOC 2 Type II certified.\\n\\n"
            "Q: Can I export my data?\\n"
            "A: Yes, you can export all your data in standard formats (CSV, JSON, PDF).\\n\\n"
            "Q: Do you support single sign-on (SSO)?\\n"
            "A: SSO is available on the Enterprise plan via SAML 2.0 and OpenID Connect.\\n\\n"
            "Q: Is there a free plan?\\n"
            "A: We don't have a permanent free plan, but all paid plans include a 14-day free trial."
        ),
    },
    {
        "url": "/about",
        "title": "About Us",
        "category": "company",
        "content": (
            "About TechDocs Inc.\\n\\n"
            "Founded in 2020, TechDocs helps teams organize and search their documentation "
            "efficiently. Our platform serves over 10,000 companies worldwide, from startups "
            "to Fortune 500 enterprises. Headquartered in San Francisco with remote teams "
            "across 15 countries.\\n\\n"
            "Our mission is to make knowledge accessible to every team member, everywhere.\\n\\n"
            "Leadership:\\n"
            "- CEO: Sarah Chen — previously VP Engineering at Dropbox\\n"
            "- CTO: James Park — co-founder of three YC startups\\n"
            "- VP Product: Maria Rodriguez — 12 years at Google Docs team"
        ),
    },
    {
        "url": "/security",
        "title": "Security",
        "category": "product",
        "content": (
            "Security & Compliance:\\n\\n"
            "Data Encryption: AES-256 at rest, TLS 1.3 in transit.\\n"
            "Certifications: SOC 2 Type II, ISO 27001, GDPR compliant, HIPAA ready (Enterprise).\\n"
            "Access Controls: Role-based access (RBAC), multi-factor authentication (MFA), SSO.\\n"
            "Audit Logging: Full audit trail of all user actions, available on Enterprise plan.\\n"
            "Data Residency: Choose data storage region (US, EU, Asia-Pacific).\\n"
            "Penetration Testing: Annual third-party pen tests, bug bounty program.\\n"
            "Backup: Daily automated backups with 30-day retention. Point-in-time recovery available."
        ),
    },
    {
        "url": "/getting-started",
        "title": "Getting Started",
        "category": "support",
        "content": (
            "Getting Started Guide:\\n\\n"
            "Step 1: Sign up at techdocs.com/signup — takes under 2 minutes.\\n"
            "Step 2: Create your first workspace and invite team members.\\n"
            "Step 3: Import existing docs from Google Drive, Notion, or Confluence.\\n"
            "Step 4: Organize with folders and tags for easy navigation.\\n"
            "Step 5: Enable integrations with your existing tools.\\n\\n"
            "Need help? Our onboarding team offers free 30-minute setup calls.\\n"
            "Contact: onboarding@techdocs.com or use the in-app chat widget."
        ),
    },
]

(data_dir / "website_pages.json").write_text(json.dumps(pages, indent=2))
print(f"✅ Created {len(pages)} simulated website pages:")
for p in pages:
    print(f"   {p['url']:20s} ({p['category']:10s}) — {len(p['content']):3d} chars")""")

# ── Step 4 ───────────────────────────────────────────────────────────────────
md("""## Step 4 — Simulate HTML Cleaning with BeautifulSoup

In a real website ingestion pipeline, raw HTML contains navigation bars,
footers, scripts, ads, and other **boilerplate** that hurts retrieval quality.
BeautifulSoup strips these elements and extracts only meaningful body text.

Here we demonstrate the cleaning pipeline with a sample HTML snippet.""")

code("""from bs4 import BeautifulSoup
import re

# Demo: cleaning a noisy HTML page
sample_html = \"\"\"
<html>
<head><title>Pricing - TechDocs</title></head>
<body>
  <nav>Home | Features | Pricing | About</nav>
  <script>var analytics = true;</script>
  <main>
    <h1>Our Pricing Plans</h1>
    <p>Starter: $9/mo — great for individuals.</p>
    <p>Professional: $29/mo — best for small teams.</p>
  </main>
  <footer>© 2024 TechDocs Inc. | Privacy | Terms</footer>
</body>
</html>
\"\"\"


def clean_html(html: str) -> str:
    \"\"\"Extract meaningful text from HTML, removing nav, scripts, and footers.\"\"\"
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["nav", "footer", "script", "style", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\\n", strip=True)
    text = re.sub(r"\\n{3,}", "\\n\\n", text)
    return text.strip()


cleaned = clean_html(sample_html)
print("Raw HTML → Cleaned text:")
print(cleaned)
print(f"\\n(Removed nav, footer, script — kept only {len(cleaned)} chars of content)")""")

# ── Step 5 ───────────────────────────────────────────────────────────────────
md("""## Step 5 — Convert Pages to LangChain Documents

Each website page becomes a `Document` with:
- `page_content` — the clean text from the page body
- `metadata` — URL, title, category, source tag

Metadata is critical for:
- **Citations** — telling the user which page the answer came from
- **Filtering** — narrowing search to specific page categories
- **Debugging** — tracing retrieval issues back to source pages""")

code("""from langchain.schema import Document

documents = []
for page in pages:
    doc = Document(
        page_content=page["content"].strip(),
        metadata={
            "url": page["url"],
            "title": page["title"],
            "category": page["category"],
            "source": "techdocs.com",
        },
    )
    documents.append(doc)

print(f"✅ Created {len(documents)} Document objects:")
for doc in documents:
    print(f"   [{doc.metadata['title']:18s}] {len(doc.page_content):3d} chars | category={doc.metadata['category']}")""")

# ── Step 6 ───────────────────────────────────────────────────────────────────
md("""## Step 6 — Chunk Documents for Retrieval

**Why chunking matters:** Embedding models perform best on focused,
moderate-length text. Chunking splits each page into pieces of ~400
characters with 50-character overlap at the boundaries.

`RecursiveCharacterTextSplitter` tries to break on natural boundaries:
paragraph breaks → lines → sentences → words.""")

code("""from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\\n\\n", "\\n", ". ", " "],
)
chunks = splitter.split_documents(documents)

total_chars = sum(len(d.page_content) for d in documents)
print(f"Split {len(documents)} pages ({total_chars:,} chars) → {len(chunks)} chunks\\n")
for i, c in enumerate(chunks):
    print(f"  Chunk {i:2d} | [{c.metadata['title']:18s}] {len(c.page_content):3d} chars | {c.page_content[:65]}...")""")

# ── Step 7 ───────────────────────────────────────────────────────────────────
md("""## Step 7 — Build the Vector Store

**ChromaDB** stores each chunk as a vector and enables fast similarity
search. We use `persist_directory` so the index survives kernel restarts.

We clear any previous index to avoid stale data from prior runs.""")

code("""from langchain_community.vectorstores import Chroma
import shutil

CHROMA_DIR = str(data_dir / "website_chroma")

# Start fresh each run
if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name="website_faq",
)

print(f"✅ Vector store created — {vectorstore._collection.count()} vectors stored")""")

# ── Step 8 ───────────────────────────────────────────────────────────────────
md("""## Step 8 — Test Raw Retrieval

Before building the full FAQ chain, let's verify that similarity search
retrieves the correct chunks for sample queries. This is a crucial
debugging step — if retrieval fails, no prompt engineering can fix it.""")

code("""test_queries = [
    "How much does the enterprise plan cost?",
    "Is my data encrypted?",
    "How do I get started?",
]

for query in test_queries:
    print(f"\\nQuery: '{query}'")
    results = vectorstore.similarity_search_with_score(query, k=2)
    for i, (doc, score) in enumerate(results):
        print(f"   [{i+1}] score={score:.4f} | {doc.metadata['title']:15s} | {doc.page_content[:80]}...")""")

# ── Step 9 ───────────────────────────────────────────────────────────────────
md("""## Step 9 — Build the FAQ Bot RAG Chain

The chain combines retrieval and generation:
1. User asks a question
2. Retriever finds top-k relevant chunks from the website index
3. Chunks are injected into the prompt as `{context}`
4. The LLM generates a grounded FAQ answer

**Key design choice:** The system prompt tells the LLM to:
- Answer ONLY from the provided website content
- Cite which page the information comes from
- Gracefully decline when the answer is not on the website""")

code("""from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

system_prompt = (
    "You are a helpful FAQ bot for TechDocs Inc. Answer the customer's question "
    "using ONLY the website content provided below.\\n\\n"
    "Rules:\\n"
    "1. Be specific and cite which page the answer comes from "
    "(e.g., 'According to our Pricing page...').\\n"
    "2. If the answer isn't in the provided content, say: "
    "'I don't have that information on our website. "
    "Please contact support@techdocs.com for help.'\\n"
    "3. Keep answers concise but complete.\\n"
    "4. Do not make up information that is not in the context.\\n\\n"
    "Website Content:\\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def faq_query(question: str) -> dict:
    \"\"\"Run the FAQ bot: retrieve context, generate answer, return both.\"\"\"
    docs = retriever.invoke(question)
    context_text = "\\n\\n---\\n\\n".join(
        f"[Page: {d.metadata['title']}]\\n{d.page_content}" for d in docs
    )
    response = (prompt | llm | StrOutputParser()).invoke(
        {"context": context_text, "input": question}
    )
    return {"answer": response, "sources": docs}


print("✅ FAQ bot chain ready!")""")

# ── Step 10 ──────────────────────────────────────────────────────────────────
md("""## Step 10 — Test the FAQ Bot with Real Questions

Let's ask a range of questions that cover different pages and topics.
For each answer we show the source pages that were retrieved so you
can verify the retrieval quality.""")

code("""questions = [
    "How much does the Professional plan cost?",
    "Do you offer SSO?",
    "Can I get a refund if I cancel after 2 weeks?",
    "How many companies use TechDocs?",
    "What integrations do you support?",
    "How do I import my docs from Notion?",
    "Is TechDocs HIPAA compliant?",
]

for q in questions:
    print(f"\\n{'='*70}")
    print(f"Q: {q}")
    result = faq_query(q)
    print(f"\\nA: {result['answer']}")
    source_pages = [s.metadata['title'] for s in result['sources']]
    print(f"\\n📄 Sources: {source_pages}")""")

# ── Step 11 ──────────────────────────────────────────────────────────────────
md("""## Step 11 — Test Out-of-Scope Questions

A well-designed FAQ bot must **gracefully decline** questions that are
not covered by the website content. These tests verify the bot doesn't
hallucinate answers from the LLM's training data.""")

code("""out_of_scope = [
    "What is the weather in San Francisco?",
    "Can you write me a poem?",
    "What programming languages does the CEO know?",
    "Who is the president of the United States?",
]

print("=== Out-of-Scope Questions (should decline) ===\\n")
for q in out_of_scope:
    result = faq_query(q)
    answer = result["answer"]
    print(f"Q: {q}")
    print(f"A: {answer}")
    grounded = any(phrase in answer.lower() for phrase in [
        "don't have", "not in", "contact support", "no information",
        "not mentioned", "not available", "cannot find", "not covered",
    ])
    print(f"   Grounded refusal: {'✅ Yes' if grounded else '⚠️  No (may have answered from context)'}\\n")""")

# ── Step 12 ──────────────────────────────────────────────────────────────────
md("""## Step 12 — Metadata-Filtered Search

Sometimes you want to search only within a specific section of the website.
ChromaDB supports **metadata filtering** — for example, retrieving only
from billing-related pages when the user asks a pricing question.

This improves precision by reducing the search space.""")

code("""# Search only within "billing" category pages
billing_results = vectorstore.similarity_search(
    "What plans do you offer?",
    k=3,
    filter={"category": "billing"},
)
print("Billing-filtered search results:")
for r in billing_results:
    print(f"  [{r.metadata['title']}] {r.page_content[:80]}...")

print()

# Search only within "product" category pages
product_results = vectorstore.similarity_search(
    "What security certifications do you have?",
    k=3,
    filter={"category": "product"},
)
print("Product-filtered search results:")
for r in product_results:
    print(f"  [{r.metadata['title']}] {r.page_content[:80]}...")""")

# ── Step 13 ──────────────────────────────────────────────────────────────────
md("""## Step 13 — Interactive FAQ Helper

A reusable `ask()` function you can call with any question.
Try your own questions about TechDocs!""")

code("""def ask(question: str) -> str:
    \"\"\"Ask the FAQ bot a question and display the answer with sources.\"\"\"
    result = faq_query(question)
    answer = result["answer"]
    sources = result["sources"]

    print(f"Answer:\\n{answer}\\n")
    print(f"Sources ({len(sources)} chunks):")
    for i, doc in enumerate(sources):
        print(f"   [{i+1}] {doc.metadata['title']}: {doc.page_content[:80]}...")
    return answer

# Try it!
_ = ask("What's the difference between Professional and Enterprise plans?")""")

# ── Limitations ──────────────────────────────────────────────────────────────
md("""## Limitations & Tradeoffs

| Aspect | What happens | How to improve |
|--------|-------------|----------------|
| **Static content** | We simulated pages — real sites change often | Add a periodic re-crawl and re-index pipeline |
| **JavaScript-rendered sites** | BeautifulSoup can't handle SPAs | Use Playwright or Selenium for JS rendering |
| **Chunk boundaries** | FAQ Q&A pairs may get split across chunks | Use Q&A-aware splitting or per-question docs |
| **Boilerplate** | Nav bars / footers may leak into chunks | Improve HTML cleaning with site-specific rules |
| **Multi-turn** | Each question is independent — no memory | Add `ConversationBufferMemory` for follow-ups |
| **Scale** | Small site works fine; thousands of pages need reranking | Add a reranker (see Project 23) |

### What this project does NOT cover
- Live web crawling with link following (see dedicated crawler projects)
- Multi-turn conversation with memory (see Project 18)
- Hybrid keyword + vector search (see Project 21)
- Reranking retrieved results (see Project 23)""")

# ── Summary ──────────────────────────────────────────────────────────────────
md("""## What You Learned

1. **Website content ingestion** — simulating crawl + clean pipeline
2. **HTML cleaning with BeautifulSoup** — stripping nav, scripts, footers
3. **Document metadata** — enriching chunks with URL, title, category
4. **Vector store indexing** — ChromaDB with persist and metadata
5. **Domain-specific prompting** — FAQ-style system prompt with citation rules
6. **Grounded refusal** — declining to answer when info is not on the site
7. **Metadata-filtered search** — narrowing retrieval by page category

## Exercises

1. **Add more pages** — create a `/blog` and `/careers` page and test retrieval
2. **Try real HTML** — fetch a real page with `requests` + BeautifulSoup and index it
3. **Adjust chunk size** — try 200 and 600 and compare retrieval quality
4. **Add conversation memory** — make the bot remember the last 3 questions
5. **Build a category router** — classify the question first, then filter search by category

---

*Next project: **12 — Local Policy Assistant** (HR/IT policy search with citations)*""")


# ── Write notebook ───────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python", "version": "3.10.0",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "nbconvert_exporter": "python", "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

# Fix source format: each line should be a separate string ending with \n
for cell in nb["cells"]:
    raw = cell["source"]
    if isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]
    if isinstance(raw, str):
        lines = raw.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Validate
nb2 = json.load(open(NB_PATH, "r", encoding="utf-8"))
cc = [c for c in nb2["cells"] if c["cell_type"] == "code"]
mc = [c for c in nb2["cells"] if c["cell_type"] == "markdown"]
src = "\n".join("".join(c["source"]) for c in cc)
compile(src, "notebook.ipynb", "exec")
print(f"✅ Notebook 11 written: {len(nb2['cells'])} cells ({len(cc)} code, {len(mc)} markdown) — syntax OK")

