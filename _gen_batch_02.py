"""Generate Projects 11-20: Local RAG — complete notebook implementations."""
import json, os

BASE = r"E:\Github\Machine-Learning-Projects\100_Local_AI_Projects\Local_RAG"

def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.split("\n")}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.split("\n")}

def save(folder, cells):
    d = os.path.join(BASE, folder); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "notebook.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"  ✓ {folder}")

# ───────────────────────────────────────
# Project 11 — Local Website FAQ Bot
# ───────────────────────────────────────
save("11_Local_Website_FAQ_Bot", [
md("""# Project 11 — Local Website FAQ Bot
## Ingest a Website and Answer Questions with RAG

**What you'll learn:**
- Crawl and extract text from web pages
- Build a Q&A system over website content
- Handle HTML → clean text conversion

**Stack:** Ollama · LangChain · BeautifulSoup · Chroma · Jupyter

**Prerequisites:** `ollama pull nomic-embed-text` and `ollama pull qwen3:8b`"""),

code("""# !pip install -q langchain langchain-ollama langchain-community chromadb beautifulsoup4 requests"""),

md("""## Step 1 — Simulate Website Content
We create sample HTML pages to simulate a company FAQ site."""),

code("""from pathlib import Path

PAGES_DIR = Path("sample_website")
PAGES_DIR.mkdir(exist_ok=True)

pages = {
    "pricing.html": \"\"\"<html><body>
<h1>Pricing</h1>
<p>We offer three plans:</p>
<ul>
<li><strong>Starter</strong> - $29/month: 1,000 API calls, email support, 1 user</li>
<li><strong>Pro</strong> - $99/month: 10,000 API calls, priority support, 5 users, webhooks</li>
<li><strong>Enterprise</strong> - Custom pricing: Unlimited calls, dedicated support, SSO, SLA</li>
</ul>
<p>All plans include a 14-day free trial. Annual billing saves 20%.</p>
</body></html>\"\"\",
    "features.html": \"\"\"<html><body>
<h1>Features</h1>
<h2>Document Processing</h2>
<p>Extract text, tables, and images from PDFs, Word docs, and images using OCR.</p>
<h2>API Integration</h2>
<p>RESTful API with SDKs for Python, JavaScript, and Java. Webhook notifications for async jobs.</p>
<h2>Security</h2>
<p>SOC 2 Type II certified. Data encrypted at rest (AES-256) and in transit (TLS 1.3). GDPR compliant.</p>
</body></html>\"\"\",
    "faq.html": \"\"\"<html><body>
<h1>FAQ</h1>
<h3>How do I get an API key?</h3>
<p>Sign up at dashboard.example.com and navigate to Settings > API Keys.</p>
<h3>What file formats are supported?</h3>
<p>PDF, DOCX, XLSX, PNG, JPG, TIFF. Max file size is 50MB.</p>
<h3>Can I self-host?</h3>
<p>Enterprise plan includes an on-premise deployment option with Docker.</p>
<h3>What's the uptime SLA?</h3>
<p>99.9% for Pro, 99.99% for Enterprise with dedicated infrastructure.</p>
</body></html>\"\"\",
}

for name, content in pages.items():
    (PAGES_DIR / name).write_text(content, encoding="utf-8")
print(f"Created {len(pages)} sample web pages")"""),

md("""## Step 2 — Parse HTML to Clean Text"""),

code("""from bs4 import BeautifulSoup
from langchain.schema import Document

def html_to_documents(html_dir: Path) -> list[Document]:
    docs = []
    for html_file in html_dir.glob("*.html"):
        with open(html_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        text = soup.get_text(separator="\\n", strip=True)
        docs.append(Document(page_content=text, metadata={"source": html_file.name}))
    return docs

documents = html_to_documents(PAGES_DIR)
for doc in documents:
    print(f"  {doc.metadata['source']}: {len(doc.page_content)} chars")"""),

md("""## Step 3 — Chunk, Embed, Index"""),

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="qwen3:8b", temperature=0.3)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="website_faq")
print("Vector store ready!")"""),

md("""## Step 4 — Build FAQ Bot"""),

code("""from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"You are a helpful FAQ bot for our product. Answer based on the website content below.
If the answer isn't in the context, say "I couldn't find that info on our website."

Context: {context}

Question: {question}

Answer:\"\"\"
)

qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True, chain_type_kwargs={"prompt": prompt},
)"""),

md("""## Step 5 — Test the Bot"""),

code("""questions = [
    "How much does the Pro plan cost?",
    "What file formats do you support?",
    "Is the platform SOC 2 certified?",
    "How do I get an API key?",
    "Do you offer annual billing discounts?",
    "Can I deploy on-premise?",
]

for q in questions:
    r = qa.invoke({"query": q})
    sources = [d.metadata["source"] for d in r["source_documents"]]
    print(f"Q: {q}")
    print(f"A: {r['result']}")
    print(f"Sources: {sources}\\n")"""),

md("""## What You Learned
- **HTML parsing** — BeautifulSoup for clean text extraction
- **Website-to-RAG pipeline** — crawl → parse → chunk → embed → query
- **Source attribution** — showing which page answered the question

## Next Steps
- Use `WebBaseLoader` for live URL crawling
- Add sitemap-based recursive crawling
- Combine with Project 19 (product docs copilot)"""),
])

# ───────────────────────────────────────
# Project 12 — Local Policy Assistant
# ───────────────────────────────────────
save("12_Local_Policy_Assistant", [
md("""# Project 12 — Local Policy Assistant
## Search Company Policies with Citations

**What you'll learn:**
- Index multiple policy documents with metadata
- Return answers with exact source citations
- Handle overlapping/contradictory policies

**Stack:** Ollama · LangChain · Chroma · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama langchain-community chromadb"""),

md("""## Step 1 — Sample Policy Documents"""),

code("""from pathlib import Path
from langchain.schema import Document

POLICIES = [
    Document(page_content=\"\"\"Remote Work Policy (HR-001, Updated Jan 2024)
Employees may work remotely up to 3 days per week with manager approval.
Full remote requires VP approval and quarterly in-office visits.
Equipment: company provides laptop and monitor. $500 annual home office stipend.
Core hours: 10am-3pm local time for meetings and collaboration.\"\"\",
        metadata={"source": "HR-001", "department": "HR", "category": "remote_work"}),

    Document(page_content=\"\"\"Travel & Expense Policy (FIN-003, Updated Mar 2024)
Pre-approval required for travel over $500. Flights: economy class for <6hr, business for 6hr+.
Hotels: up to $200/night domestic, $300/night international.
Meal per diem: $75/day domestic, $100/day international.
Receipts required for all expenses over $25. Submit within 30 days.\"\"\",
        metadata={"source": "FIN-003", "department": "Finance", "category": "expenses"}),

    Document(page_content=\"\"\"IT Security Policy (IT-007, Updated Feb 2024)
All devices must have full-disk encryption and company MDM enrolled.
Passwords: minimum 12 characters, changed every 90 days, no reuse of last 10.
MFA required for all company applications. Personal devices: no company data.
Report security incidents to security@company.com within 1 hour.\"\"\",
        metadata={"source": "IT-007", "department": "IT", "category": "security"}),

    Document(page_content=\"\"\"PTO Policy (HR-002, Updated Jan 2024)
Full-time employees: 20 days PTO per year (accrued monthly).
Unused PTO carries over up to 5 days. PTO requests: 2 weeks advance notice.
Sick leave: 10 days per year (no carryover). Bereavement: 5 days for immediate family.
Parental leave: 16 weeks paid for primary caregiver, 8 weeks for secondary.\"\"\",
        metadata={"source": "HR-002", "department": "HR", "category": "leave"}),
]

print(f"Loaded {len(POLICIES)} policy documents")
for p in POLICIES:
    print(f"  [{p.metadata['source']}] {p.metadata['category']} - {len(p.page_content)} chars")"""),

md("""## Step 2 — Index with Metadata"""),

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="qwen3:8b", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(POLICIES)
print(f"Chunks: {len(chunks)}")

vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="policies")"""),

md("""## Step 3 — Policy QA with Citations"""),

code("""from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"You are a company policy assistant. Answer questions using ONLY the policy documents below.
Always cite the policy number (e.g., HR-001) and quote the relevant rule.
If no policy covers the question, say "No applicable policy found."

Policies:
{context}

Question: {question}

Answer (with citations):\"\"\"
)

qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True, chain_type_kwargs={"prompt": prompt},
)"""),

md("""## Step 4 — Test Policy Queries"""),

code("""questions = [
    "How many days can I work from home?",
    "What's the meal allowance for international travel?",
    "How often do I need to change my password?",
    "How much parental leave do I get?",
    "Can I carry over unused PTO?",
    "Do I need approval for a $300 flight?",
]

for q in questions:
    r = qa.invoke({"query": q})
    citations = set(d.metadata["source"] for d in r["source_documents"])
    print(f"Q: {q}")
    print(f"A: {r['result']}")
    print(f"Citations: {citations}\\n")"""),

md("""## Step 5 — Filter by Department"""),

code("""# Metadata filtering — only search HR policies
hr_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3, "filter": {"department": "HR"}}
)

result = hr_retriever.invoke("What is the PTO policy?")
for doc in result:
    print(f"[{doc.metadata['source']}] {doc.page_content[:100]}...")"""),

md("""## What You Learned
- **Metadata-tagged documents** — department, category, policy ID
- **Citation-aware prompting** — requiring source references
- **Metadata filtering** — narrowing search by department
- **Policy-specific RAG** — handling corporate knowledge

## Next Steps: Combine multiple policy sources (→ Project 13), add approval workflow (→ 40)"""),
])

# ───────────────────────────────────────
# Project 13 — Local Multi-PDF Research Librarian
# ───────────────────────────────────────
save("13_Local_Multi_PDF_Research_Librarian", [
md("""# Project 13 — Local Multi-PDF Research Librarian
## Answer Across Multiple Papers with Evidence

**What you'll learn:**
- Index multiple documents with source tracking
- Cross-document Q&A with evidence from each source
- Compare findings across documents

**Stack:** Ollama · LlamaIndex · Jupyter"""),

code("""# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"""),

md("""## Step 1 — Sample Research Papers"""),

code("""from pathlib import Path

PAPERS_DIR = Path("sample_papers")
PAPERS_DIR.mkdir(exist_ok=True)

papers = {
    "attention_paper.txt": \"\"\"Title: Attention Is All You Need (2017)
The Transformer relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions.
Multi-head attention allows jointly attending to information from different subspaces.
Achieves 28.4 BLEU on WMT En-De translation. Training: 3.5 days on 8 GPUs.
Key innovation: positional encoding replaces sequential processing.\"\"\",

    "bert_paper.txt": \"\"\"Title: BERT: Pre-training of Deep Bidirectional Transformers (2018)
BERT uses masked language modeling (MLM) and next sentence prediction (NSP) for pre-training.
Bidirectional context gives advantages over left-to-right models like GPT.
Fine-tuning on downstream tasks achieves SOTA on 11 NLP benchmarks.
Base model: 110M parameters. Large model: 340M parameters.\"\"\",

    "gpt2_paper.txt": \"\"\"Title: Language Models are Unsupervised Multitask Learners (2019)
GPT-2 demonstrates that language models can perform tasks without explicit supervision.
Trained on WebText (40GB). 1.5B parameters (largest version).
Zero-shot performance competitive with supervised baselines on many tasks.
Key finding: model capacity scales with dataset size and compute.\"\"\",

    "rag_paper.txt": \"\"\"Title: Retrieval-Augmented Generation for Knowledge-Intensive Tasks (2020)
RAG combines pre-trained parametric memory (generator) with non-parametric memory (retriever).
Uses DPR for retrieval and BART for generation.
Outperforms pure generative models on open-domain QA, fact verification, and Jeopardy.
Key advantage: knowledge can be updated without retraining the generator.\"\"\",
}

for name, content in papers.items():
    (PAPERS_DIR / name).write_text(content, encoding="utf-8")
print(f"Created {len(papers)} sample papers")"""),

md("""## Step 2 — Index All Papers"""),

code("""from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

documents = SimpleDirectoryReader(input_dir=str(PAPERS_DIR)).load_data()
print(f"Loaded {len(documents)} documents")

index = VectorStoreIndex.from_documents(documents, show_progress=True)
engine = index.as_query_engine(similarity_top_k=4)
print("Index ready!")"""),

md("""## Step 3 — Cross-Document Questions"""),

code("""questions = [
    "Compare the architectures of BERT and GPT-2.",
    "How does RAG improve over pure generative models?",
    "What was the key innovation that transformers introduced?",
    "Which paper had the largest model? How many parameters?",
    "What training approaches are used across these papers?",
]

for q in questions:
    print(f"\\nQ: {q}")
    response = engine.query(q)
    sources = [n.metadata.get("file_name", "?") for n in response.source_nodes]
    print(f"A: {response}")
    print(f"Evidence from: {sources}")
    print("-" * 50)"""),

md("""## Step 4 — Paper Comparison Table"""),

code("""comparison_query = \"\"\"Create a comparison table of all papers with columns:
Paper Title, Year, Key Innovation, Model Size, Main Result.
Use only information from the provided documents.\"\"\"

response = engine.query(comparison_query)
print(response)"""),

md("""## What You Learned
- **Multi-document indexing** — combining papers with provenance
- **Cross-document Q&A** — finding info across multiple sources
- **Evidence attribution** — knowing which paper answered what
- **Comparison queries** — synthesizing across sources

## Next Steps: Add hybrid search (→ 21), reranking (→ 23), citation verification (→ 29)"""),
])

# ───────────────────────────────────────
# Project 14 — Local Financial Report Analyst
# ───────────────────────────────────────
save("14_Local_Financial_Report_Analyst", [
md("""# Project 14 — Local Financial Report Analyst
## Q&A Over Annual Reports with Text + Table Parsing

**What you'll learn:**
- Parse both narrative text and financial tables
- Build RAG over mixed text/tabular data
- Answer quantitative questions from documents

**Stack:** Ollama · LangChain · Pandas · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama langchain-community chromadb pandas"""),

md("""## Step 1 — Sample Financial Report"""),

code("""import pandas as pd
from langchain.schema import Document

# Simulate report sections
REPORT_TEXT = [
    Document(page_content=\"\"\"Annual Report 2024 — TechCorp Inc.
Revenue grew 23% year-over-year to $4.2B, driven by cloud services (+35%) and 
enterprise subscriptions (+18%). Operating margin improved to 28% from 24%.
R&D spending increased to $890M (21% of revenue), focused on AI and automation.
Free cash flow reached $1.1B, up 30% from prior year.\"\"\",
        metadata={"section": "financial_highlights", "year": "2024"}),

    Document(page_content=\"\"\"Risk Factors
Key risks include: 1) Increased competition in cloud services from major providers,
2) Regulatory changes in data privacy (GDPR, CCPA expansion), 3) Talent retention
in AI/ML engineering, 4) Foreign exchange exposure (40% international revenue),
5) Cybersecurity threats and potential data breaches.\"\"\",
        metadata={"section": "risks", "year": "2024"}),

    Document(page_content=\"\"\"Business Outlook
Management projects FY2025 revenue of $4.9-5.1B (17-21% growth).
Cloud services expected to reach 60% of total revenue (from 52%).
Planned acquisitions of two AI startups in Q1-Q2 2025.
Headcount expected to grow 15% with focus on engineering roles.\"\"\",
        metadata={"section": "outlook", "year": "2024"}),
]

# Financial tables as structured data
financial_data = pd.DataFrame({
    "Metric": ["Revenue", "Cloud Revenue", "Operating Income", "Net Income", "R&D Spend", "FCF"],
    "2022": ["$2.8B", "$1.1B", "$560M", "$420M", "$590M", "$650M"],
    "2023": ["$3.4B", "$1.6B", "$816M", "$612M", "$720M", "$850M"],
    "2024": ["$4.2B", "$2.2B", "$1.18B", "$890M", "$890M", "$1.1B"],
    "YoY_Growth": ["23%", "35%", "45%", "45%", "24%", "30%"],
})

print("Financial Summary:")
print(financial_data.to_string(index=False))

# Add table as document
table_doc = Document(
    page_content=f"Financial Data Table:\\n{financial_data.to_string(index=False)}",
    metadata={"section": "financials_table", "year": "2024"}
)
REPORT_TEXT.append(table_doc)"""),

md("""## Step 2 — Build Financial Report Index"""),

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="qwen3:8b", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(REPORT_TEXT)

vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="financial_report")
print(f"Indexed {len(chunks)} chunks from financial report")"""),

md("""## Step 3 — Financial QA Chain"""),

code("""from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"You are a financial analyst. Answer based ONLY on the report data below.
For numerical questions, cite specific figures. If calculating, show your work.
If the data doesn't contain the answer, say so.

Report Data:
{context}

Question: {question}

Analysis:\"\"\"
)

qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True, chain_type_kwargs={"prompt": prompt},
)"""),

md("""## Step 4 — Ask Financial Questions"""),

code("""questions = [
    "What was the revenue growth rate in 2024?",
    "How much did R&D spending increase as a percentage of revenue?",
    "What are the top 3 risk factors?",
    "What is the projected revenue for FY2025?",
    "How has cloud revenue trended over the 3 years?",
    "What was the operating margin improvement?",
]

for q in questions:
    r = qa.invoke({"query": q})
    sections = set(d.metadata["section"] for d in r["source_documents"])
    print(f"Q: {q}")
    print(f"A: {r['result']}")
    print(f"Sections: {sections}\\n")"""),

md("""## What You Learned
- **Mixed text/table parsing** — combining narrative and structured data
- **Financial document RAG** — answering quantitative questions
- **Section metadata** — tracking which part of the report answered
- **Precision prompting** — requiring citations and calculations

## Next Steps: Add multi-year trend analysis, combine with table RAG (→ 26)"""),
])

# ───────────────────────────────────────
# Project 15 — Local Contract Clause Finder
# ───────────────────────────────────────
save("15_Local_Contract_Clause_Finder", [
md("""# Project 15 — Local Contract Clause Finder
## Retrieve Risky or Important Contract Clauses

**What you'll learn:**
- Index legal documents with clause-level granularity
- Risk classification of contract terms
- Structured extraction of key contract elements

**Stack:** Ollama · LangChain · Chroma · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama langchain-community chromadb"""),

md("""## Step 1 — Sample Contract Clauses"""),

code("""from langchain.schema import Document

CLAUSES = [
    Document(page_content=\"\"\"Limitation of Liability (Section 8.1)
In no event shall either party's aggregate liability exceed the total fees paid
in the 12 months prior to the claim. Neither party shall be liable for indirect,
incidental, special, consequential, or punitive damages, regardless of the theory
of liability.\"\"\", metadata={"clause_type": "liability", "risk": "medium", "section": "8.1"}),

    Document(page_content=\"\"\"Indemnification (Section 9.1)
Customer shall indemnify and hold harmless Provider against any third-party claims
arising from Customer's use of the Service in violation of this Agreement or
applicable law. Provider shall indemnify Customer against IP infringement claims.\"\"\",
        metadata={"clause_type": "indemnification", "risk": "high", "section": "9.1"}),

    Document(page_content=\"\"\"Termination for Convenience (Section 12.2)
Either party may terminate this Agreement without cause upon 90 days written notice.
Upon termination, Customer must pay all fees through the end of the current term.
Provider shall provide data export within 30 days of termination.\"\"\",
        metadata={"clause_type": "termination", "risk": "medium", "section": "12.2"}),

    Document(page_content=\"\"\"Auto-Renewal (Section 12.4)
This Agreement automatically renews for successive 1-year terms unless either party 
provides written notice of non-renewal at least 60 days before the end of the current term.
Price increases of up to 5% may apply upon renewal.\"\"\",
        metadata={"clause_type": "renewal", "risk": "high", "section": "12.4"}),

    Document(page_content=\"\"\"Data Processing (Section 14.1)
Provider processes Customer Data solely for providing the Service. Data is stored in 
US data centers with SOC 2 Type II certification. Provider shall notify Customer of
any data breach within 72 hours. Customer owns all Customer Data.\"\"\",
        metadata={"clause_type": "data_processing", "risk": "low", "section": "14.1"}),

    Document(page_content=\"\"\"Non-Compete (Section 16.1)
During the term and for 24 months after termination, Customer shall not directly or
indirectly develop or market a product that competes with the Service. This restriction
applies globally.\"\"\", metadata={"clause_type": "non_compete", "risk": "critical", "section": "16.1"}),
]

print(f"Loaded {len(CLAUSES)} contract clauses")
for c in CLAUSES:
    print(f"  [{c.metadata['risk'].upper()}] Section {c.metadata['section']}: {c.metadata['clause_type']}")"""),

md("""## Step 2 — Index Clauses"""),

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="qwen3:8b", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(CLAUSES, embeddings, collection_name="contracts")
print("Contract clauses indexed.")"""),

md("""## Step 3 — Search Contract Clauses"""),

code("""from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"You are a contract review assistant. Analyze the contract clauses below.
Identify risks, obligations, and important terms. Always cite the section number.

Clauses:
{context}

Question: {question}

Analysis (cite section numbers):\"\"\"
)

qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True, chain_type_kwargs={"prompt": prompt},
)

queries = [
    "What are the highest-risk clauses in this contract?",
    "What happens if I want to terminate early?",
    "Are there any non-compete restrictions?",
    "What are the data protection commitments?",
    "Does the contract auto-renew? What are the terms?",
]

for q in queries:
    r = qa.invoke({"query": q})
    print(f"Q: {q}")
    print(f"A: {r['result']}\\n")"""),

md("""## Step 4 — Risk Assessment Summary"""),

code("""import json, re

risk_prompt = PromptTemplate(
    input_variables=["clauses"],
    template=\"\"\"Review these contract clauses and create a risk assessment.
Return JSON with: {{"high_risk": [list of risky clauses with section numbers and why],
"recommendations": [list of negotiation suggestions], "missing_protections": [what's not covered]}}

Clauses:
{clauses}

JSON risk assessment:\"\"\"
)

all_text = "\\n\\n".join([c.page_content for c in CLAUSES])
result = (risk_prompt | llm).invoke({"clauses": all_text})
print(result.content)"""),

md("""## What You Learned
- **Clause-level indexing** — granular document segmentation
- **Risk metadata** — tagging content by risk level
- **Legal Q&A** — contract-specific prompting
- **Risk assessment** — automated clause analysis

## Next Steps: Add clause comparison across contracts, compliance checking (→ 40)"""),
])

# ───────────────────────────────────────
# Project 16 — Local Course Tutor
# ───────────────────────────────────────
save("16_Local_Course_Tutor", [
md("""# Project 16 — Local Course Tutor
## Q&A Over Lecture Notes and Course Materials

**What you'll learn:**
- Index multi-topic educational content
- Build a tutoring chatbot with concept explanations
- Generate practice questions from course material

**Stack:** Ollama · LangChain · Chroma · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama langchain-community chromadb"""),

md("""## Step 1 — Sample Lecture Notes"""),

code("""from langchain.schema import Document

LECTURES = [
    Document(page_content=\"\"\"Lecture 1: Introduction to Databases
A database is an organized collection of structured data. DBMS (Database Management Systems)
handle storage, retrieval, and management. Key models: relational (tables with rows/columns),
document (JSON-like), graph (nodes and edges), key-value (simple lookups).
SQL is the standard language for relational databases. ACID properties ensure transaction
reliability: Atomicity, Consistency, Isolation, Durability.\"\"\",
        metadata={"lecture": 1, "topic": "databases_intro", "week": 1}),

    Document(page_content=\"\"\"Lecture 2: SQL Fundamentals
SELECT retrieves data: SELECT col FROM table WHERE condition.
JOINs combine tables: INNER (matching rows), LEFT (all from left + matches),
RIGHT (all from right), FULL OUTER (all rows).
Aggregations: COUNT, SUM, AVG, MIN, MAX with GROUP BY.
Subqueries: SELECT within SELECT for complex filtering.
Indexes speed up queries but slow down writes — use on frequently queried columns.\"\"\",
        metadata={"lecture": 2, "topic": "sql_fundamentals", "week": 2}),

    Document(page_content=\"\"\"Lecture 3: Database Design & Normalization
Normal forms reduce redundancy: 1NF (atomic values), 2NF (no partial dependencies),
3NF (no transitive dependencies), BCNF (every determinant is a candidate key).
ER diagrams model entities, attributes, and relationships.
Denormalization: intentionally adding redundancy for read performance.
Schema design trade-off: normalized = less redundancy, denormalized = faster reads.\"\"\",
        metadata={"lecture": 3, "topic": "db_design", "week": 3}),
]

print(f"Loaded {len(LECTURES)} lectures")"""),

md("""## Step 2 — Build the Tutor"""),

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = ChatOllama(model="qwen3:8b", temperature=0.3)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(LECTURES, embeddings, collection_name="course")

tutor_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"You are a patient course tutor. Answer using the lecture material.
If explaining a concept, use simple language and an example.
If the student asks something not in the lectures, say so and give a brief general answer.

Lecture Notes:
{context}

Student's Question: {question}

Tutor's Answer:\"\"\"
)

tutor = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True, chain_type_kwargs={"prompt": tutor_prompt},
)"""),

md("""## Step 3 — Ask Course Questions"""),

code("""questions = [
    "What are the ACID properties?",
    "How does a LEFT JOIN work?",
    "What's the difference between 2NF and 3NF?",
    "When should I use an index?",
    "What is denormalization and when is it useful?",
]

for q in questions:
    r = tutor.invoke({"query": q})
    lectures = [d.metadata["lecture"] for d in r["source_documents"]]
    print(f"Q: {q}")
    print(f"A: {r['result']}")
    print(f"From lectures: {lectures}\\n")"""),

md("""## Step 4 — Generate Practice Questions"""),

code("""from langchain.prompts import ChatPromptTemplate

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 practice questions with answers from this lecture material. Mix conceptual and applied questions."),
    ("human", "{content}"),
])

for lec in LECTURES:
    print(f"\\nLecture {lec.metadata['lecture']}: {lec.metadata['topic']}")
    result = (quiz_prompt | llm).invoke({"content": lec.page_content})
    print(result.content)
    print("-" * 40)"""),

md("""## What You Learned
- **Educational RAG** — building a tutoring system over course materials
- **Concept-focused prompting** — explaining with examples
- **Practice generation** — creating questions from lecture content

## Next Steps: Add spaced repetition (→ 9), connect to paper librarian (→ 13)"""),
])

# ───────────────────────────────────────
# Project 17 — Local Personal Wiki Copilot
# ───────────────────────────────────────
save("17_Local_Personal_Wiki_Copilot", [
md("""# Project 17 — Local Personal Wiki Copilot
## Query Exported Notes and Wiki Files with LlamaIndex

**What you'll learn:**
- Index diverse note formats (markdown, text, org)
- Handle inter-linked notes and references
- Build a personal knowledge assistant

**Stack:** Ollama · LlamaIndex · Jupyter"""),

code("""# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"""),

md("""## Step 1 — Simulate a Personal Wiki"""),

code("""from pathlib import Path

WIKI_DIR = Path("sample_wiki")
WIKI_DIR.mkdir(exist_ok=True)

notes = {
    "projects.md": "# Active Projects\\n## RAG Pipeline\\nBuilding a local RAG system. Using Chroma + Ollama.\\nStatus: In progress. Need to add reranking.\\n\\n## Blog Rewrite\\nRewriting technical blog series. 3/10 posts done.\\nDeadline: End of month.\\n",
    "meeting_log.md": "# Meeting Log\\n## 2024-12-01 — Team Sync\\nDiscussed Q1 priorities. Agreed on RAG project timeline.\\nAction: Mike to set up vector DB by Friday.\\n\\n## 2024-12-08 — Product Review\\nDemo'd RAG prototype. Feedback: improve citation display.\\n",
    "learning.md": "# Learning Notes\\n## LangChain\\nChains compose LLM calls. Key types: stuff, map_reduce, refine.\\nAgents can use tools. ReAct pattern: Reason + Act.\\n\\n## LlamaIndex\\nSimpleDirectoryReader for loading. VectorStoreIndex for search.\\nSummaryIndex for full-doc summarization.\\n",
    "ideas.md": "# Idea Backlog\\n- Build a local coding assistant with repo indexing\\n- Create a meeting summarizer with Whisper + Ollama\\n- Design a personal finance tracker with LLM categorization\\n- Experiment with multi-agent debate for research\\n",
}

for name, content in notes.items():
    (WIKI_DIR / name).write_text(content, encoding="utf-8")
print(f"Created {len(notes)} wiki notes")"""),

md("""## Step 2 — Index the Wiki"""),

code("""from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

documents = SimpleDirectoryReader(input_dir=str(WIKI_DIR)).load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
engine = index.as_query_engine(similarity_top_k=3)
print(f"Indexed {len(documents)} wiki pages")"""),

md("""## Step 3 — Query Your Wiki"""),

code("""queries = [
    "What projects am I currently working on?",
    "What was decided in the last team meeting?",
    "What do I know about LangChain agents?",
    "What ideas do I have in my backlog?",
    "What's the status of the RAG pipeline project?",
]

for q in queries:
    response = engine.query(q)
    sources = [n.metadata.get("file_name", "?") for n in response.source_nodes]
    print(f"Q: {q}")
    print(f"A: {response}")
    print(f"From: {sources}\\n")"""),

md("""## Step 4 — Weekly Summary Generator"""),

code("""summary_query = \"\"\"Based on all my notes, generate a weekly summary with:
1. Active projects and their status
2. Recent meeting decisions
3. Upcoming deadlines
4. Top ideas to explore\"\"\"

response = engine.query(summary_query)
print("Weekly Summary")
print("=" * 50)
print(response)"""),

md("""## What You Learned
- **Personal knowledge base** — indexing diverse note types
- **Cross-note queries** — finding info across your wiki
- **Summary generation** — synthesizing from multiple notes

## Next Steps: Point to real Obsidian vault, add daily note ingestion"""),
])

# ───────────────────────────────────────
# Project 18 — Local Customer Support Memory Bot
# ───────────────────────────────────────
save("18_Local_Customer_Support_Memory_Bot", [
md("""# Project 18 — Local Customer Support Memory Bot
## Retrieve Similar Past Tickets and Suggest Solutions

**What you'll learn:**
- Build a ticket knowledge base with embeddings
- Find similar past issues for resolution guidance
- Add conversation memory across support sessions

**Stack:** Ollama · LangChain · Chroma · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama langchain-community chromadb"""),

md("""## Step 1 — Sample Support Tickets"""),

code("""from langchain.schema import Document

TICKETS = [
    Document(page_content="Ticket #1001: User can't login after password reset. Error: 'Invalid credentials'. Resolution: Clear browser cache and cookies, then try again. If persists, check if CAPS LOCK is on. Last resort: admin resets password manually.",
        metadata={"ticket_id": "1001", "category": "auth", "resolution_time": "15min", "status": "resolved"}),
    Document(page_content="Ticket #1002: API returning 429 Too Many Requests. Customer on Pro plan hitting rate limits during peak hours. Resolution: Temporarily increased rate limit to 1000/min. Suggested upgrading to Enterprise for dedicated limits.",
        metadata={"ticket_id": "1002", "category": "api", "resolution_time": "30min", "status": "resolved"}),
    Document(page_content="Ticket #1003: Dashboard charts not loading, showing blank white space. Browser: Chrome 120. Resolution: Known issue with Chrome 120 + our chart library. Workaround: use Firefox. Fix deployed in v2.4.1 patch.",
        metadata={"ticket_id": "1003", "category": "ui", "resolution_time": "2hr", "status": "resolved"}),
    Document(page_content="Ticket #1004: Export to CSV producing corrupted files with encoding issues. Non-ASCII characters (Japanese) appearing as garbled text. Resolution: Updated CSV export to use UTF-8 BOM encoding. Fix in v2.4.0.",
        metadata={"ticket_id": "1004", "category": "export", "resolution_time": "4hr", "status": "resolved"}),
    Document(page_content="Ticket #1005: Webhook notifications not firing for new order events. Customer using custom endpoint. Resolution: Webhook URL had trailing slash causing 301 redirect. Removed trailing slash, confirmed delivery.",
        metadata={"ticket_id": "1005", "category": "integrations", "resolution_time": "45min", "status": "resolved"}),
    Document(page_content="Ticket #1006: SSO login failing with SAML assertion error. Enterprise customer using Okta. Resolution: Clock skew between Okta and our servers exceeded 5min tolerance. Customer synced NTP. Also extended tolerance to 10min.",
        metadata={"ticket_id": "1006", "category": "auth", "resolution_time": "2hr", "status": "resolved"}),
]

print(f"Loaded {len(TICKETS)} past support tickets")"""),

md("""## Step 2 — Build Ticket Knowledge Base"""),

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOllama(model="qwen3:8b", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(TICKETS, embeddings, collection_name="support_tickets")
print("Ticket knowledge base ready!")"""),

md("""## Step 3 — Find Similar Tickets"""),

code("""def find_similar_tickets(issue: str, k: int = 3):
    results = vectorstore.similarity_search_with_score(issue, k=k)
    print(f"\\nNew Issue: {issue}")
    print(f"Similar past tickets:")
    for doc, score in results:
        print(f"  [{doc.metadata['ticket_id']}] (similarity: {score:.3f})")
        print(f"   Category: {doc.metadata['category']} | Resolved in: {doc.metadata['resolution_time']}")
        print(f"   {doc.page_content[:120]}...")
        print()

find_similar_tickets("Customer reports they can't sign in, getting credential errors")
find_similar_tickets("API calls are being throttled, customer needs higher limits")
find_similar_tickets("CSV download has weird characters for international text")"""),

md("""## Step 4 — Support Bot with Conversation Memory"""),

code("""from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

support_bot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
)

# Simulate multi-turn support conversation
conversation = [
    "A customer says their login isn't working after they reset their password",
    "What if clearing the cache doesn't help?",
    "Are there any other auth-related tickets I should know about?",
]

for msg in conversation:
    result = support_bot.invoke({"question": msg})
    tickets = [d.metadata["ticket_id"] for d in result["source_documents"]]
    print(f"Agent: {msg}")
    print(f"Bot: {result['answer']}")
    print(f"Related tickets: {tickets}\\n")"""),

md("""## What You Learned
- **Ticket knowledge base** — embedding past resolutions
- **Similarity search with scores** — finding relevant tickets
- **Conversation memory** — maintaining context across messages
- **Support workflow** — combining retrieval with guided responses

## Next Steps: Add ticket escalation routing (→ 39), CRM enrichment (→ 59)"""),
])

# ───────────────────────────────────────
# Project 19 — Local Product Docs Copilot
# ───────────────────────────────────────
save("19_Local_Product_Docs_Copilot", [
md("""# Project 19 — Local Product Docs Copilot
## Chat Over Internal or API Documentation

**What you'll learn:**
- Index API documentation with endpoint metadata
- Generate code examples from docs
- Handle multi-section technical documentation

**Stack:** Ollama · LangChain · Chroma · Jupyter"""),

code("""# !pip install -q langchain langchain-ollama langchain-community chromadb"""),

md("""## Step 1 — Sample API Documentation"""),

code("""from langchain.schema import Document

DOCS = [
    Document(page_content=\"\"\"Authentication API
POST /api/v1/auth/login
Body: {"email": "string", "password": "string"}
Response: {"token": "jwt_string", "expires_in": 3600}
Rate limit: 5 attempts per minute. Lockout after 10 failed attempts.
Token valid for 1 hour. Use refresh endpoint for new token.\"\"\",
        metadata={"section": "auth", "method": "POST", "endpoint": "/api/v1/auth/login"}),

    Document(page_content=\"\"\"Users API
GET /api/v1/users — List all users (paginated, 20/page)
GET /api/v1/users/{id} — Get user by ID
POST /api/v1/users — Create user (requires admin role)
PATCH /api/v1/users/{id} — Update user fields
DELETE /api/v1/users/{id} — Soft-delete user (admin only)
Query params: page, limit, sort, filter[role], filter[status]\"\"\",
        metadata={"section": "users", "method": "CRUD", "endpoint": "/api/v1/users"}),

    Document(page_content=\"\"\"Webhooks API
POST /api/v1/webhooks — Register a webhook endpoint
Supported events: order.created, order.updated, payment.completed, user.signup
Payload format: {"event": "string", "data": {}, "timestamp": "ISO8601"}
Retry policy: 3 attempts with exponential backoff (1s, 5s, 25s)
Signature: HMAC-SHA256 in X-Webhook-Signature header for verification\"\"\",
        metadata={"section": "webhooks", "method": "POST", "endpoint": "/api/v1/webhooks"}),

    Document(page_content=\"\"\"Error Handling
Standard error format: {"error": {"code": "string", "message": "string", "details": []}}
Common codes: 400 (bad request), 401 (unauthorized), 403 (forbidden), 404 (not found),
429 (rate limited), 500 (server error).
Rate limiting: X-RateLimit-Remaining and X-RateLimit-Reset headers.
Best practice: implement exponential backoff for 429 and 5xx responses.\"\"\",
        metadata={"section": "errors", "method": "N/A", "endpoint": "N/A"}),
]

print(f"Loaded {len(DOCS)} API documentation sections")"""),

md("""## Step 2 — Index and Build Copilot"""),

code("""from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = ChatOllama(model="qwen3:8b", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(DOCS, embeddings, collection_name="api_docs")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=\"\"\"You are an API documentation copilot. Answer using the docs below.
When possible, include code examples (Python requests or curl).
Cite endpoints and response formats.

Documentation:
{context}

Developer Question: {question}

Answer (with code example if applicable):\"\"\"
)

copilot = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True, chain_type_kwargs={"prompt": prompt},
)"""),

md("""## Step 3 — Ask API Questions"""),

code("""questions = [
    "How do I authenticate with the API?",
    "Show me how to list users with pagination",
    "How do I set up webhooks for new orders?",
    "What should I do when I get a 429 error?",
    "How do I verify webhook signatures?",
]

for q in questions:
    r = copilot.invoke({"query": q})
    sections = [d.metadata["section"] for d in r["source_documents"]]
    print(f"Q: {q}")
    print(f"A: {r['result']}")
    print(f"Sections: {sections}\\n{'='*50}\\n")"""),

md("""## What You Learned
- **API documentation indexing** — endpoint metadata tracking
- **Code example generation** — practical developer assistance
- **Technical Q&A** — precise, citation-backed answers

## Next Steps: Add API spec parsing (→ 97), combine with code copilot (→ 91)"""),
])

# ───────────────────────────────────────
# Project 20 — Local Medical Literature Finder
# ───────────────────────────────────────
save("20_Local_Medical_Literature_Finder", [
md("""# Project 20 — Local Medical Literature Finder
## Search Papers by Topic and Evidence Level with Metadata Filters

**What you'll learn:**
- Metadata-rich document indexing (year, journal, evidence level)
- Filtered retrieval by metadata attributes
- Evidence-based answer generation

**Stack:** Ollama · LlamaIndex · Jupyter

> **Disclaimer:** This is a learning project and NOT a medical advice tool."""),

code("""# !pip install -q llama-index llama-index-llms-ollama llama-index-embeddings-ollama"""),

md("""## Step 1 — Sample Medical Literature"""),

code("""from llama_index.core import Document

PAPERS = [
    Document(text=\"\"\"Efficacy of Metformin in Type 2 Diabetes Management
Metformin remains first-line treatment for T2DM. Meta-analysis of 35 RCTs (n=12,000) 
shows HbA1c reduction of 1.0-1.5%. Cardiovascular benefit demonstrated in UKPDS.
Main side effects: GI disturbance (20-30%), rare lactic acidosis. Contraindicated 
in eGFR <30. Cost-effective at ~$4/month generic.\"\"\",
        metadata={"year": 2023, "journal": "Lancet Diabetes", "evidence_level": "high",
                  "topic": "diabetes", "study_type": "meta-analysis"}),

    Document(text=\"\"\"Machine Learning in Medical Imaging: A Systematic Review
Deep learning models achieve radiologist-level performance in chest X-ray interpretation 
(AUC 0.94-0.97). CNN-based tools FDA-approved for diabetic retinopathy screening.
Challenges: dataset bias, lack of explainability, integration with clinical workflows.
Current adoption limited to research settings in most institutions.\"\"\",
        metadata={"year": 2024, "journal": "Nature Medicine", "evidence_level": "high",
                  "topic": "ai_medicine", "study_type": "systematic_review"}),

    Document(text=\"\"\"Telemedicine Effectiveness During COVID-19: Outcomes Analysis
Analysis of 50,000 patient encounters shows telemedicine achieves comparable outcomes 
to in-person visits for chronic disease management (diabetes, hypertension, mental health).
Patient satisfaction: 87%. No-show rates reduced from 18% to 7%. Cost savings: 30% per visit.
Limitations: physical examination constraints, technology barriers for elderly patients.\"\"\",
        metadata={"year": 2023, "journal": "JAMA", "evidence_level": "moderate",
                  "topic": "telemedicine", "study_type": "observational"}),

    Document(text=\"\"\"Continuous Glucose Monitoring Impact on Diabetic Outcomes
CGM use associated with 0.5% additional HbA1c reduction vs traditional monitoring.
Time in range improved from 55% to 70%. Hypoglycemia episodes reduced by 40%.
Real-world data from 5,000 patients over 12 months. Device adherence: 78%.
Cost-effectiveness favorable for patients with HbA1c >8%.\"\"\",
        metadata={"year": 2024, "journal": "Diabetes Care", "evidence_level": "moderate",
                  "topic": "diabetes", "study_type": "cohort"}),
]

print(f"Loaded {len(PAPERS)} medical papers")
for p in PAPERS:
    print(f"  [{p.metadata['evidence_level']}] {p.metadata['year']} — {p.metadata['topic']} ({p.metadata['study_type']})")"""),

md("""## Step 2 — Index with Metadata"""),

code("""from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen3:8b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

index = VectorStoreIndex.from_documents(PAPERS, show_progress=True)
print("Medical literature index ready!")"""),

md("""## Step 3 — Query with Evidence Awareness"""),

code("""from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

# Basic query
engine = index.as_query_engine(similarity_top_k=3)

queries = [
    "What is the first-line treatment for type 2 diabetes?",
    "How effective is telemedicine compared to in-person visits?",
    "What role does AI play in medical imaging?",
    "What are the benefits of continuous glucose monitoring?",
]

for q in queries:
    response = engine.query(q)
    print(f"Q: {q}")
    print(f"A: {response}")
    sources = [(n.metadata.get("journal","?"), n.metadata.get("year","?")) for n in response.source_nodes]
    print(f"Sources: {sources}\\n")"""),

md("""## Step 4 — Filtered Search by Evidence Level"""),

code("""# Filter for high-evidence papers only
filters = MetadataFilters(filters=[
    MetadataFilter(key="evidence_level", value="high", operator=FilterOperator.EQ)
])

filtered_engine = index.as_query_engine(
    similarity_top_k=3,
    filters=filters,
)

response = filtered_engine.query("What treatments have the strongest evidence?")
print("High-evidence results only:")
print(response)"""),

md("""## Step 5 — Evidence Summary Report"""),

code("""report_query = \"\"\"Create a structured evidence summary for diabetes management with:
1. Treatment options with evidence levels
2. Key outcome metrics
3. Limitations of current evidence
4. Recommendations for further reading\"\"\"

response = engine.query(report_query)
print("Evidence Summary Report")
print("=" * 50)
print(response)"""),

md("""## What You Learned
- **Metadata-rich indexing** — year, journal, evidence level
- **Filtered retrieval** — searching by metadata attributes
- **Evidence-based prompting** — citing study types and quality
- **Medical literature search** — structured evidence summaries

## Next Steps: Add cross-paper synthesis (→ 13), hybrid search (→ 21)"""),
])

print("\n✅ Batch 2 complete — Projects 11-20 (Local RAG)")
