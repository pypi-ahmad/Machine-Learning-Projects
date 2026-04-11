"""Group 9 — Projects 81-90: Multimodal / OCR / Speech / VLM."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── Project 81: Local OCR + RAG Assistant ───────────────────────────
    paths.append(write_nb(9, "81_Local_OCR_RAG_Assistant", [
        md("# Project 81 — Local OCR + RAG Assistant\n## OCR Extraction → Chunking → Vector Search → Q&A\n\n**Stack:** LangChain · Ollama · ChromaDB · Pillow · Jupyter"),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb Pillow"),
        md("## Step 1 — Create Sample Document Image (Simulated OCR)"),
        code("""
from pathlib import Path

Path("sample_data").mkdir(exist_ok=True)

# Simulate OCR output (in production, use pytesseract or PaddleOCR)
ocr_texts = {
    "page_1.txt": \"\"\"INVOICE
Invoice Number: INV-2024-0042
Date: January 15, 2024
Bill To: Acme Corporation, 123 Business Ave, Suite 400

Item                    Qty    Price     Total
Cloud Hosting (Annual)   1    $12,000   $12,000
Premium Support          1    $3,000    $3,000
Data Migration Service   1    $5,000    $5,000

Subtotal: $20,000
Tax (8%): $1,600
Total Due: $21,600
Payment Terms: Net 30\"\"\",

    "page_2.txt": \"\"\"SERVICE AGREEMENT
This Service Level Agreement (SLA) between Acme Corporation
and CloudTech Solutions establishes the terms of service.

Uptime Guarantee: 99.9% monthly uptime
Response Time: Critical issues within 1 hour
Support Hours: 24/7 for critical, business hours for standard
Data Retention: 7 years minimum
Backup Frequency: Daily incremental, weekly full
Disaster Recovery: RTO 4 hours, RPO 1 hour\"\"\",

    "page_3.txt": \"\"\"QUARTERLY REPORT - Q4 2024
Revenue: $2.3M (up 15% QoQ)
Active Customers: 450 (up 12%)
Churn Rate: 2.1% (down from 3.4%)
NPS Score: 72 (up 8 points)
Key Wins: Enterprise deal with GlobalCorp ($500K ARR)
Challenges: Increased competition in mid-market segment
Next Quarter Focus: Product-led growth initiative\"\"\",
}

for filename, content in ocr_texts.items():
    Path(f"sample_data/{filename}").write_text(content, encoding="utf-8")
    print(f"Created: {filename} ({len(content)} chars)")
"""),
        md("## Step 2 — Build Vector Store from OCR Text"),
        code("""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil

llm = ChatOllama(model="qwen3:8b", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load OCR text as documents
docs = []
for filename, content in ocr_texts.items():
    docs.append(Document(
        page_content=content,
        metadata={"source": filename, "type": filename.split("_")[0]}
    ))

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)

shutil.rmtree("chroma_ocr", ignore_errors=True)
store = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_ocr")
retriever = store.as_retriever(search_kwargs={"k": 3})

print(f"Indexed {len(chunks)} chunks from {len(docs)} OCR pages")
"""),
        md("## Step 3 — Q&A Over OCR Documents"),
        code("""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the OCR-extracted document context. Cite the source page."),
    ("human", "Context:\\n{context}\\n\\nQuestion: {question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

questions = [
    "What is the total amount due on the invoice?",
    "What is the uptime guarantee in the SLA?",
    "What was the revenue in Q4 2024?",
    "What is the disaster recovery RTO?",
    "How many active customers are there?",
]

for q in questions:
    docs_found = retriever.invoke(q)
    context = "\\n---\\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in docs_found])
    answer = qa_chain.invoke({"context": context, "question": q})
    print(f"Q: {q}")
    print(f"A: {answer[:200]}")
    print()
"""),
        md("## What You Learned\n- **OCR-to-RAG pipeline** for scanned documents\n- **Document chunking** with source tracking\n- **Cited Q&A** over extracted text"),
    ]))

    # ── Project 82: Local Invoice Extraction Copilot ────────────────────
    paths.append(write_nb(9, "82_Local_Invoice_Extraction_Copilot", [
        md("# Project 82 — Local Invoice Extraction Copilot\n## Extract Structured Fields from Invoice Text\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic"),
        md("## Step 1 — Define Invoice Schema"),
        code("""
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    vendor: str
    bill_to: str
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total_due: float
    payment_terms: str

extractor = llm.with_structured_output(InvoiceData)
print("Invoice extractor ready!")
"""),
        md("## Step 2 — Extract from Sample Invoices"),
        code("""
invoices = [
    \"\"\"INVOICE #2024-0089
    Date: February 28, 2024
    From: DataPipe Solutions, 456 Tech Drive
    To: StartupXYZ, 789 Innovation Blvd

    Description              Qty   Price    Amount
    API Gateway License       1    $800     $800
    Data Processing (hrs)    40    $150     $6,000
    Training Workshop         2    $500     $1,000

    Subtotal: $7,800
    Sales Tax (7%): $546
    TOTAL: $8,346
    Due: Net 15\"\"\",

    \"\"\"Invoice Number: INV-5501
    Invoice Date: March 10, 2024
    Vendor: SecureCloud Inc
    Customer: MegaRetail Corp

    Professional Services — 80 hours @ $200/hr = $16,000
    Annual License — 1 unit @ $25,000 = $25,000
    Implementation Support — 1 @ $8,000 = $8,000

    Subtotal: $49,000
    Tax: $3,920
    Total Amount: $52,920
    Terms: Net 45\"\"\",
]

for i, inv_text in enumerate(invoices):
    print(f"\\n{'='*50}")
    print(f"Invoice {i+1}")
    print("="*50)
    result = extractor.invoke(f"Extract all fields from this invoice:\\n{inv_text}")
    print(f"  Invoice #: {result.invoice_number}")
    print(f"  Date:      {result.date}")
    print(f"  Vendor:    {result.vendor}")
    print(f"  Bill To:   {result.bill_to}")
    print(f"  Items:")
    for item in result.line_items:
        print(f"    {item.description}: {item.quantity} × ${item.unit_price} = ${item.total}")
    print(f"  Subtotal:  ${result.subtotal:,.2f}")
    print(f"  Tax:       ${result.tax:,.2f}")
    print(f"  Total:     ${result.total_due:,.2f}")
    print(f"  Terms:     {result.payment_terms}")
"""),
        md("## What You Learned\n- **Structured extraction** from unstructured invoice text\n- **Pydantic schemas** for complex nested data\n- **Field-level parsing** with local LLM"),
    ]))

    # ── Projects 83-90: Multimodal template projects ────────────────────
    mm_projects = [
        (83, "83_Local_Receipt_Intelligence", "Local Receipt Intelligence",
         "Categorize expenses from receipt text and generate spending reports",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

class ReceiptEntry(BaseModel):
    store: str
    date: str
    items: list[str]
    total: float
    category: str = Field(description="groceries, dining, transport, utilities, entertainment, other")
    tax: float

extractor = llm.with_structured_output(ReceiptEntry)

receipts = [
    "Walmart Supercenter\\nDate: 01/20/2025\\nMilk $3.99\\nBread $2.49\\nChicken $8.99\\nApples $4.50\\nTax: $1.60\\nTotal: $21.57",
    "Uber Eats\\n01/21/2025\\nPad Thai $14.99\\nSpring Rolls $6.99\\nDelivery $3.99\\nTax: $2.08\\nTotal: $28.05",
    "Shell Gas Station\\n01/22/2025\\n15.2 gal Regular @ $3.29\\nTax: $0.00\\nTotal: $50.01",
    "Netflix\\n01/22/2025\\nMonthly subscription\\nTax: $0.00\\nTotal: $15.99",
    "Target\\n01/23/2025\\nLaundry detergent $12.99\\nPaper towels $8.49\\nTrash bags $7.99\\nTax: $2.36\\nTotal: $31.83",
]

entries = []
for r in receipts:
    entry = extractor.invoke(f"Extract receipt data:\\n{r}")
    entries.append(entry.model_dump())
    print(f"  {entry.store}: ${entry.total:.2f} ({entry.category})")

df = pd.DataFrame(entries)
print(f"\\nSPENDING SUMMARY")
print(f"Total: ${df['total'].sum():.2f}")
print(f"By category:")
print(df.groupby('category')['total'].sum().sort_values(ascending=False).to_string())
"""),

        (84, "84_Local_Slide_Deck_Explainer", "Local Slide Deck Explainer",
         "Parse slide content and generate speaker notes & summaries",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

slides = [
    {"number": 1, "title": "Q1 2025 Strategy", "bullets": [
        "Revenue target: $5M", "3 new product launches", "Expand to 2 new markets"]},
    {"number": 2, "title": "Product Roadmap", "bullets": [
        "AI-powered analytics (March)", "Mobile app v2 (April)", "Enterprise SSO (May)"]},
    {"number": 3, "title": "Team Growth", "bullets": [
        "Hiring 15 engineers", "New VP of Sales", "Offshore QA team"]},
    {"number": 4, "title": "Financial Projections", "bullets": [
        "Q1: $1.1M → Q2: $1.3M → Q3: $1.4M → Q4: $1.2M",
        "Gross margin target: 72%", "Break-even by Q3"]},
]

notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate professional speaker notes for this presentation slide. "
     "Include: key talking points, transition to next slide, and a compelling narrative."),
    ("human", "Slide {number}: {title}\\nBullets: {bullets}")
])
notes_chain = notes_prompt | llm | StrOutputParser()

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a 2-sentence executive summary of the entire presentation."),
    ("human", "Presentation slides:\\n{all_slides}")
])
summary_chain = summary_prompt | llm | StrOutputParser()

print("SLIDE DECK ANALYSIS")
print("="*50)
for slide in slides:
    notes = notes_chain.invoke({
        "number": slide["number"],
        "title": slide["title"],
        "bullets": "\\n".join(slide["bullets"]),
    })
    print(f"\\nSlide {slide['number']}: {slide['title']}")
    print(f"Speaker Notes:\\n{notes[:300]}")

all_text = "\\n".join([f"Slide {s['number']}: {s['title']} — {', '.join(s['bullets'])}" for s in slides])
summary = summary_chain.invoke({"all_slides": all_text})
print(f"\\n{'='*50}")
print(f"EXECUTIVE SUMMARY: {summary}")
"""),

        (85, "85_Local_Image_Captioning", "Local Image Captioning",
         "Generate captions and alt-text using VLM or text-based description",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

# Simulate image descriptions (in production, use VLM like llava)
image_descriptions = [
    {"file": "team_photo.jpg",
     "visual": "Group of 8 people in business casual standing in modern office lobby, "
               "smiling, glass walls, plants, company logo on wall"},
    {"file": "product_screenshot.png",
     "visual": "Dashboard interface showing graphs, pie chart (blue/green), KPI cards, "
               "dark sidebar with navigation, header reading 'Analytics Pro'"},
    {"file": "architecture_diagram.png",
     "visual": "System diagram with boxes: User → API Gateway → Load Balancer → "
               "3 microservices → Database cluster, arrows showing data flow"},
]

class ImageCaption(BaseModel):
    short_caption: str = Field(description="One-line caption for social media")
    alt_text: str = Field(description="Accessible alt-text for screen readers")
    detailed_description: str = Field(description="Detailed description for documentation")
    tags: list[str]

captioner = llm.with_structured_output(ImageCaption)

for img in image_descriptions:
    print(f"\\nFile: {img['file']}")
    print(f"Visual input: {img['visual'][:60]}...")
    caption = captioner.invoke(f"Generate captions for this image:\\n{img['visual']}")
    print(f"  Caption:     {caption.short_caption}")
    print(f"  Alt-text:    {caption.alt_text}")
    print(f"  Description: {caption.detailed_description[:150]}...")
    print(f"  Tags:        {caption.tags}")
"""),

        (86, "86_Local_Chart_Understanding", "Local Chart Understanding",
         "Interpret chart data and explain trends from textual descriptions",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

# Chart data representations (simulating chart "reading")
charts = [
    {"title": "Monthly Revenue 2024",
     "type": "line",
     "data": "Jan:$100K, Feb:$120K, Mar:$95K, Apr:$140K, May:$160K, Jun:$155K, "
             "Jul:$170K, Aug:$180K, Sep:$175K, Oct:$200K, Nov:$220K, Dec:$250K"},
    {"title": "Customer Segments",
     "type": "pie",
     "data": "Enterprise:40%, Mid-market:35%, SMB:20%, Individual:5%"},
    {"title": "Support Ticket Volume",
     "type": "bar",
     "data": "Login Issues:450, Billing:320, Feature Requests:280, Bugs:200, Performance:150"},
]

class ChartAnalysis(BaseModel):
    key_insight: str
    trend: str = Field(description="increasing, decreasing, stable, mixed")
    notable_points: list[str]
    recommendation: str

analyzer = llm.with_structured_output(ChartAnalysis)

for chart in charts:
    print(f"\\nChart: {chart['title']} ({chart['type']})")
    print(f"Data: {chart['data'][:80]}...")
    analysis = analyzer.invoke(
        f"Analyze this {chart['type']} chart:\\n"
        f"Title: {chart['title']}\\nData: {chart['data']}"
    )
    print(f"  Trend:        {analysis.trend}")
    print(f"  Key Insight:  {analysis.key_insight}")
    print(f"  Notable:      {analysis.notable_points}")
    print(f"  Recommend:    {analysis.recommendation}")
"""),

        (87, "87_Local_Screenshot_Debugging", "Local Screenshot Debugging",
         "Analyze UI descriptions to identify bugs and suggest fixes",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

# Simulated UI screenshot descriptions
screenshots = [
    {"page": "Login Page",
     "description": "Login form with email and password fields. Submit button is grayed out "
                    "even though both fields are filled. Error message: 'Invalid email format' "
                    "but email looks correct: user@example.com. Console shows JS error."},
    {"page": "Dashboard",
     "description": "Main dashboard loads but the revenue chart shows 'No data' even though "
                    "the date range is set to 'Last 30 days'. Other widgets load correctly. "
                    "Network tab shows 500 error on /api/revenue endpoint."},
    {"page": "Settings",
     "description": "Settings page has overlapping text in the notification preferences section. "
                    "Toggle switches are misaligned. Save button extends beyond the card boundary. "
                    "Appears to be a CSS/responsive layout issue on smaller screens."},
]

class BugReport(BaseModel):
    severity: str = Field(description="critical, major, minor, cosmetic")
    bug_type: str = Field(description="functional, visual, performance, data")
    root_cause_hypothesis: str
    suggested_fix: str
    affected_component: str
    reproduction_steps: list[str]

debugger = llm.with_structured_output(BugReport)

for ss in screenshots:
    print(f"\\nPage: {ss['page']}")
    report = debugger.invoke(f"Analyze this UI screenshot for bugs:\\n{ss['description']}")
    print(f"  Severity:  {report.severity}")
    print(f"  Type:      {report.bug_type}")
    print(f"  Component: {report.affected_component}")
    print(f"  Cause:     {report.root_cause_hypothesis}")
    print(f"  Fix:       {report.suggested_fix}")
    print(f"  Steps:     {report.reproduction_steps}")
"""),

        (88, "88_Local_Audio_Transcription_Summary", "Local Audio Transcription + Summary",
         "Simulate Whisper transcription → summarization pipeline",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

# Simulated Whisper transcription output
# In production: whisper.load_model("base").transcribe("audio.mp3")
transcriptions = [
    {"file": "meeting_standup.wav", "duration": "5:30",
     "text": "Good morning everyone. So yesterday I finished the database migration. "
             "Today I'm working on the API endpoint tests. No blockers from my side. "
             "Alice, you mentioned the frontend was delayed? Yes, the design review took "
             "longer than expected. I'll have the components ready by Thursday. "
             "Bob, how about the deployment pipeline? I'm almost done. Just need to "
             "configure the staging environment. Should be ready for testing tomorrow."},
    {"file": "lecture_ml_101.wav", "duration": "12:00",
     "text": "Today we're going to talk about the fundamentals of machine learning. "
             "Machine learning is essentially teaching computers to learn from data. "
             "There are three main types: supervised learning where you have labeled data, "
             "unsupervised learning where you find patterns without labels, and reinforcement "
             "learning where an agent learns through trial and error. The key steps in any "
             "ML project are data collection, preprocessing, feature engineering, model "
             "selection, training, evaluation, and deployment."},
]

class TranscriptSummary(BaseModel):
    title: str
    duration: str
    speakers: list[str]
    key_points: list[str]
    action_items: list[str]
    topics: list[str]

summarizer = llm.with_structured_output(TranscriptSummary)

for t in transcriptions:
    print(f"\\n{'='*50}")
    print(f"File: {t['file']} ({t['duration']})")
    summary = summarizer.invoke(f"Summarize this audio transcription:\\n{t['text']}")
    print(f"  Title:   {summary.title}")
    print(f"  Speakers: {summary.speakers}")
    print(f"  Key Points:")
    for kp in summary.key_points:
        print(f"    • {kp}")
    if summary.action_items:
        print(f"  Action Items:")
        for ai in summary.action_items:
            print(f"    → {ai}")
    print(f"  Topics: {summary.topics}")
"""),

        (89, "89_Local_Voice_Notes_Organizer", "Local Voice Notes Organizer",
         "Cluster and tag voice memo transcripts by topic",
         """
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Simulated voice note transcripts
voice_notes = [
    {"id": 1, "text": "Need to buy groceries. Milk, eggs, bread, and those organic apples from Whole Foods."},
    {"id": 2, "text": "Project idea: build an app that tracks local farmers market schedules and inventory."},
    {"id": 3, "text": "Remember to call Dr. Johnson about the appointment next Tuesday."},
    {"id": 4, "text": "The new feature for the dashboard needs drag and drop. Look into React DnD."},
    {"id": 5, "text": "Gym routine: Monday chest, Wednesday back, Friday legs. Add more cardio."},
    {"id": 6, "text": "Meeting notes: team agreed to switch to bi-weekly sprints starting next month."},
    {"id": 7, "text": "Recipe for dinner: salmon with lemon butter, roasted asparagus, rice pilaf."},
    {"id": 8, "text": "Bug fix: the login timeout should be 30 minutes not 30 seconds."},
]

class NoteOrganization(BaseModel):
    note_id: int
    category: str = Field(description="personal, work, health, shopping, food, ideas")
    priority: str = Field(description="high, medium, low")
    actionable: bool
    tags: list[str]

organizer = llm.with_structured_output(NoteOrganization)

organized = []
for note in voice_notes:
    org = organizer.invoke(f"Categorize and tag this voice note:\\n{note['text']}")
    organized.append({**org.model_dump(), "text": note["text"][:50]})
    icon = "⚡" if org.priority == "high" else "●" if org.priority == "medium" else "○"
    print(f"  {icon} [{org.category}] {note['text'][:50]}... | tags: {org.tags}")

# Group by category
import pandas as pd
df = pd.DataFrame(organized)
print(f"\\nORGANIZATION SUMMARY")
print(df.groupby("category").size().sort_values(ascending=False).to_string())
print(f"\\nActionable items: {df['actionable'].sum()}/{len(df)}")
"""),

        (90, "90_Local_Multimodal_Research", "Local Multimodal Research Combiner",
         "Combine text analysis + image descriptions + data for research",
         """
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

# Research project combining multiple modalities
research_data = {
    "text_sources": [
        "Study shows 65% of consumers prefer sustainable packaging. "
        "The market for eco-friendly packaging grew 12% in 2024.",
        "Competitor analysis: Brand A uses 100% recycled materials. "
        "Brand B has committed to carbon-neutral packaging by 2026.",
    ],
    "chart_data": [
        "Consumer preference survey: Recyclable 45%, Biodegradable 30%, "
        "Reusable 15%, No preference 10%",
        "Cost comparison: Traditional $0.05/unit, Recycled $0.08/unit, "
        "Biodegradable $0.12/unit, Compostable $0.15/unit",
    ],
    "image_descriptions": [
        "Photo of ocean plastic pollution — hundreds of plastic bottles on beach",
        "Product mockup showing new biodegradable packaging design, kraft paper with green leaf logo",
    ],
}

class ResearchSynthesis(BaseModel):
    title: str
    executive_summary: str
    key_findings: list[str]
    evidence_strength: str = Field(description="strong, moderate, weak")
    recommendations: list[str]
    data_gaps: list[str]

synthesizer = llm.with_structured_output(ResearchSynthesis)

# Combine all sources
combined = (
    "TEXT SOURCES:\\n" + "\\n".join(research_data["text_sources"]) +
    "\\n\\nCHART DATA:\\n" + "\\n".join(research_data["chart_data"]) +
    "\\n\\nIMAGE CONTEXT:\\n" + "\\n".join(research_data["image_descriptions"])
)

synthesis = synthesizer.invoke(
    f"Synthesize this multimodal research on sustainable packaging:\\n\\n{combined}"
)

print("RESEARCH SYNTHESIS REPORT")
print("="*50)
print(f"Title: {synthesis.title}")
print(f"\\nExecutive Summary:\\n{synthesis.executive_summary}")
print(f"\\nEvidence Strength: {synthesis.evidence_strength}")
print(f"\\nKey Findings:")
for f in synthesis.key_findings:
    print(f"  • {f}")
print(f"\\nRecommendations:")
for r in synthesis.recommendations:
    print(f"  → {r}")
print(f"\\nData Gaps:")
for g in synthesis.data_gaps:
    print(f"  ? {g}")
"""),
    ]

    for proj_num, folder, title, desc, main_code in mm_projects:
        paths.append(write_nb(9, folder, [
            md(f"# Project {proj_num} — {title}\n## {desc}\n\n**Stack:** LangChain · Ollama · Jupyter"),
            code("# !pip install -q langchain langchain-ollama pandas pydantic"),
            md("## Implementation"),
            code(main_code),
            md(f"## What You Learned\n- **{title}** — {desc.lower()}\n- **Multimodal AI** with local models\n- **Structured extraction** from diverse data sources"),
        ]))

    print(f"Group 9 complete: {len(paths)} notebooks written")
    for p in paths:
        print(f"  ✓ {p}")
    return paths

if __name__ == "__main__":
    build()
