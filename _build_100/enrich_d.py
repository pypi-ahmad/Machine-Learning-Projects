"""Enrich batch D — Projects 82-90 (Multimodal/OCR/Speech) to 10+ cells each."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nb_helpers import md, code, write_nb

def build():
    paths = []

    # ── 82 — Invoice Extraction Copilot ─────────────────────────────────
    paths.append(write_nb(9, "82_Local_Invoice_Extraction_Copilot", [
        md("# Project 82 — Local Invoice Extraction Copilot\n## OCR Text → Structured Invoice → Validation → Export\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Simulate OCR Output"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd
from pathlib import Path

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

# Simulated OCR text from scanned invoices
ocr_outputs = [
    {
        "file": "invoice_001.pdf",
        "text": \"\"\"INVOICE
CloudTech Solutions LLC
123 Innovation Drive, Austin TX 78701

Bill To: DataDriven Inc.
Invoice #: INV-2025-0042
Date: February 10, 2025
Due: March 10, 2025

Description                    Qty    Rate       Amount
API Gateway Setup               1    $2,500     $2,500.00
Cloud Migration (hrs)          40      $150     $6,000.00
Security Audit                  1    $3,000     $3,000.00

                              Subtotal:         $11,500.00
                              Tax (8.25%):         $948.75
                              TOTAL:            $12,448.75

Payment Terms: Net 30
Bank: First National, Acct: ****4521\"\"\"
    },
    {
        "file": "invoice_002.pdf",
        "text": \"\"\"DataPipe Corp --- Invoice
To: Acme Analytics
INV# DP-8891  |  Date: 2025-01-15

- ETL Pipeline Development  120hrs x $125  = $15,000
- Data Quality Module        1 unit         = $4,500
- Training (2 days)          2 x $2,000     = $4,000

Subtotal: $23,500.00
Discount (10%): -$2,350.00
Total Due: $21,150.00
Due by: Feb 15, 2025\"\"\"
    },
    {
        "file": "receipt_003.jpg",
        "text": \"\"\"Quick Mart
Date: 03/05/2025 14:32
Cashier: Mike

Coffee Large         $4.99
Sandwich             $8.49
Water                $1.99
Cookie               $2.49

Subtotal            $17.96
Tax                  $1.48
Total               $19.44
VISA ****7812\"\"\"
    },
]
print(f"OCR documents to process: {len(ocr_outputs)}")
"""),
        md("## Step 2 — Define Extraction Schemas"),
        code("""
class LineItem(BaseModel):
    description: str
    quantity: float = 1
    unit_price: float = 0
    amount: float

class InvoiceData(BaseModel):
    document_type: str = Field(description="invoice, receipt, purchase_order")
    invoice_number: str = ""
    date: str
    due_date: str = ""
    vendor_name: str
    vendor_address: str = ""
    bill_to: str = ""
    line_items: list[LineItem]
    subtotal: float
    tax: float = 0
    discount: float = 0
    total: float
    payment_method: str = ""
    currency: str = "USD"

extractor = llm.with_structured_output(InvoiceData)
print("Schema: InvoiceData with LineItem sub-model")
print(f"Fields: {len(InvoiceData.model_fields)}")
"""),
        md("## Step 3 — Extract All Invoices"),
        code("""
extracted = []
for doc in ocr_outputs:
    try:
        invoice = extractor.invoke(
            f"Extract structured invoice data from this OCR text:\\n\\n{doc['text']}"
        )
        extracted.append({"file": doc["file"], "data": invoice, "error": None})
        print(f"✓ {doc['file']}: {invoice.vendor_name} | "
              f"{len(invoice.line_items)} items | ${invoice.total:,.2f}")
    except Exception as e:
        extracted.append({"file": doc["file"], "data": None, "error": str(e)})
        print(f"✗ {doc['file']}: {e}")
"""),
        md("## Step 4 — Validation & Anomaly Detection"),
        code("""
print("VALIDATION REPORT")
print("=" * 60)
for inv in extracted:
    if inv["data"] is None:
        print(f"✗ {inv['file']}: extraction failed — {inv['error']}")
        continue
    d = inv["data"]
    issues = []

    # Check math
    calc_subtotal = sum(i.amount for i in d.line_items)
    if abs(calc_subtotal - d.subtotal) > 1.0:
        issues.append(f"Subtotal mismatch: items sum to ${calc_subtotal:.2f}, stated ${d.subtotal:.2f}")

    # Check total = subtotal + tax - discount
    expected_total = d.subtotal + d.tax - d.discount
    if abs(expected_total - d.total) > 1.0:
        issues.append(f"Total mismatch: expected ${expected_total:.2f}, stated ${d.total:.2f}")

    # Check missing fields
    if not d.invoice_number:
        issues.append("Missing invoice number")
    if not d.due_date and d.document_type == "invoice":
        issues.append("Missing due date")

    status = "✓ VALID" if not issues else f"⚠ {len(issues)} issue(s)"
    print(f"\\n{inv['file']}: {status}")
    for issue in issues:
        print(f"  → {issue}")
"""),
        md("## Step 5 — Export & Summary"),
        code("""
# Export to structured format
Path("sample_data").mkdir(exist_ok=True)
export = [{"file": e["file"], **e["data"].model_dump()} for e in extracted if e["data"]]
with open("sample_data/extracted_invoices.json", "w") as f:
    json.dump(export, f, indent=2, default=str)

# Summary table
rows = []
for e in extracted:
    if e["data"]:
        rows.append({
            "file": e["file"], "vendor": e["data"].vendor_name,
            "type": e["data"].document_type, "total": e["data"].total,
            "items": len(e["data"].line_items),
        })
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\\nTotal across all documents: ${df['total'].sum():,.2f}")
"""),
        md("## What You Learned\n- **OCR → structured extraction** with Pydantic schemas\n- **Line-item parsing** from messy text\n- **Math validation** (subtotal, tax, discount, total)\n- **Anomaly detection** and data quality checks"),
    ]))

    # ── 83 — Receipt Intelligence ───────────────────────────────────────
    paths.append(write_nb(9, "83_Local_Receipt_Intelligence", [
        md("# Project 83 — Local Receipt Intelligence\n## Parse Receipts → Categorize → Budget Analysis\n\n**Stack:** LangChain · Ollama · Pydantic · pandas · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Receipt Corpus"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import pandas as pd, json

llm = ChatOllama(model="qwen3:8b", temperature=0.0)

receipts = [
    "Whole Foods Market 03/01/2025 Organic Apples $5.99 Almond Milk $4.49 "
    "Quinoa $6.99 Chicken Breast $12.49 Total: $29.96",
    "Shell Gas Station 03/02/2025 Regular Unleaded 12.5gal @$3.29 = $41.13 "
    "Car Wash $8.00 Total: $49.13",
    "Amazon.com Order 03/03/2025 Wireless Mouse $24.99 USB Hub $15.99 "
    "HDMI Cable $9.99 Shipping: Free Total: $50.97",
    "Starbucks 03/04/2025 Grande Latte $5.75 Croissant $3.95 Total: $9.70",
    "CVS Pharmacy 03/05/2025 Ibuprofen $8.99 Vitamins $12.49 "
    "Band-Aids $4.99 Total: $26.47",
    "Netflix 03/01/2025 Monthly subscription Standard plan Total: $15.49",
    "Con Edison 03/01/2025 Electricity February 2025 Usage: 650kWh Total: $98.50",
    "Planet Fitness 03/01/2025 Monthly membership Classic Total: $10.00",
]
print(f"Receipts to process: {len(receipts)}")
"""),
        md("## Step 2 — Extract & Categorize"),
        code("""
class ReceiptItem(BaseModel):
    name: str
    price: float

class Receipt(BaseModel):
    merchant: str
    date: str
    category: str = Field(description="groceries, gas, electronics, food_drink, health, subscription, utilities, fitness")
    items: list[ReceiptItem]
    total: float
    payment_type: str = Field(default="unknown", description="cash, credit, debit, unknown")

extractor = llm.with_structured_output(Receipt)

parsed = []
for text in receipts:
    r = extractor.invoke(f"Extract receipt data:\\n{text}")
    parsed.append(r)
    print(f"  {r.merchant:<25} {r.category:<15} ${r.total:>8.2f}  ({len(r.items)} items)")
"""),
        md("## Step 3 — Budget Analysis"),
        code("""
df = pd.DataFrame([r.model_dump() for r in parsed])
df["items_count"] = df["items"].apply(len)

print("SPENDING SUMMARY")
print("=" * 50)
by_cat = df.groupby("category")["total"].agg(["sum", "count", "mean"]).round(2)
by_cat.columns = ["total_spent", "transactions", "avg_per_txn"]
by_cat = by_cat.sort_values("total_spent", ascending=False)
print(by_cat.to_string())
print(f"\\nGrand total: ${df['total'].sum():.2f}")
print(f"Average transaction: ${df['total'].mean():.2f}")

# Budget limits
budgets = {"groceries": 200, "food_drink": 50, "subscription": 30, "gas": 100, "electronics": 100}
print("\\nBUDGET CHECK:")
for cat, limit in budgets.items():
    spent = df[df["category"] == cat]["total"].sum()
    pct = spent / limit * 100 if limit > 0 else 0
    status = "✓ OK" if spent <= limit else "⚠ OVER"
    print(f"  {cat:<15} ${spent:>8.2f} / ${limit:.2f} ({pct:.0f}%) {status}")
"""),
        md("## Step 4 — Spending Insights"),
        code("""
insight_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a personal finance advisor. Analyze spending and give 3 actionable tips."),
    ("human", "Spending data:\\n{data}\\n\\nBudgets:\\n{budgets}\\n\\nGive 3 specific tips:")
])
insight_chain = insight_prompt | llm | StrOutputParser()

tips = insight_chain.invoke({
    "data": by_cat.to_string(),
    "budgets": json.dumps(budgets),
})
print("SPENDING INSIGHTS")
print("=" * 50)
print(tips[:600])
"""),
        md("## What You Learned\n- **Receipt parsing** with structured extraction\n- **Auto-categorization** of expenses\n- **Budget tracking** with variance analysis\n- **AI spending insights** for personal finance"),
    ]))

    # ── 84 — Slide Deck Explainer ───────────────────────────────────────
    paths.append(write_nb(9, "84_Local_Slide_Deck_Explainer", [
        md("# Project 84 — Local Slide Deck Explainer\n## Slide Text → Structured Notes → Speaker Script\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Simulated Slide Deck"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

slides = [
    {"slide": 1, "title": "Q1 2025 Results",
     "content": "Revenue: $4.2M (+18% YoY)\\nARR: $15.8M\\nNet retention: 112%\\nNew logos: 23"},
    {"slide": 2, "title": "Product Highlights",
     "content": "Launched AI Assistant (Jan)\\nMobile app v3.0 (Feb)\\nAPI v2 with webhooks (Mar)\\n"
                "NPS improved from 42 to 58"},
    {"slide": 3, "title": "Customer Wins",
     "content": "Enterprise: TechCorp ($500K ACV), DataFlow ($350K ACV)\\n"
                "Mid-Market: 12 new accounts ($1.2M pipeline)\\nChurn: 3 accounts ($180K lost)"},
    {"slide": 4, "title": "Engineering Metrics",
     "content": "Deployment frequency: 4x/week\\nMTTR: 45min (prev 2hrs)\\n"
                "Test coverage: 82%\\nTech debt ratio: 15% of sprint capacity"},
    {"slide": 5, "title": "Q2 Roadmap",
     "content": "AI-powered analytics dashboard\\nSOC2 Type II certification\\n"
                "European data center\\nSelf-serve onboarding\\nTarget: $5M revenue"},
]
print(f"Slide deck: {len(slides)} slides")
for s in slides:
    print(f"  Slide {s['slide']}: {s['title']}")
"""),
        md("## Step 2 — Generate Structured Notes"),
        code("""
class SlideAnalysis(BaseModel):
    slide_number: int
    title: str
    key_points: list[str]
    metrics: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    audience_questions: list[str] = Field(default_factory=list, description="Questions audience might ask")

analyzer = llm.with_structured_output(SlideAnalysis)

analyses = []
for slide in slides:
    analysis = analyzer.invoke(
        f"Analyze this presentation slide:\\n"
        f"Title: {slide['title']}\\nContent: {slide['content']}"
    )
    analyses.append(analysis)
    print(f"\\nSlide {analysis.slide_number}: {analysis.title}")
    print(f"  Key points: {len(analysis.key_points)}")
    print(f"  Metrics: {analysis.metrics[:2]}")
    print(f"  Potential questions: {analysis.audience_questions[:2]}")
"""),
        md("## Step 3 — Generate Speaker Script"),
        code("""
script_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a natural speaker script for presenting this slide. "
     "Include transitions, emphasis points, and timing notes. Target 60 seconds per slide."),
    ("human", "Slide {num}: {title}\\nKey Points: {points}\\n\\nScript:")
])
script_chain = script_prompt | llm | StrOutputParser()

full_script = []
for analysis in analyses:
    script = script_chain.invoke({
        "num": analysis.slide_number,
        "title": analysis.title,
        "points": "; ".join(analysis.key_points[:4]),
    })
    full_script.append({"slide": analysis.slide_number, "script": script})
    print(f"\\n--- Slide {analysis.slide_number} Script ---")
    print(script[:250] + "...")
"""),
        md("## Step 4 — Q&A Preparation"),
        code("""
all_questions = []
for a in analyses:
    for q in a.audience_questions:
        all_questions.append({"slide": a.slide_number, "question": q})

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Prepare a concise, confident answer for a presentation Q&A."),
    ("human", "Context: {title}\\nQuestion: {question}\\nAnswer:")
])
qa_chain = qa_prompt | llm | StrOutputParser()

print("Q&A PREP SHEET")
print("=" * 50)
for qa in all_questions[:6]:
    slide = slides[qa["slide"] - 1]
    answer = qa_chain.invoke({"title": slide["title"], "question": qa["question"]})
    print(f"\\n  Q: {qa['question']}")
    print(f"  A: {answer[:150]}")
"""),
        md("## What You Learned\n- **Slide deck analysis** with structured extraction\n- **Speaker script generation** with timing\n- **Q&A preparation** for anticipated questions\n- **Presentation intelligence** from raw slide content"),
    ]))

    # ── 85 — Image Captioning ──────────────────────────────────────────
    paths.append(write_nb(9, "85_Local_Image_Captioning", [
        md("# Project 85 — Local Image Captioning Pipeline\n## Scene Description → Detailed Caption → Alt-Text → SEO Tags\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter\n\n*Note: Uses text-based scene simulation since Ollama doesn't support vision natively.*"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Scene Descriptions (simulating image input)"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.3)

scenes = [
    {"id": "img_001", "description": "A golden retriever playing fetch on a sandy beach at sunset, "
     "waves crashing in the background, owner visible in the distance"},
    {"id": "img_002", "description": "A busy coffee shop interior with exposed brick walls, "
     "people working on laptops, barista making latte art, warm lighting"},
    {"id": "img_003", "description": "A mountain landscape with snow-capped peaks, "
     "pine forest in the foreground, a clear blue lake reflecting the mountains"},
    {"id": "img_004", "description": "A modern kitchen with marble countertops, someone "
     "preparing a colorful salad, fresh vegetables on a cutting board"},
    {"id": "img_005", "description": "A street market in Asia with colorful food stalls, "
     "lanterns hanging overhead, crowds of people browsing"},
]
print(f"Scenes to caption: {len(scenes)}")
"""),
        md("## Step 2 — Multi-Format Caption Generation"),
        code("""
class ImageCaptions(BaseModel):
    short_caption: str = Field(description="1 sentence, 10-15 words")
    detailed_caption: str = Field(description="2-3 sentences with scene details")
    alt_text: str = Field(description="Accessible alt text for screen readers, 125 chars max")
    seo_tags: list[str] = Field(description="5-8 SEO keywords/tags")
    mood: str = Field(description="emotional tone: warm, serene, energetic, etc.")
    color_palette: list[str] = Field(description="3-5 dominant colors")

captioner = llm.with_structured_output(ImageCaptions)

caption_results = []
for scene in scenes:
    captions = captioner.invoke(
        f"Generate captions for an image showing: {scene['description']}"
    )
    caption_results.append({"id": scene["id"], "scene": scene["description"], "captions": captions})
    print(f"\\n{scene['id']}:")
    print(f"  Short: {captions.short_caption}")
    print(f"  Alt: {captions.alt_text[:80]}...")
    print(f"  Tags: {captions.seo_tags[:5]}")
    print(f"  Mood: {captions.mood} | Colors: {captions.color_palette}")
"""),
        md("## Step 3 — Social Media Post Generator"),
        code("""
social_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a social media post for this image. Include relevant hashtags. "
     "Match the platform style."),
    ("human", "Image: {description}\\nPlatform: {platform}\\nPost:")
])
social_chain = social_prompt | llm | StrOutputParser()

platforms = ["Instagram", "Twitter", "LinkedIn"]
for scene in scenes[:3]:
    print(f"\\n{'='*50}")
    print(f"Scene: {scene['description'][:50]}...")
    for platform in platforms:
        post = social_chain.invoke({
            "description": scene["description"],
            "platform": platform,
        })
        print(f"\\n  [{platform}] {post[:120]}...")
"""),
        md("## Step 4 — Accessibility Report"),
        code("""
print("ACCESSIBILITY REPORT")
print("=" * 50)
for cr in caption_results:
    alt = cr["captions"].alt_text
    issues = []
    if len(alt) > 125:
        issues.append(f"Alt text too long ({len(alt)} chars, max 125)")
    if len(alt) < 20:
        issues.append("Alt text too short")
    if not any(c.isalpha() for c in alt):
        issues.append("Alt text has no descriptive words")

    status = "✓" if not issues else "⚠"
    print(f"  {status} {cr['id']}: {alt[:60]}... ({len(alt)} chars)")
    for issue in issues:
        print(f"    → {issue}")

# Export
from pathlib import Path
Path("sample_data").mkdir(exist_ok=True)
export = [{"id": cr["id"], **cr["captions"].model_dump()} for cr in caption_results]
with open("sample_data/image_captions.json", "w") as f:
    json.dump(export, f, indent=2)
print(f"\\n✓ Exported {len(export)} captions")
"""),
        md("## What You Learned\n- **Multi-format captioning** (short, detailed, alt-text, SEO)\n- **Social media adaptation** per platform\n- **Accessibility guidelines** enforcement\n- **Structured metadata** from visual content"),
    ]))

    # ── 86 — Chart Understanding ───────────────────────────────────────
    paths.append(write_nb(9, "86_Local_Chart_Understanding", [
        md("# Project 86 — Local Chart Understanding Agent\n## Data Table → Chart Synthesis → Insight Extraction\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Simulated Chart Data"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

charts = [
    {
        "type": "bar_chart", "title": "Monthly Revenue by Product",
        "data": {"Product A": [120, 135, 150, 148, 165, 180],
                 "Product B": [80, 85, 90, 95, 110, 105],
                 "Product C": [45, 50, 55, 60, 65, 70]},
        "x_axis": "Months (Jan-Jun)", "y_axis": "Revenue ($K)",
    },
    {
        "type": "line_chart", "title": "User Growth Trend",
        "data": {"DAU": [5200, 5800, 6100, 7200, 8500, 9100, 10200, 11500],
                 "MAU": [15000, 16500, 17800, 19200, 21000, 23500, 26000, 28500]},
        "x_axis": "Months (Jan-Aug)", "y_axis": "Users",
    },
    {
        "type": "pie_chart", "title": "Customer Segments",
        "data": {"Enterprise": 35, "Mid-Market": 28, "SMB": 22, "Startup": 15},
        "note": "By revenue contribution (%)",
    },
]
print(f"Charts to analyze: {len(charts)}")
for c in charts:
    print(f"  {c['type']}: {c['title']}")
"""),
        md("## Step 2 — Structured Chart Analysis"),
        code("""
class DataTrend(BaseModel):
    direction: str = Field(description="increasing, decreasing, stable, fluctuating")
    growth_rate: str = Field(description="percentage or description")
    notable_points: list[str]

class ChartInsight(BaseModel):
    chart_type: str
    title: str
    key_finding: str
    trends: list[DataTrend]
    comparisons: list[str]
    anomalies: list[str]
    business_implication: str
    confidence: float = Field(ge=0, le=1)

analyzer = llm.with_structured_output(ChartInsight)

analyses = []
for chart in charts:
    insight = analyzer.invoke(
        f"Analyze this chart:\\n"
        f"Type: {chart['type']}\\nTitle: {chart['title']}\\n"
        f"Data: {json.dumps(chart['data'])}\\n"
        f"X-axis: {chart.get('x_axis','')} | Y-axis: {chart.get('y_axis','')}"
    )
    analyses.append(insight)
    print(f"\\n{insight.title}:")
    print(f"  Finding: {insight.key_finding}")
    print(f"  Trends: {len(insight.trends)} detected")
    print(f"  Anomalies: {insight.anomalies[:2]}")
    print(f"  Implication: {insight.business_implication[:80]}")
"""),
        md("## Step 3 — Executive Summary Generation"),
        code("""
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a 3-paragraph executive summary of these chart analyses. "
     "Use specific numbers. Highlight risks and opportunities."),
    ("human", "Chart analyses:\\n{analyses}")
])
summary_chain = summary_prompt | llm | StrOutputParser()

exec_summary = summary_chain.invoke({
    "analyses": json.dumps([a.model_dump() for a in analyses], indent=2)
})
print("EXECUTIVE SUMMARY")
print("=" * 50)
print(exec_summary[:600])
"""),
        md("## Step 4 — Recommendation Engine"),
        code("""
rec_prompt = ChatPromptTemplate.from_messages([
    ("system", "Based on chart data, provide 5 specific, actionable recommendations. "
     "Each should reference specific data points."),
    ("human", "Data:\\n{data}\\n\\nInsights:\\n{insights}\\n\\n5 Recommendations:")
])
rec_chain = rec_prompt | llm | StrOutputParser()

recommendations = rec_chain.invoke({
    "data": json.dumps([c["data"] for c in charts], default=str),
    "insights": json.dumps([a.key_finding for a in analyses]),
})
print("RECOMMENDATIONS")
print("=" * 50)
print(recommendations[:500])
"""),
        md("## What You Learned\n- **Data-to-insight extraction** from chart representations\n- **Trend detection** with direction and rate\n- **Executive summary** generation from multiple charts\n- **Actionable recommendations** grounded in data"),
    ]))

    # ── 87 — Screenshot Debugging ──────────────────────────────────────
    paths.append(write_nb(9, "87_Local_Screenshot_Debugging", [
        md("# Project 87 — Local Screenshot Debugging Agent\n## Error Screenshot → Diagnosis → Fix Suggestion\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter\n\n*Uses text-based error simulation for local-first approach.*"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Simulated Error Screenshots (as text)"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.1)

error_screens = [
    {
        "id": "err_001",
        "context": "Python web application",
        "error_text": \"\"\"Traceback (most recent call last):
  File "app.py", line 42, in handle_request
    result = db.query(User).filter_by(email=email).first()
  File "sqlalchemy/orm/query.py", line 3425, in first
    return self._iter().first()
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError)
could not connect to server: Connection refused
    Is the server running on host "localhost" and accepting
    TCP/IP connections on port 5432?\"\"\"
    },
    {
        "id": "err_002",
        "context": "React frontend build",
        "error_text": \"\"\"ERROR in ./src/components/Dashboard.tsx
Module not found: Error: Can't resolve '@/utils/formatDate' in
'/app/src/components'
  @ ./src/components/Dashboard.tsx 3:0-45
  @ ./src/App.tsx
  @ ./src/index.tsx

webpack compiled with 1 error\"\"\"
    },
    {
        "id": "err_003",
        "context": "Docker deployment",
        "error_text": \"\"\"Error response from daemon: driver failed programming external
connectivity on endpoint api-server: Bind for 0.0.0.0:8080 failed:
port is already allocated.

docker: Error response from daemon: OCI runtime create failed:
container_linux.go:349: starting container process caused:
exec: "python": executable file not found in $PATH\"\"\"
    },
    {
        "id": "err_004",
        "context": "Kubernetes pod crash",
        "error_text": \"\"\"NAME          READY   STATUS             RESTARTS   AGE
api-server    0/1     CrashLoopBackOff   5          10m

Events:
  Warning  BackOff  2m  kubelet  Back-off restarting failed container
  Warning  Failed   3m  kubelet  Error: secret "db-credentials" not found\"\"\"
    },
]
print(f"Error screens to debug: {len(error_screens)}")
"""),
        md("## Step 2 — Automated Diagnosis"),
        code("""
class Diagnosis(BaseModel):
    error_type: str
    root_cause: str
    severity: str = Field(description="critical, high, medium, low")
    affected_component: str
    fix_steps: list[str]
    verification_command: str
    prevention_tip: str

debugger = llm.with_structured_output(Diagnosis)

diagnoses = []
for screen in error_screens:
    diag = debugger.invoke(
        f"Debug this error from a {screen['context']}:\\n\\n{screen['error_text']}"
    )
    diagnoses.append({"id": screen["id"], "context": screen["context"], "diagnosis": diag})
    print(f"\\n{screen['id']} ({screen['context']}):")
    print(f"  Type: {diag.error_type}")
    print(f"  Root cause: {diag.root_cause}")
    print(f"  Severity: {diag.severity}")
    print(f"  Fix steps: {len(diag.fix_steps)}")
"""),
        md("## Step 3 — Generate Fix Scripts"),
        code("""
fix_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate a bash script that fixes this error. Include comments. "
     "Make it safe to run (check before acting)."),
    ("human", "Error: {error}\\nRoot cause: {cause}\\nFix steps: {steps}\\n\\nBash fix script:")
])
fix_chain = fix_prompt | llm | StrOutputParser()

for d in diagnoses[:3]:
    diag = d["diagnosis"]
    fix_script = fix_chain.invoke({
        "error": diag.error_type,
        "cause": diag.root_cause,
        "steps": "\\n".join(f"- {s}" for s in diag.fix_steps),
    })
    print(f"\\n--- Fix for {d['id']} ---")
    print(fix_script[:300])
    print("...")
"""),
        md("## Step 4 — Knowledge Base Builder"),
        code("""
# Build a reusable debugging knowledge base
print("DEBUGGING KNOWLEDGE BASE")
print("=" * 50)
kb_entries = []
for d in diagnoses:
    diag = d["diagnosis"]
    entry = {
        "error_type": diag.error_type,
        "component": diag.affected_component,
        "severity": diag.severity,
        "root_cause": diag.root_cause,
        "fix": diag.fix_steps[0] if diag.fix_steps else "N/A",
        "prevention": diag.prevention_tip,
        "verify": diag.verification_command,
    }
    kb_entries.append(entry)
    print(f"\\n  [{diag.severity.upper()}] {diag.error_type}")
    print(f"  Cause: {diag.root_cause}")
    print(f"  Fix: {diag.fix_steps[0] if diag.fix_steps else 'N/A'}")
    print(f"  Prevent: {diag.prevention_tip}")

from pathlib import Path
Path("sample_data").mkdir(exist_ok=True)
with open("sample_data/debug_knowledge_base.json", "w") as f:
    json.dump(kb_entries, f, indent=2)
print(f"\\n✓ Saved {len(kb_entries)} entries to knowledge base")
"""),
        md("## What You Learned\n- **Error pattern recognition** from screenshots/logs\n- **Automated root cause analysis** with severity\n- **Fix script generation** with safety checks\n- **Knowledge base building** from resolved issues"),
    ]))

    # ── 88 — Audio Transcription Summary ────────────────────────────────
    paths.append(write_nb(9, "88_Local_Audio_Transcription_Summary", [
        md("# Project 88 — Local Audio Transcription Summary\n## Transcript → Structured Notes → Action Items\n\n**Stack:** LangChain · Ollama · Pydantic · Jupyter\n\n*Uses pre-existing transcript text to demonstrate the summary pipeline.*"),
        code("# !pip install -q langchain langchain-ollama pydantic pandas"),
        md("## Step 1 — Sample Meeting Transcripts"),
        code("""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.2)

transcripts = [
    {
        "title": "Q1 Sprint Retrospective",
        "duration": "45 min",
        "text": \"\"\"
Sarah: Good morning everyone. Let's start with what went well this sprint.
Mike: The new API integration shipped ahead of schedule. Team collaboration was great.
Alice: Agreed, the pair programming sessions really helped. Code review turnaround dropped from 2 days to 4 hours.
Sarah: Great. What didn't go well?
Bob: The deployment pipeline broke twice. We need better monitoring.
Mike: Also, the requirements for the dashboard feature changed three times mid-sprint.
Sarah: Let's talk about action items.
Alice: I'll set up PagerDuty alerts for the pipeline. Due by Friday.
Bob: I'll draft a change request process document. Should have it by next Wednesday.
Sarah: Mike, can you lead the requirements workshop with product?
Mike: Sure, I'll schedule it for next Monday.
Sarah: Perfect. Any other concerns?
Bob: We should probably upgrade our test infrastructure. The CI builds are taking 45 minutes now.
Sarah: Good point. Let's add that to the next sprint backlog. Meeting adjourned.\"\"\"
    },
    {
        "title": "Product Strategy Review",
        "duration": "30 min",
        "text": \"\"\"
CEO: Let's review our Q2 product strategy. Revenue target is $5M.
VP Product: We have three main initiatives: AI features, enterprise tier, and mobile app.
CEO: What's the AI features timeline?
VP Eng: Beta in April, GA in May. We need two more ML engineers.
VP Product: The enterprise tier is our biggest revenue opportunity — $2M projected.
CEO: What's blocking enterprise?
VP Product: SOC2 compliance. We're about 60% through the audit.
VP Eng: Should be complete by end of March if we prioritize it.
CEO: Make it top priority. What about mobile?
VP Product: MVP is on track for June. We're using React Native for cross-platform.
CEO: Good. Hiring — VP Eng, how many do we need?
VP Eng: Two ML engineers, one DevOps, and one security engineer. Budget impact: $600K annually.
CEO: Approved. Let's reconvene in two weeks with progress updates.\"\"\"
    },
]
print(f"Transcripts to process: {len(transcripts)}")
"""),
        md("## Step 2 — Structured Meeting Notes"),
        code("""
class ActionItem(BaseModel):
    owner: str
    task: str
    deadline: str
    priority: str = Field(description="high, medium, low")

class MeetingNotes(BaseModel):
    title: str
    participants: list[str]
    duration: str
    summary: str = Field(description="2-3 sentence overview")
    key_decisions: list[str]
    action_items: list[ActionItem]
    open_questions: list[str]
    risks: list[str]
    next_steps: str

note_extractor = llm.with_structured_output(MeetingNotes)

meeting_notes = []
for transcript in transcripts:
    notes = note_extractor.invoke(
        f"Extract structured meeting notes:\\n"
        f"Title: {transcript['title']}\\n"
        f"Duration: {transcript['duration']}\\n\\n"
        f"Transcript:\\n{transcript['text']}"
    )
    meeting_notes.append(notes)
    print(f"\\n{'='*50}")
    print(f"📋 {notes.title}")
    print(f"Participants: {', '.join(notes.participants)}")
    print(f"Summary: {notes.summary}")
    print(f"\\nDecisions: {len(notes.key_decisions)}")
    for d in notes.key_decisions:
        print(f"  • {d}")
    print(f"\\nAction Items: {len(notes.action_items)}")
    for a in notes.action_items:
        print(f"  [{a.priority}] {a.owner}: {a.task} (due: {a.deadline})")
"""),
        md("## Step 3 — Follow-Up Email Draft"),
        code("""
email_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a professional follow-up email summarizing the meeting. "
     "Include action items with owners and deadlines. Keep it concise."),
    ("human", "Meeting: {title}\\nNotes: {notes}\\n\\nEmail:")
])
email_chain = email_prompt | llm | StrOutputParser()

for notes in meeting_notes:
    email = email_chain.invoke({
        "title": notes.title,
        "notes": json.dumps(notes.model_dump(), indent=2),
    })
    print(f"\\n--- Email for '{notes.title}' ---")
    print(email[:400])
    print("...")
"""),
        md("## Step 4 — Action Item Tracker"),
        code("""
import pandas as pd

all_actions = []
for notes in meeting_notes:
    for a in notes.action_items:
        all_actions.append({
            "meeting": notes.title[:25],
            "owner": a.owner,
            "task": a.task[:40],
            "deadline": a.deadline,
            "priority": a.priority,
            "status": "pending",
        })

df = pd.DataFrame(all_actions)
print("ACTION ITEM TRACKER")
print("=" * 60)
print(df.to_string(index=False))

print(f"\\nSummary:")
print(f"  Total items: {len(df)}")
print(f"  By priority: {df['priority'].value_counts().to_dict()}")
print(f"  By owner: {df['owner'].value_counts().to_dict()}")
"""),
        md("## What You Learned\n- **Transcript → structured notes** extraction\n- **Action item extraction** with owners and deadlines\n- **Follow-up email** generation\n- **Meeting intelligence tracking** across meetings"),
    ]))

    # ── 89 — Voice Notes Organizer ──────────────────────────────────────
    paths.append(write_nb(9, "89_Local_Voice_Notes_Organizer", [
        md("# Project 89 — Local Voice Notes Organizer\n## Raw Notes → Categorize → Link → Search\n\n**Stack:** LangChain · Ollama · ChromaDB · Jupyter"),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb pydantic"),
        md("## Step 1 — Simulated Voice Notes"),
        code("""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json

llm = ChatOllama(model="qwen3:8b", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

voice_notes = [
    {"id": "vn_001", "timestamp": "2025-03-01 09:15",
     "text": "Idea for the app — we should add a dark mode toggle in settings. "
             "Users have been requesting it. Check the GitHub issues for upvotes."},
    {"id": "vn_002", "timestamp": "2025-03-01 14:30",
     "text": "Meeting with Sarah went well. She approved the Q2 budget — we got "
             "an extra $50K for cloud infrastructure. Need to update the proposal."},
    {"id": "vn_003", "timestamp": "2025-03-02 08:00",
     "text": "Remember to buy groceries — milk, eggs, bread, chicken. Also pick up "
             "dry cleaning before 6pm."},
    {"id": "vn_004", "timestamp": "2025-03-02 11:45",
     "text": "The caching bug is in the Redis layer. When TTL expires during a write "
             "operation, we get stale data. Need to implement write-through cache."},
    {"id": "vn_005", "timestamp": "2025-03-03 16:00",
     "text": "Book recommendation from Dave — 'Designing Data-Intensive Applications' "
             "by Martin Kleppmann. Good for understanding distributed systems."},
    {"id": "vn_006", "timestamp": "2025-03-03 19:30",
     "text": "Workout plan for this week: Monday legs, Wednesday upper body, "
             "Friday cardio and core. Remember to stretch before each session."},
    {"id": "vn_007", "timestamp": "2025-03-04 10:00",
     "text": "API rate limiting should be 100 requests per minute per user. "
             "Use token bucket algorithm. Need to discuss with the team tomorrow."},
]
print(f"Voice notes: {len(voice_notes)}")
"""),
        md("## Step 2 — Auto-Categorize & Extract Metadata"),
        code("""
class NoteMetadata(BaseModel):
    category: str = Field(description="work, personal, idea, bug, meeting, reference")
    priority: str = Field(description="high, medium, low")
    has_action_item: bool
    action_item: str = ""
    related_people: list[str]
    tags: list[str]

classifier = llm.with_structured_output(NoteMetadata)

enriched = []
for note in voice_notes:
    meta = classifier.invoke(f"Categorize this voice note:\\n{note['text']}")
    enriched.append({**note, "metadata": meta})
    print(f"  {note['id']} [{meta.category}] P:{meta.priority} "
          f"{'📌' if meta.has_action_item else '  '} tags={meta.tags[:3]}")
"""),
        md("## Step 3 — Store in Vector DB for Search"),
        code("""
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("voice_notes")

for note in enriched:
    meta = note["metadata"]
    collection.add(
        ids=[note["id"]],
        documents=[note["text"]],
        metadatas=[{
            "timestamp": note["timestamp"],
            "category": meta.category,
            "priority": meta.priority,
            "has_action": str(meta.has_action_item),
        }],
    )

print(f"✓ Indexed {collection.count()} notes in ChromaDB")
"""),
        md("## Step 4 — Semantic Search & Linking"),
        code("""
queries = [
    "What bugs do I need to fix?",
    "What meetings happened this week?",
    "Any ideas for the app?",
    "What tasks need to be done today?",
]

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on the voice notes. Be specific and reference the notes."),
    ("human", "Voice notes:\\n{notes}\\n\\nQuestion: {question}\\nAnswer:")
])
qa_chain = qa_prompt | llm | StrOutputParser()

for query in queries:
    results = collection.query(query_texts=[query], n_results=3)
    context = "\\n".join(f"- {doc}" for doc in results["documents"][0])
    answer = qa_chain.invoke({"notes": context, "question": query})
    print(f"\\nQ: {query}")
    print(f"A: {answer[:150]}")
"""),
        md("## Step 5 — Daily Digest"),
        code("""
digest_prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a daily digest from these voice notes. Group by category. "
     "Highlight action items. Be concise."),
    ("human", "Notes:\\n{notes}\\n\\nDaily Digest:")
])
digest_chain = digest_prompt | llm | StrOutputParser()

all_notes = "\\n".join(f"[{n['timestamp']}] {n['text']}" for n in voice_notes)
digest = digest_chain.invoke({"notes": all_notes})
print("DAILY DIGEST")
print("=" * 50)
print(digest[:500])
"""),
        md("## What You Learned\n- **Voice note auto-categorization** with structured output\n- **Vector search** for semantic note retrieval\n- **Cross-note linking** via embeddings\n- **Daily digest generation** from unstructured notes"),
    ]))

    # ── 90 — Multimodal Research Combiner ──────────────────────────────
    paths.append(write_nb(9, "90_Local_Multimodal_Research", [
        md("# Project 90 — Local Multimodal Research Combiner\n## Multiple Sources → Unified Knowledge → Research Report\n\n**Stack:** LangChain · Ollama · ChromaDB · Pydantic · Jupyter"),
        code("# !pip install -q langchain langchain-ollama langchain-community chromadb pydantic pandas"),
        md("## Step 1 — Multi-Source Research Corpus"),
        code("""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import json, pandas as pd

llm = ChatOllama(model="qwen3:8b", temperature=0.2)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

research_sources = {
    "paper_abstract": [
        {"title": "Attention Is All You Need",
         "text": "The dominant sequence transduction models are based on complex recurrent or "
                 "convolutional neural networks. We propose the Transformer, based solely on "
                 "attention mechanisms. Experiments show superior quality while being more "
                 "parallelizable and requiring less training time."},
        {"title": "BERT: Pre-training of Deep Bidirectional Transformers",
         "text": "We introduce BERT, designed to pre-train deep bidirectional representations "
                 "by jointly conditioning on both left and right context. BERT can be fine-tuned "
                 "with just one additional output layer for a wide range of tasks."},
    ],
    "tech_blog": [
        {"title": "Building Production RAG Systems",
         "text": "RAG combines retrieval with generation. Key components: document chunking, "
                 "embedding generation, vector storage, query processing, and answer generation. "
                 "Common pitfalls: poor chunking strategy, embedding model mismatch, "
                 "and insufficient context window utilization."},
    ],
    "meeting_notes": [
        {"title": "AI Team Sync — March 2025",
         "text": "Discussed RAG implementation timeline. Current approach uses ChromaDB with "
                 "LangChain. Performance issues with large document sets (>10K docs). "
                 "Considering hybrid retrieval with BM25. Action: benchmark by end of week."},
    ],
    "code_docs": [
        {"title": "LangChain RAG Tutorial",
         "text": "from langchain.chains import RetrievalQA; from langchain.vectorstores import "
                 "Chroma; Setup: load documents, split into chunks, embed with OllamaEmbeddings, "
                 "store in Chroma, query with RetrievalQA chain."},
    ],
}

total = sum(len(v) for v in research_sources.values())
print(f"Sources: {len(research_sources)} types, {total} documents")
"""),
        md("## Step 2 — Index All Sources in ChromaDB"),
        code("""
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("research")

doc_id = 0
for source_type, docs in research_sources.items():
    for doc in docs:
        collection.add(
            ids=[f"doc_{doc_id}"],
            documents=[doc["text"]],
            metadatas=[{"source_type": source_type, "title": doc["title"]}],
        )
        doc_id += 1

print(f"✓ Indexed {collection.count()} documents across {len(research_sources)} source types")
"""),
        md("## Step 3 — Cross-Source Research Queries"),
        code("""
research_questions = [
    "What are the key architectures for NLP models?",
    "How should I build a production RAG system?",
    "What performance issues exist with vector databases?",
    "What code is needed to set up a RAG pipeline?",
]

class ResearchFinding(BaseModel):
    question: str
    sources_used: list[str]
    synthesis: str
    confidence: float = Field(ge=0, le=1)
    gaps: list[str] = Field(description="Information gaps that need more research")

researcher = llm.with_structured_output(ResearchFinding)

findings = []
for question in research_questions:
    results = collection.query(query_texts=[question], n_results=3)
    context = "\\n\\n".join(
        f"[{m['source_type']}] {m['title']}: {doc}"
        for doc, m in zip(results["documents"][0], results["metadatas"][0])
    )
    finding = researcher.invoke(
        f"Research question: {question}\\n\\nSources:\\n{context}"
    )
    findings.append(finding)
    print(f"\\nQ: {question}")
    print(f"  Sources: {finding.sources_used}")
    print(f"  Confidence: {finding.confidence:.0%}")
    print(f"  Gaps: {finding.gaps[:2]}")
"""),
        md("## Step 4 — Generate Research Report"),
        code("""
report_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a structured research report based on these findings. "
     "Include: Executive Summary, Key Findings, Methodology, Gaps & Next Steps. "
     "Use Markdown formatting."),
    ("human", "Research findings:\\n{findings}\\n\\nReport:")
])
report_chain = report_prompt | llm | StrOutputParser()

report = report_chain.invoke({
    "findings": json.dumps([f.model_dump() for f in findings], indent=2)
})
print("RESEARCH REPORT")
print("=" * 60)
print(report[:800])
"""),
        md("## Step 5 — Source Coverage Matrix"),
        code("""
# Check which source types contributed to each finding
coverage = {}
for f in findings:
    for s in f.sources_used:
        coverage.setdefault(s, 0)
        coverage[s] += 1

print("SOURCE UTILIZATION")
print("=" * 40)
for source, count in sorted(coverage.items(), key=lambda x: -x[1]):
    bar = "█" * count
    print(f"  {source:<20} {bar} ({count})")

print(f"\\nResearch gaps to fill:")
all_gaps = [g for f in findings for g in f.gaps]
for i, gap in enumerate(all_gaps[:5], 1):
    print(f"  {i}. {gap}")
"""),
        md("## What You Learned\n- **Multi-source research aggregation** in ChromaDB\n- **Cross-reference synthesis** from diverse document types\n- **Research report generation** with structured findings\n- **Coverage analysis** and gap identification"),
    ]))

    print(f"\nEnriched {len(paths)} notebooks (82-90)")
    for p in paths:
        print(f"  ✓ {p}")

if __name__ == "__main__":
    build()
