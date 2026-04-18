"""Build Advance RAG 28 - Freshness-Aware News RAG notebook."""
import os, nbformat

NB_DIR = r"e:\Github\Machine-Learning-Projects\Advance RAG\28_Freshness_Aware_News_RAG"
os.makedirs(NB_DIR, exist_ok=True)
NB_PATH = os.path.join(NB_DIR, "freshness_aware_news_rag.ipynb")

nb = nbformat.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"},
})

def md(src): return nbformat.v4.new_markdown_cell(src)
def code(src): return nbformat.v4.new_code_cell(src)

cells = []

# ── 1. Title ──────────────────────────────────────────────
cells.append(md(
"# Freshness-Aware News RAG\n"
"\n"
"## Prefer Recent Documents and Label Stale Information Risk\n"
"\n"
"**Project 28** - Advance RAG Learning Series\n"
"\n"
"| Property | Value |\n"
"|----------|-------|\n"
"| Task | Time-aware document retrieval with staleness detection |\n"
"| Method | Recency-weighted scoring + metadata date filters + staleness labels |\n"
"| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |\n"
"| Corpus | 30 timestamped news articles spanning 2020-2025 |\n"
"| Evaluation | Recency ranking quality, staleness detection accuracy |\n"
))

# ── 2. Overview ───────────────────────────────────────────
cells.append(md(
"## Project Overview\n"
"\n"
"### The Problem: Outdated Information in RAG\n"
"\n"
"Standard RAG retrieves by **semantic similarity** alone. But when\n"
"information changes over time, the most similar document may be\n"
"**outdated**:\n"
"\n"
"- *\"Who is the CEO of Twitter?\"* -- answer changed in 2022\n"
"- *\"What is the latest GPT model?\"* -- answer changes every year\n"
"- *\"How many COVID cases worldwide?\"* -- answer changed daily\n"
"\n"
"### Solution: Freshness-Aware Retrieval\n"
"\n"
"```\n"
"User question\n"
"      |\n"
"      v\n"
"Dense retrieval (semantic similarity scores)\n"
"      |\n"
"      v\n"
"Apply recency boost (fresher docs score higher)\n"
"      |\n"
"      v\n"
"Re-rank by combined score\n"
"      |\n"
"      v\n"
"Detect & label stale documents\n"
"      |\n"
"      v\n"
"Return results with freshness warnings\n"
"```\n"
))

# ── 3. Learning Goals ────────────────────────────────────
cells.append(md(
"## Learning Goals\n"
"\n"
"1. Understand why **recency** matters in RAG systems\n"
"2. Implement **recency scoring** that boosts recent documents\n"
"3. Build **metadata filters** to exclude documents older than a threshold\n"
"4. Detect and **label stale information** risks\n"
"5. Compare standard vs freshness-aware retrieval quality\n"
"6. Tune the **freshness weight** parameter\n"
))

# ── 4. Problem Statement ─────────────────────────────────
cells.append(md(
"## Problem Statement\n"
"\n"
"Given a timestamped news corpus where multiple articles cover\n"
"the **same topic at different points in time**:\n"
"\n"
"1. Retrieve documents that are both **relevant** and **recent**\n"
"2. When older documents are retrieved, **label them** with\n"
"   staleness warnings\n"
"3. Detect **contradictions** when old and new articles disagree\n"
"4. Allow users to control the **freshness weight** parameter\n"
))

# ── 5. Why It Matters ────────────────────────────────────
cells.append(md(
"## Why Freshness-Aware RAG Matters\n"
"\n"
"| Scenario | Standard RAG | Freshness-Aware RAG |\n"
"|----------|-------------|--------------------|\n"
"| \"Latest GPT model\" | May return GPT-3 article (higher sim) | Boosts GPT-4o article |\n"
"| \"SpaceX launches\" | Returns best-matching old article | Prefers most recent mission |\n"
"| Stock market query | Could cite 2020 prices | Prioritizes 2025 data |\n"
"| API documentation | Old deprecated API docs | Current version docs |\n"
"\n"
"### Two Approaches to Time-Awareness\n"
"\n"
"| Approach | How it works | Best for |\n"
"|----------|-------------|----------|\n"
"| **Recency ranking** | Blend similarity + date score | Soft preference for newer |\n"
"| **Metadata filter** | Hard cutoff: exclude old docs | Strict recency requirement |\n"
"\n"
"This notebook implements both and compares them.\n"
))

# ── 6. Environment ────────────────────────────────────────
cells.append(md("## Environment Setup"))

cells.append(code(
'import subprocess, sys, warnings\n'
'\n'
'def _install(pkg):\n'
'    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])\n'
'\n'
'for pkg in ["sentence-transformers"]:\n'
'    try:\n'
'        __import__(pkg.replace("-", "_"))\n'
'    except ImportError:\n'
'        _install(pkg)\n'
'\n'
'warnings.filterwarnings("ignore")\n'
'print("Environment ready.")\n'
))

# ── 7. Imports ────────────────────────────────────────────
cells.append(md("## Imports"))

cells.append(code(
'import random\n'
'from datetime import datetime, timedelta\n'
'from typing import List, Dict, Optional, Tuple\n'
'from dataclasses import dataclass, field\n'
'\n'
'import numpy as np\n'
'from sentence_transformers import SentenceTransformer\n'
'\n'
'SEED = 42\n'
'random.seed(SEED)\n'
'np.random.seed(SEED)\n'
'\n'
'print("All imports loaded.")\n'
))

# ── 8. Configuration ─────────────────────────────────────
cells.append(md("## Configuration"))

cells.append(code(
'EMBEDDING_MODEL = "all-MiniLM-L6-v2"\n'
'K = 3                       # documents to retrieve\n'
'FRESHNESS_WEIGHT = 0.3      # blend: (1-w)*similarity + w*recency\n'
'STALE_DAYS = 365            # docs older than this get staleness warning\n'
'REFERENCE_DATE = datetime(2025, 4, 1)  # \"today\" for scoring\n'
'MAX_AGE_DAYS = 1825         # 5 years -- oldest doc gets recency=0\n'
'\n'
'print(f"Config: model={EMBEDDING_MODEL}, K={K}")\n'
'print(f"  freshness_weight={FRESHNESS_WEIGHT}")\n'
'print(f"  stale_threshold={STALE_DAYS} days")\n'
'print(f"  reference_date={REFERENCE_DATE.date()}")\n'
))

# ── 9. News Corpus ────────────────────────────────────────
cells.append(md(
"## Timestamped News Corpus\n"
"\n"
"30 articles spanning 2020-2025. Multiple articles cover the\n"
"same topic at different times, with **evolving or contradictory**\n"
"information -- exactly the scenario where freshness matters.\n"
))

cells.append(code(
'corpus = [\n'
'    # -- AI Models (same topic, different dates) --\n'
'    {"id": "AI01", "date": "2020-06-15", "topic": "ai-models",\n'
'     "text": "OpenAI released GPT-3, the largest language model to date with 175 billion parameters. GPT-3 demonstrated remarkable few-shot learning abilities across diverse tasks. The model set new benchmarks in text generation quality."},\n'
'    {"id": "AI02", "date": "2022-11-30", "topic": "ai-models",\n'
'     "text": "OpenAI launched ChatGPT, a conversational AI based on GPT-3.5. ChatGPT gained 100 million users in two months, making it the fastest-growing consumer application in history. The model excels at following instructions and maintaining dialogue context."},\n'
'    {"id": "AI03", "date": "2023-03-14", "topic": "ai-models",\n'
'     "text": "OpenAI released GPT-4, a multimodal model accepting text and image inputs. GPT-4 passed the bar exam in the 90th percentile and showed significant improvements in reasoning, reliability, and factual accuracy over GPT-3.5."},\n'
'    {"id": "AI04", "date": "2024-05-13", "topic": "ai-models",\n'
'     "text": "OpenAI unveiled GPT-4o, a natively multimodal model processing text, audio, and vision in a single architecture. GPT-4o reduced latency by 50% compared to GPT-4 Turbo and matched GPT-4 performance at lower cost."},\n'
'    {"id": "AI05", "date": "2025-02-20", "topic": "ai-models",\n'
'     "text": "OpenAI released GPT-5, featuring improved reasoning and coding capabilities. GPT-5 introduced persistent memory across conversations and achieved near-human performance on graduate-level science and math benchmarks."},\n'
'\n'
'    # -- Electric Vehicles --\n'
'    {"id": "EV01", "date": "2021-01-10", "topic": "ev-market",\n'
'     "text": "Tesla delivered 936,172 vehicles in 2021, maintaining its position as the world largest electric vehicle manufacturer. The Model 3 and Model Y accounted for 97% of deliveries. Tesla stock reached a market cap of $1 trillion."},\n'
'    {"id": "EV02", "date": "2023-07-05", "topic": "ev-market",\n'
'     "text": "BYD surpassed Tesla in quarterly EV sales for the first time in Q4 2023. BYD sold 526,000 battery electric vehicles versus Tesla 484,000. BYD success was driven by affordable models and strong demand in China."},\n'
'    {"id": "EV03", "date": "2025-01-15", "topic": "ev-market",\n'
'     "text": "The global EV market reached 18 million units in 2024. BYD led with 3.2 million units, followed by Tesla at 2.8 million. Chinese EV makers now hold 60% of the global market. New entrants include Xiaomi with 120,000 units in its first year."},\n'
'\n'
'    # -- Social Media --\n'
'    {"id": "SM01", "date": "2020-08-20", "topic": "social-media",\n'
'     "text": "Twitter reported 187 million daily active users in Q2 2020. Jack Dorsey serves as CEO. The platform generated $683 million in Q2 revenue, down 19% due to the COVID-19 advertising slowdown."},\n'
'    {"id": "SM02", "date": "2022-10-28", "topic": "social-media",\n'
'     "text": "Elon Musk completed the acquisition of Twitter for $44 billion. Musk became the new CEO and immediately began sweeping changes including laying off 50% of staff. The company was taken private and delisted from NYSE."},\n'
'    {"id": "SM03", "date": "2023-07-24", "topic": "social-media",\n'
'     "text": "Twitter officially rebranded to X under Elon Musk ownership. The iconic blue bird logo was replaced with an X symbol. Musk vision is to transform X into an everything app handling payments, messaging, and social media."},\n'
'    {"id": "SM04", "date": "2025-03-10", "topic": "social-media",\n'
'     "text": "X (formerly Twitter) reported 550 million monthly active users. CEO Linda Yaccarino announced profitability for the first time since the acquisition. The platform launched X Payments in 12 countries and integrated Grok AI assistant."},\n'
'\n'
'    # -- Space Exploration --\n'
'    {"id": "SP01", "date": "2020-05-30", "topic": "space",\n'
'     "text": "SpaceX Crew Dragon launched NASA astronauts to the ISS, marking the first crewed orbital launch from US soil since the Space Shuttle retired in 2011. The mission validated SpaceX human spaceflight capabilities."},\n'
'    {"id": "SP02", "date": "2022-04-08", "topic": "space",\n'
'     "text": "SpaceX completed its first fully private mission, Axiom-1, sending four private astronauts to the ISS. The commercial space industry reached $469 billion in revenue. SpaceX launched a record 61 missions in 2022."},\n'
'    {"id": "SP03", "date": "2024-06-06", "topic": "space",\n'
'     "text": "SpaceX Starship completed its fourth test flight, achieving successful re-entry for the first time. The vehicle splashed down in the Indian Ocean as planned. Starship is the largest and most powerful rocket ever built."},\n'
'    {"id": "SP04", "date": "2025-03-25", "topic": "space",\n'
'     "text": "SpaceX Starship successfully caught the booster on its landing arms during the seventh flight test. NASA confirmed Starship as the lunar lander for Artemis III scheduled for late 2026. SpaceX completed over 100 orbital missions in Q1 2025."},\n'
'\n'
'    # -- Cybersecurity --\n'
'    {"id": "CS01", "date": "2020-12-13", "topic": "cybersecurity",\n'
'     "text": "The SolarWinds supply chain attack was discovered, affecting 18,000 organizations including US government agencies. Russian state-sponsored hackers inserted malware into SolarWinds Orion software updates."},\n'
'    {"id": "CS02", "date": "2023-01-18", "topic": "cybersecurity",\n'
'     "text": "Ransomware attacks increased 95% in 2022, with average ransom payments reaching $812,000. The healthcare and education sectors were most targeted. AI-generated phishing emails became a growing threat vector."},\n'
'    {"id": "CS03", "date": "2025-02-05", "topic": "cybersecurity",\n'
'     "text": "AI-powered cyberattacks surged 300% in 2024. Deepfake voice cloning was used in $25 million wire fraud cases. Zero-trust architecture adoption reached 67% among Fortune 500 companies. Quantum-resistant encryption standards were finalized by NIST."},\n'
'\n'
'    # -- Climate / Energy --\n'
'    {"id": "CL01", "date": "2021-04-22", "topic": "climate",\n'
'     "text": "President Biden pledged to cut US emissions 50% by 2030 at the Leaders Summit on Climate. The US rejoined the Paris Agreement. Renewable energy investment reached $300 billion globally in 2020."},\n'
'    {"id": "CL02", "date": "2023-12-13", "topic": "climate",\n'
'     "text": "COP28 in Dubai concluded with a historic agreement to transition away from fossil fuels. Global renewable energy capacity hit 3,700 GW. Solar became the cheapest source of electricity in most countries."},\n'
'    {"id": "CL03", "date": "2025-01-30", "topic": "climate",\n'
'     "text": "Global renewable energy capacity crossed 4,500 GW in 2024. China installed more solar capacity than the rest of the world combined. Battery storage costs fell below $100/kWh for the first time, enabling grid-scale deployment."},\n'
'\n'
'    # -- Cryptocurrency --\n'
'    {"id": "CR01", "date": "2021-11-10", "topic": "crypto",\n'
'     "text": "Bitcoin reached an all-time high of $68,789. The total cryptocurrency market cap exceeded $3 trillion. El Salvador became the first country to adopt Bitcoin as legal tender. Institutional adoption accelerated."},\n'
'    {"id": "CR02", "date": "2022-11-11", "topic": "crypto",\n'
'     "text": "FTX, the third-largest crypto exchange, filed for bankruptcy amid allegations of fraud. Customer deposits worth $8 billion were missing. Bitcoin fell to $15,500. The collapse triggered calls for stricter regulation."},\n'
'    {"id": "CR03", "date": "2024-01-11", "topic": "crypto",\n'
'     "text": "The SEC approved 11 spot Bitcoin ETFs, marking a watershed moment for cryptocurrency. Bitcoin surged past $70,000 following the approval. BlackRock and Fidelity led ETF inflows exceeding $10 billion in the first month."},\n'
'    {"id": "CR04", "date": "2025-03-01", "topic": "crypto",\n'
'     "text": "Bitcoin reached $95,000, driven by institutional ETF demand and the 2024 halving effect. Ethereum completed its next major upgrade. Stablecoin transaction volume exceeded Visa globally for the first time."},\n'
'\n'
'    # -- Pandemic / Health --\n'
'    {"id": "PH01", "date": "2020-03-11", "topic": "health",\n'
'     "text": "The WHO declared COVID-19 a global pandemic. Cases surpassed 118,000 across 114 countries. Italy became the epicenter in Europe with nationwide lockdown. Stock markets crashed with the Dow losing 20% in weeks."},\n'
'    {"id": "PH02", "date": "2021-12-20", "topic": "health",\n'
'     "text": "Over 4 billion COVID-19 vaccine doses administered worldwide. Pfizer and Moderna mRNA vaccines showed 90%+ efficacy. The Omicron variant emerged with high transmissibility but lower severity. Booster shots were recommended."},\n'
'    {"id": "PH03", "date": "2023-05-05", "topic": "health",\n'
'     "text": "The WHO declared the end of COVID-19 as a public health emergency. Over 7 million deaths were recorded officially. The pandemic accelerated telemedicine adoption and mRNA vaccine technology development for other diseases."},\n'
'    {"id": "PH04", "date": "2025-02-10", "topic": "health",\n'
'     "text": "mRNA technology expanded beyond COVID. Moderna Phase 3 trials for cancer vaccines showed 44% reduction in recurrence. Bird flu H5N1 surveillance intensified after human cases in dairy workers. Global health systems focused on pandemic preparedness."},\n'
']\n'
'\n'
'# Parse dates\n'
'for doc in corpus:\n'
'    doc["datetime"] = datetime.strptime(doc["date"], "%Y-%m-%d")\n'
'\n'
'print(f"News corpus: {len(corpus)} articles")\n'
'topics = {}\n'
'for doc in corpus:\n'
'    topics[doc["topic"]] = topics.get(doc["topic"], 0) + 1\n'
'for topic, count in sorted(topics.items()):\n'
'    print(f"  {topic}: {count} articles")\n'
'dates = [doc["datetime"] for doc in corpus]\n'
'print(f"Date range: {min(dates).date()} to {max(dates).date()}")\n'
))

# ── 10. Dense Retriever ──────────────────────────────────
cells.append(md("## Dense Retriever (Semantic Similarity Only)"))

cells.append(code(
'print(f"Loading embedding model: {EMBEDDING_MODEL}...")\n'
'encoder = SentenceTransformer(EMBEDDING_MODEL)\n'
'\n'
'doc_texts = [doc["text"] for doc in corpus]\n'
'doc_embeddings = encoder.encode(doc_texts, convert_to_numpy=True, show_progress_bar=False)\n'
'doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)\n'
'\n'
'\n'
'def similarity_scores(query: str) -> np.ndarray:\n'
'    """Compute cosine similarity between query and all documents."""\n'
'    q_emb = encoder.encode([query], convert_to_numpy=True)\n'
'    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)\n'
'    return (doc_embeddings @ q_emb.T).flatten()\n'
'\n'
'\n'
'test = similarity_scores("latest GPT model")\n'
'print(f"Retriever ready (index: {doc_embeddings.shape})")\n'
'top3 = np.argsort(test)[::-1][:3]\n'
'for idx in top3:\n'
'    print(f"  [{corpus[idx][\"id\"]}] {corpus[idx][\"date\"]} score={test[idx]:.3f} | {corpus[idx][\"text\"][:55]}...")\n'
))

# ── 11. Recency Scoring ──────────────────────────────────
cells.append(md(
"## Recency Scoring\n"
"\n"
"### How Recency Ranking Works\n"
"\n"
"Recency scoring converts a document's date into a 0-1 score:\n"
"\n"
"$$\\text{recency}(d) = \\max\\left(0, \\; 1 - \\frac{\\text{age\\_days}(d)}{\\text{max\\_age\\_days}}\\right)$$\n"
"\n"
"- A document published **today** gets recency = 1.0\n"
"- A document published **max_age_days ago** gets recency = 0.0\n"
"- Documents **older** than max_age_days are clamped to 0.0\n"
"\n"
"### Combined Score\n"
"\n"
"$$\\text{score}(d) = (1 - w) \\times \\text{similarity}(d) + w \\times \\text{recency}(d)$$\n"
"\n"
"Where $w$ is the **freshness weight**:\n"
"- $w = 0$: pure semantic similarity (standard RAG)\n"
"- $w = 0.3$: mild recency preference (default)\n"
"- $w = 0.7$: strong recency preference\n"
"- $w = 1.0$: pure recency (ignores content relevance)\n"
))

cells.append(code(
'def compute_recency(doc_date: datetime,\n'
'                    reference: datetime = REFERENCE_DATE,\n'
'                    max_age: int = MAX_AGE_DAYS) -> float:\n'
'    """Compute recency score 0-1 for a document date."""\n'
'    age_days = (reference - doc_date).days\n'
'    if age_days < 0:\n'
'        return 1.0  # future date treated as maximally recent\n'
'    return max(0.0, 1.0 - age_days / max_age)\n'
'\n'
'\n'
'# Demo: recency scores across the AI models topic\n'
'print("Recency scores for AI model articles:")\n'
'print(f"  Reference date: {REFERENCE_DATE.date()}, max_age: {MAX_AGE_DAYS} days\\n")\n'
'for doc in corpus:\n'
'    if doc["topic"] == "ai-models":\n'
'        r = compute_recency(doc["datetime"])\n'
'        age = (REFERENCE_DATE - doc["datetime"]).days\n'
'        print(f"  [{doc[\"id\"]}] {doc[\"date\"]}  age={age:>5d}d  recency={r:.3f} | {doc[\"text\"][:50]}...")\n'
))

# ── 12. Freshness-Aware Retriever ─────────────────────────
cells.append(md(
"## Freshness-Aware Retriever\n"
"\n"
"Combines semantic similarity with recency scoring.\n"
))

cells.append(code(
'@dataclass\n'
'class RetrievalResult:\n'
'    """A single retrieved document with all scores."""\n'
'    doc_id: str\n'
'    date: str\n'
'    topic: str\n'
'    text: str\n'
'    similarity: float\n'
'    recency: float\n'
'    combined: float\n'
'    is_stale: bool\n'
'    age_days: int\n'
'\n'
'\n'
'def retrieve_freshness_aware(\n'
'    query: str,\n'
'    k: int = K,\n'
'    freshness_weight: float = FRESHNESS_WEIGHT,\n'
'    max_age_filter: Optional[int] = None,\n'
') -> List[RetrievalResult]:\n'
'    """Retrieve documents with recency-aware scoring.\n'
'    \n'
'    Args:\n'
'        query: search query\n'
'        k: number of results\n'
'        freshness_weight: 0=pure similarity, 1=pure recency\n'
'        max_age_filter: if set, hard-exclude docs older than this many days\n'
'    """\n'
'    sims = similarity_scores(query)\n'
'    results = []\n'
'    \n'
'    for i, doc in enumerate(corpus):\n'
'        age_days = (REFERENCE_DATE - doc["datetime"]).days\n'
'        \n'
'        # Hard metadata filter\n'
'        if max_age_filter is not None and age_days > max_age_filter:\n'
'            continue\n'
'        \n'
'        rec = compute_recency(doc["datetime"])\n'
'        sim = float(sims[i])\n'
'        combined = (1 - freshness_weight) * sim + freshness_weight * rec\n'
'        is_stale = age_days > STALE_DAYS\n'
'        \n'
'        results.append(RetrievalResult(\n'
'            doc_id=doc["id"], date=doc["date"], topic=doc["topic"],\n'
'            text=doc["text"], similarity=sim, recency=rec,\n'
'            combined=combined, is_stale=is_stale, age_days=age_days,\n'
'        ))\n'
'    \n'
'    results.sort(key=lambda r: r.combined, reverse=True)\n'
'    return results[:k]\n'
'\n'
'\n'
'# Standard retrieval (no freshness)\n'
'def retrieve_standard(query: str, k: int = K) -> List[RetrievalResult]:\n'
'    """Standard retrieval -- similarity only (freshness_weight=0)."""\n'
'    return retrieve_freshness_aware(query, k=k, freshness_weight=0.0)\n'
'\n'
'\n'
'print("Freshness-aware retriever defined.")\n'
))

# ── 13. Staleness Detection ──────────────────────────────
cells.append(md(
"## Staleness Detection\n"
"\n"
"Label each retrieved document with staleness risk.\n"
"\n"
"| Age | Label | Meaning |\n"
"|-----|-------|--------|\n"
"| < 180 days | FRESH | Current information |\n"
"| 180-365 days | AGING | May be slightly outdated |\n"
"| > 365 days | STALE | High risk of outdated information |\n"
"| > 730 days | VERY STALE | Likely superseded by newer information |\n"
))

cells.append(code(
'def staleness_label(age_days: int) -> str:\n'
'    """Classify document staleness."""\n'
'    if age_days <= 180:\n'
'        return "FRESH"\n'
'    elif age_days <= 365:\n'
'        return "AGING"\n'
'    elif age_days <= 730:\n'
'        return "STALE"\n'
'    else:\n'
'        return "VERY STALE"\n'
'\n'
'\n'
'def format_result(r: RetrievalResult) -> str:\n'
'    """Format a retrieval result with staleness warning."""\n'
'    label = staleness_label(r.age_days)\n'
'    warning = ""\n'
'    if label in ("STALE", "VERY STALE"):\n'
'        warning = f" ** WARNING: {label} -- information may be outdated **"\n'
'    return (f"[{r.doc_id}] {r.date} ({label}, {r.age_days}d) "\n'
'            f"sim={r.similarity:.3f} rec={r.recency:.3f} "\n'
'            f"combined={r.combined:.3f}{warning}")\n'
'\n'
'\n'
'# Demo\n'
'print("Staleness labels demo:")\n'
'for days in [30, 200, 400, 800, 1500]:\n'
'    print(f"  {days} days old -> {staleness_label(days)}")\n'
))

# ── 14. Test Questions ───────────────────────────────────
cells.append(md(
"## Test Questions\n"
"\n"
"Questions where recency matters -- the correct answer depends\n"
"on how recent the retrieved document is.\n"
))

cells.append(code(
'test_questions = [\n'
'    {\n'
'        "question": "What is the latest GPT model from OpenAI?",\n'
'        "best_doc": "AI05",  # GPT-5, Feb 2025\n'
'        "stale_docs": ["AI01", "AI02"],  # GPT-3, ChatGPT\n'
'        "topic": "ai-models",\n'
'        "type": "latest-version",\n'
'    },\n'
'    {\n'
'        "question": "Who leads the global EV market in sales?",\n'
'        "best_doc": "EV03",  # 2025: BYD leads\n'
'        "stale_docs": ["EV01"],  # 2021: Tesla leads\n'
'        "topic": "ev-market",\n'
'        "type": "evolving-ranking",\n'
'    },\n'
'    {\n'
'        "question": "Who is the CEO of Twitter?",\n'
'        "best_doc": "SM04",  # 2025: Linda Yaccarino / X\n'
'        "stale_docs": ["SM01"],  # 2020: Jack Dorsey\n'
'        "topic": "social-media",\n'
'        "type": "changed-fact",\n'
'    },\n'
'    {\n'
'        "question": "What is the status of SpaceX Starship?",\n'
'        "best_doc": "SP04",  # 2025: successful booster catch\n'
'        "stale_docs": ["SP01"],  # 2020: Crew Dragon (not Starship)\n'
'        "topic": "space",\n'
'        "type": "latest-version",\n'
'    },\n'
'    {\n'
'        "question": "What are the biggest cybersecurity threats?",\n'
'        "best_doc": "CS03",  # 2025: AI attacks, deepfakes\n'
'        "stale_docs": ["CS01"],  # 2020: SolarWinds\n'
'        "topic": "cybersecurity",\n'
'        "type": "evolving-ranking",\n'
'    },\n'
'    {\n'
'        "question": "What is the current Bitcoin price and market status?",\n'
'        "best_doc": "CR04",  # 2025: $95K\n'
'        "stale_docs": ["CR01", "CR02"],  # 2021: $68K ATH, 2022: crash\n'
'        "topic": "crypto",\n'
'        "type": "changed-fact",\n'
'    },\n'
'    {\n'
'        "question": "Is COVID-19 still a pandemic?",\n'
'        "best_doc": "PH03",  # 2023: WHO declared end\n'
'        "stale_docs": ["PH01"],  # 2020: pandemic declared\n'
'        "topic": "health",\n'
'        "type": "changed-fact",\n'
'    },\n'
'    {\n'
'        "question": "How much renewable energy capacity exists globally?",\n'
'        "best_doc": "CL03",  # 2025: 4,500 GW\n'
'        "stale_docs": ["CL01"],  # 2021: $300B investment\n'
'        "topic": "climate",\n'
'        "type": "evolving-ranking",\n'
'    },\n'
']\n'
'\n'
'print(f"Test questions: {len(test_questions)}")\n'
'for i, q in enumerate(test_questions, 1):\n'
'    print(f"  Q{i} [{q[\"type\"]:>16s}] best={q[\"best_doc\"]} | {q[\"question\"][:55]}")\n'
))

# ── 15. Benchmark: Standard vs Freshness-Aware ───────────
cells.append(md("## Benchmark: Standard vs Freshness-Aware Retrieval"))

cells.append(code(
'results_standard = []\n'
'results_fresh = []\n'
'\n'
'for tq in test_questions:\n'
'    q = tq["question"]\n'
'    \n'
'    std = retrieve_standard(q, k=K)\n'
'    frs = retrieve_freshness_aware(q, k=K, freshness_weight=FRESHNESS_WEIGHT)\n'
'    \n'
'    results_standard.append({"tq": tq, "results": std})\n'
'    results_fresh.append({"tq": tq, "results": frs})\n'
'\n'
'print(f"Benchmark complete: {len(test_questions)} questions x 2 methods")\n'
))

# ── 16. Per-Question Comparison ──────────────────────────
cells.append(md("## Per-Question Comparison"))

cells.append(code(
'for i, (sq, fq) in enumerate(zip(results_standard, results_fresh), 1):\n'
'    tq = sq["tq"]\n'
'    print(f"\\nQ{i}: {tq[\'question\']}")\n'
'    print(f"  Best doc: {tq[\'best_doc\']} | Stale: {tq[\'stale_docs\']}")\n'
'    \n'
'    # Standard results\n'
'    std_ids = [r.doc_id for r in sq["results"]]\n'
'    std_hit = tq["best_doc"] in std_ids\n'
'    std_stale = [r.doc_id for r in sq["results"] if r.doc_id in tq["stale_docs"]]\n'
'    std_rank = std_ids.index(tq["best_doc"]) + 1 if std_hit else None\n'
'    \n'
'    # Freshness results\n'
'    frs_ids = [r.doc_id for r in fq["results"]]\n'
'    frs_hit = tq["best_doc"] in frs_ids\n'
'    frs_stale = [r.doc_id for r in fq["results"] if r.doc_id in tq["stale_docs"]]\n'
'    frs_rank = frs_ids.index(tq["best_doc"]) + 1 if frs_hit else None\n'
'    \n'
'    print(f"  Standard:  top-{K}={std_ids} | best@{std_rank} | stale_retrieved={std_stale}")\n'
'    print(f"  Freshness: top-{K}={frs_ids} | best@{frs_rank} | stale_retrieved={frs_stale}")\n'
'    \n'
'    # Show full detail for freshness\n'
'    for r in fq["results"]:\n'
'        label = staleness_label(r.age_days)\n'
'        marker = " <-- BEST" if r.doc_id == tq["best_doc"] else ""\n'
'        print(f"    [{r.doc_id}] {r.date} {label:>10s} sim={r.similarity:.3f} rec={r.recency:.3f} comb={r.combined:.3f}{marker}")\n'
))

# ── 17. Aggregate Metrics ────────────────────────────────
cells.append(md("## Aggregate Results"))

cells.append(code(
'def evaluate_method(results_list, label):\n'
'    """Compute aggregate metrics for a retrieval method."""\n'
'    hits_at_1 = 0\n'
'    hits_at_k = 0\n'
'    stale_retrieved = 0\n'
'    total_stale_possible = 0\n'
'    avg_best_recency = []\n'
'    \n'
'    for entry in results_list:\n'
'        tq = entry["tq"]\n'
'        ids = [r.doc_id for r in entry["results"]]\n'
'        \n'
'        # Best doc hit\n'
'        if ids and ids[0] == tq["best_doc"]:\n'
'            hits_at_1 += 1\n'
'        if tq["best_doc"] in ids:\n'
'            hits_at_k += 1\n'
'        \n'
'        # Stale doc retrieval\n'
'        for r in entry["results"]:\n'
'            if r.doc_id in tq["stale_docs"]:\n'
'                stale_retrieved += 1\n'
'        total_stale_possible += len(tq["stale_docs"]) * K  # max possible\n'
'        \n'
'        # Average recency of top result\n'
'        if entry["results"]:\n'
'            avg_best_recency.append(entry["results"][0].recency)\n'
'    \n'
'    n = len(results_list)\n'
'    return {\n'
'        "label": label,\n'
'        "hit_at_1": hits_at_1 / n,\n'
'        "hit_at_k": hits_at_k / n,\n'
'        "stale_retrieved": stale_retrieved,\n'
'        "avg_top1_recency": np.mean(avg_best_recency) if avg_best_recency else 0,\n'
'    }\n'
'\n'
'\n'
'std_metrics = evaluate_method(results_standard, "Standard")\n'
'frs_metrics = evaluate_method(results_fresh, "Freshness-Aware")\n'
'\n'
'print(f"{\"Metric\":<25} {\"Standard\":>12} {\"Freshness\":>12}")\n'
'print("-" * 51)\n'
'for key in ["hit_at_1", "hit_at_k", "stale_retrieved", "avg_top1_recency"]:\n'
'    sv = std_metrics[key]\n'
'    fv = frs_metrics[key]\n'
'    fmt = ".3f" if isinstance(sv, float) else "d"\n'
'    print(f"{key:<25} {sv:>12{fmt}} {fv:>12{fmt}}")\n'
'\n'
'print(f"\\nFreshness-aware retrieval:")\n'
'print(f"  Best-doc-at-1 improved: {std_metrics[\'hit_at_1\']:.0%} -> {frs_metrics[\'hit_at_1\']:.0%}")\n'
'print(f"  Stale docs retrieved: {std_metrics[\'stale_retrieved\']} -> {frs_metrics[\'stale_retrieved\']}")\n'
'print(f"  Top-1 avg recency: {std_metrics[\'avg_top1_recency\']:.3f} -> {frs_metrics[\'avg_top1_recency\']:.3f}")\n'
))

# ── 18. Freshness Weight Sensitivity ─────────────────────
cells.append(md(
"## Freshness Weight Sensitivity Analysis\n"
"\n"
"How does the freshness weight $w$ affect retrieval quality?\n"
"Too low = ignores recency. Too high = ignores relevance.\n"
))

cells.append(code(
'weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]\n'
'\n'
'print(f"{\"Weight\":>7} {\"Hit@1\":>7} {\"Hit@K\":>7} {\"Stale\":>7} {\"Top1 Recency\":>13}")\n'
'print("-" * 43)\n'
'\n'
'for w in weights:\n'
'    w_results = []\n'
'    for tq in test_questions:\n'
'        r = retrieve_freshness_aware(tq["question"], k=K, freshness_weight=w)\n'
'        w_results.append({"tq": tq, "results": r})\n'
'    m = evaluate_method(w_results, f"w={w}")\n'
'    print(f"{w:>7.1f} {m[\"hit_at_1\"]:>7.2f} {m[\"hit_at_k\"]:>7.2f} {m[\"stale_retrieved\"]:>7d} {m[\"avg_top1_recency\"]:>13.3f}")\n'
'\n'
'print("\\nInterpretation:")\n'
'print("  w=0.0: pure similarity -- may retrieve stale articles")\n'
'print("  w=0.2-0.4: balanced -- usually the sweet spot")\n'
'print("  w=0.7+: strong recency bias -- may miss relevant older context")\n'
'print("  w=1.0: pure recency -- retrieves newest regardless of content")\n'
))

# ── 19. Metadata Filter Demo ─────────────────────────────
cells.append(md(
"## Metadata Filters: Hard Date Cutoff\n"
"\n"
"Instead of soft recency scoring, sometimes you want a\n"
"**hard cutoff**: exclude all documents older than N days.\n"
"\n"
"This is a **metadata filter** -- applied before similarity\n"
"scoring, reducing the candidate pool.\n"
"\n"
"| Approach | Behavior | Risk |\n"
"|----------|----------|------|\n"
"| Recency scoring | Soft boost to recent docs | May still show old docs if highly relevant |\n"
"| Metadata filter | Hard exclude old docs | May return no results if all docs are old |\n"
"| Combined | Filter first, then recency-score remaining | Best of both worlds |\n"
))

cells.append(code(
'# Compare: no filter vs 2-year filter vs 1-year filter\n'
'filter_configs = [\n'
'    ("No filter", None),\n'
'    ("2-year filter (730d)", 730),\n'
'    ("1-year filter (365d)", 365),\n'
']\n'
'\n'
'demo_q = "What is the latest GPT model from OpenAI?"\n'
'print(f"Query: {demo_q}\\n")\n'
'\n'
'for label, max_age in filter_configs:\n'
'    results = retrieve_freshness_aware(\n'
'        demo_q, k=K, freshness_weight=FRESHNESS_WEIGHT, max_age_filter=max_age\n'
'    )\n'
'    print(f"  {label}:")\n'
'    if not results:\n'
'        print("    (no results -- all docs filtered out)")\n'
'    for r in results:\n'
'        sl = staleness_label(r.age_days)\n'
'        print(f"    [{r.doc_id}] {r.date} {sl:>10s} comb={r.combined:.3f} | {r.text[:50]}...")\n'
'    print()\n'
))

# ── 20. Staleness Report ─────────────────────────────────
cells.append(md("## Staleness Report on Freshness-Aware Results"))

cells.append(code(
'print("Staleness Distribution in Freshness-Aware Results")\n'
'print("=" * 50)\n'
'\n'
'label_counts = {"FRESH": 0, "AGING": 0, "STALE": 0, "VERY STALE": 0}\n'
'for entry in results_fresh:\n'
'    for r in entry["results"]:\n'
'        label_counts[staleness_label(r.age_days)] += 1\n'
'\n'
'total = sum(label_counts.values())\n'
'for label, count in label_counts.items():\n'
'    pct = count / total * 100 if total > 0 else 0\n'
'    bar = "#" * int(pct / 2)\n'
'    print(f"  {label:>10s}: {count:>3d} ({pct:>5.1f}%) {bar}")\n'
'\n'
'print(f"\\nTotal retrieved documents: {total}")\n'
'stale_pct = (label_counts["STALE"] + label_counts["VERY STALE"]) / total * 100 if total > 0 else 0\n'
'print(f"Stale + Very Stale: {stale_pct:.1f}%")\n'
))

# ── 21. Qualitative Examples ─────────────────────────────
cells.append(md("## Qualitative Comparison: Three Example Queries"))

cells.append(code(
'examples = [\n'
'    "What is the latest GPT model from OpenAI?",\n'
'    "Who is the CEO of Twitter?",\n'
'    "What is the current Bitcoin price and market status?",\n'
']\n'
'\n'
'for question in examples:\n'
'    print(f"\\n{\'=\' * 65}")\n'
'    print(f"Q: {question}")\n'
'    \n'
'    std = retrieve_standard(question, k=3)\n'
'    frs = retrieve_freshness_aware(question, k=3, freshness_weight=0.3)\n'
'    \n'
'    print(f"\\n  STANDARD (similarity only):")\n'
'    for r in std:\n'
'        sl = staleness_label(r.age_days)\n'
'        print(f"    [{r.doc_id}] {r.date} {sl:>10s} sim={r.similarity:.3f} | {r.text[:60]}...")\n'
'    \n'
'    print(f"\\n  FRESHNESS-AWARE (w={FRESHNESS_WEIGHT}):")\n'
'    for r in frs:\n'
'        sl = staleness_label(r.age_days)\n'
'        warn = " ** STALE **" if sl in ("STALE", "VERY STALE") else ""\n'
'        print(f"    [{r.doc_id}] {r.date} {sl:>10s} comb={r.combined:.3f} | {r.text[:60]}...{warn}")\n'
))

# ── 22. Error Analysis ───────────────────────────────────
cells.append(md("## Error Analysis"))

cells.append(code(
'print("Error Analysis")\n'
'print("=" * 50)\n'
'\n'
'# Cases where freshness-aware still fails\n'
'print("\\nFreshness-aware failures (best doc not at rank 1):")\n'
'for i, entry in enumerate(results_fresh, 1):\n'
'    tq = entry["tq"]\n'
'    ids = [r.doc_id for r in entry["results"]]\n'
'    if ids[0] != tq["best_doc"]:\n'
'        print(f"  Q{i}: {tq[\"question\"][:55]}")\n'
'        print(f"    Expected rank 1: {tq[\"best_doc\"]}")\n'
'        print(f"    Got rank 1: {ids[0]} (sim={entry[\"results\"][0].similarity:.3f}, rec={entry[\"results\"][0].recency:.3f})")\n'
'        # Find best doc rank\n'
'        if tq["best_doc"] in ids:\n'
'            rank = ids.index(tq["best_doc"]) + 1\n'
'            best_r = entry["results"][rank - 1]\n'
'            print(f"    Best doc at rank {rank}: sim={best_r.similarity:.3f}, rec={best_r.recency:.3f}")\n'
'        else:\n'
'            print(f"    Best doc not in top-{K}")\n'
'\n'
'# Cases where standard retrieval returned very stale docs at rank 1\n'
'print("\\nStandard retrieval: stale docs at rank 1:")\n'
'stale_at_1 = 0\n'
'for i, entry in enumerate(results_standard, 1):\n'
'    r = entry["results"][0]\n'
'    sl = staleness_label(r.age_days)\n'
'    if sl in ("STALE", "VERY STALE"):\n'
'        stale_at_1 += 1\n'
'        print(f"  Q{i}: [{r.doc_id}] {r.date} ({sl}) | {entry[\"tq\"][\"question\"][:55]}")\n'
'print(f"  Total: {stale_at_1}/{len(results_standard)} queries returned stale rank-1 doc")\n'
'\n'
'print("\\nRoot causes:")\n'
'print("  1. Semantic similarity may favor detailed older articles over brief newer ones")\n'
'print("  2. Freshness weight may not be high enough to overcome large similarity gaps")\n'
'print("  3. Some queries genuinely need historical context alongside current info")\n'
))

# ── 23. Limitations ──────────────────────────────────────
cells.append(md(
"## Limitations\n"
"\n"
"1. **Not all queries need recency.** Historical questions\n"
"   (\"When was Bitcoin created?\") should not penalize old documents.\n"
"\n"
"2. **Linear decay is simplistic.** Information doesn't become\n"
"   stale linearly -- some facts change suddenly (CEO change),\n"
"   others gradually (market share).\n"
"\n"
"3. **No contradiction detection.** When old and new articles\n"
"   disagree, we prefer the newer one but don't explicitly\n"
"   flag the contradiction.\n"
"\n"
"4. **Fixed reference date.** In production, this would be\n"
"   the current timestamp, not a hardcoded date.\n"
"\n"
"5. **No query-time classification.** An LLM could decide\n"
"   whether a query is time-sensitive before applying freshness.\n"
"\n"
"6. **Small corpus.** 30 articles make the effect visible but\n"
"   real news corpora have millions of articles.\n"
))

# ── 24. Common Mistakes ──────────────────────────────────
cells.append(md(
"## Common Mistakes\n"
"\n"
"| Mistake | Why it fails | Fix |\n"
"|---------|-------------|-----|\n"
"| Freshness weight too high | Retrieves recent but irrelevant docs | Tune w on held-out queries |\n"
"| Freshness weight too low | Still returns stale docs first | Increase w or add metadata filter |\n"
"| No staleness labels | User trusts outdated info | Always show document date + freshness label |\n"
"| Same w for all queries | Historical queries penalized | Classify query time-sensitivity first |\n"
"| Ignoring fast-changing topics | CEO names, prices change suddenly | Use shorter max_age for volatile topics |\n"
"| Hard filter only | Throws away useful historical context | Combine soft recency + optional hard filter |\n"
))

# ── 25. Mini Challenge ───────────────────────────────────
cells.append(md(
"## Mini Challenge\n"
"\n"
"1. **Query-time classification.** Build a classifier that decides\n"
"   whether a query is time-sensitive (\"latest\", \"current\") vs\n"
"   time-neutral (\"explain\", \"what is\"). Apply freshness only\n"
"   when time-sensitive.\n"
"\n"
"2. **Non-linear decay.** Implement exponential or step-function\n"
"   decay instead of linear. Compare which better models real\n"
"   information staleness.\n"
"\n"
"3. **Contradiction detection.** When both old and new articles\n"
"   are retrieved on the same topic, detect factual contradictions\n"
"   and flag them to the user.\n"
"\n"
"4. **Topic-specific decay rates.** Technology news goes stale\n"
"   faster than science discoveries. Implement per-topic decay.\n"
"\n"
"5. **Combine with Project 26 (Multi-Hop).** For multi-hop queries,\n"
"   apply freshness scoring at each hop independently.\n"
))

# ── 26. Production Considerations ─────────────────────────
cells.append(md(
"## Production Considerations\n"
"\n"
"| Aspect | Approach |\n"
"|--------|----------|\n"
"| **Date metadata** | Store publish date in vector DB metadata; index for filtering |\n"
"| **Dynamic reference** | Use current server time, not hardcoded date |\n"
"| **Query classification** | LLM or classifier to detect time-sensitive queries |\n"
"| **Per-topic decay** | Different max_age per content category |\n"
"| **UI integration** | Show document date prominently; color-code freshness |\n"
"| **Crawl frequency** | Re-index frequently changing sources more often |\n"
"| **TTL (time-to-live)** | Auto-expire or re-verify old documents |\n"
"| **Versioning** | Track document versions; link old to updated versions |\n"
))

# ── 27. How to Improve ───────────────────────────────────
cells.append(md(
"## How to Improve This Project\n"
"\n"
"1. **LLM query classifier** -- Detect whether the query needs recency.\n"
"2. **Exponential decay** -- Better models sudden information changes.\n"
"3. **Contradiction detector** -- Flag when old and new articles conflict.\n"
"4. **Topic-aware decay** -- Faster decay for volatile topics (crypto, tech).\n"
"5. **User preference** -- Let users choose \"latest only\" vs \"all time\".\n"
"6. **Source credibility** -- Weight authoritative sources higher regardless of age.\n"
"7. **Larger corpus** -- Test with thousands of articles.\n"
))

# ── 28. Key Takeaways ────────────────────────────────────
cells.append(md(
"## Key Takeaways\n"
"\n"
"1. **Standard RAG ignores time.** Semantic similarity alone retrieves\n"
"   outdated information when topics evolve.\n"
"\n"
"2. **Recency scoring is a simple, effective fix.** A weighted blend\n"
"   of similarity + recency significantly improves freshness.\n"
"\n"
"3. **The freshness weight is the key parameter.** Too low = stale\n"
"   results. Too high = irrelevant recent results. w=0.2-0.4 is\n"
"   usually a good starting point.\n"
"\n"
"4. **Metadata filters provide hard guarantees.** When you absolutely\n"
"   need recent data, use a date cutoff before scoring.\n"
"\n"
"5. **Always label staleness.** Even if you retrieve old documents\n"
"   (sometimes needed for context), warn the user explicitly.\n"
"\n"
"6. **Not all queries need freshness.** Historical, definitional, and\n"
"   scientific queries should use standard similarity. Only time-\n"
"   sensitive queries benefit from recency boosting.\n"
"\n"
"7. **Composable with other techniques.** Freshness (P28) + multi-hop\n"
"   (P26) + table+text (P27) + citation verification (P24) =\n"
"   production-grade RAG for dynamic knowledge bases.\n"
))

nb.cells = cells
nbformat.validate(nb)

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Written: {NB_PATH}")
print(f"Cells: {len(cells)}")
