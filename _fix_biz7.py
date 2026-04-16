"""Fix the 7 notebooks that have merge artifacts by writing clean content."""
import os

BASE = r"E:\Github\Machine-Learning-Projects\NLP"

SETUP = '''import os, json, re, warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

LLM_MODEL = "qwen3.5:9b"
llm = ChatOllama(model=LLM_MODEL, temperature=0)

def clean(text):
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()

def ask(system, user):
    return clean(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)

print(f"LLM ready: {ask('Reply OK.', 'Test')[:20]}")'''

PARSE = '''    text = resp
    if "```" in text:
        text = re.sub(r"```(?:json)?\\s*", "", text).replace("```", "")
    s, e = text.find("{"), text.rfind("}") + 1'''


def write(folder, filename, content):
    path = os.path.join(BASE, folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    lines = len(content.splitlines())
    print(f"  WROTE {filename:<55} {lines:>4} lines")


# --- 81 ---
NB81 = '''#%% md
# Support Ticket Triage System

## 1. Project Overview
**Task:** Route support tickets by urgency and team using LLM classification.
**Stack:** `LangChain` + `ChatOllama` + `qwen3.5:9b`.
#%% md
## 2. Why This Matters
- Reduce response time by routing tickets to the right team instantly
- Flag critical issues for immediate human attention
- Reduce misrouting that causes ticket ping-pong
#%% md
## 3. Setup
#%%
from collections import Counter
''' + SETUP + '''
#%% md
## 4. Sample Tickets
#%%
TICKETS = [
    {"id": "T-001", "subject": "Payment failed", "body": "I tried to pay for my subscription but the payment keeps failing. Error code 4032. I need this resolved ASAP, my team depends on it.", "channel": "email"},
    {"id": "T-002", "subject": "Feature request", "body": "It would be great if you could add dark mode to the mobile app. Several users have requested this.", "channel": "web"},
    {"id": "T-003", "subject": "URGENT: Data breach suspected", "body": "We noticed unauthorized access to our account. Multiple logins from unknown IPs. Please investigate immediately.", "channel": "phone"},
    {"id": "T-004", "subject": "Can't reset password", "body": "The password reset email never arrives. I've checked spam. Been locked out for 2 days.", "channel": "chat"},
    {"id": "T-005", "subject": "Billing discrepancy", "body": "I was charged twice for the same month. Transaction IDs: TXN-445 and TXN-446. Please refund.", "channel": "email"},
    {"id": "T-006", "subject": "API rate limit too low", "body": "Our integration hits the rate limit every day around 2pm. We need the enterprise tier limit. Current plan: Pro.", "channel": "web"},
    {"id": "T-007", "subject": "App crashes on startup", "body": "Since the latest update v3.2.1, the iOS app crashes immediately on launch. iPhone 15, iOS 17.4.", "channel": "app"},
    {"id": "T-008", "subject": "Cancel my account", "body": "I want to cancel my subscription effective immediately. Please confirm and process any refund.", "channel": "email"},
    {"id": "T-009", "subject": "Slow dashboard loading", "body": "The analytics dashboard takes over 30 seconds to load. It used to be instant. Started yesterday.", "channel": "chat"},
    {"id": "T-010", "subject": "Compliance audit request", "body": "We need SOC2 compliance documentation for our annual audit. Can you provide the latest report?", "channel": "email"},
]
print(f"Tickets: {len(TICKETS)}")
#%% md
---
## 5. Ticket Triage
#%%
TRIAGE_SYS = """Triage this support ticket. Return JSON:
{"urgency": "critical|high|medium|low",
 "team": "billing|security|engineering|product|account|support",
 "category": "bug|billing|security|feature_request|account|performance|compliance",
 "sentiment": "angry|frustrated|neutral|positive",
 "sla_hours": 1-72,
 "auto_reply_possible": true/false,
 "summary": "one-line summary"} /no_think"""

print("TICKET TRIAGE")
print("=" * 80)
results = []
for t in TICKETS:
    resp = ask(TRIAGE_SYS, f"Subject: {t['subject']}\\nBody: {t['body']}\\nChannel: {t['channel']}")
''' + PARSE + '''
    r = {"urgency": "?", "team": "?", "category": "?"}
    if s >= 0 and e > s:
        try:
            r = json.loads(text[s:e])
        except Exception:
            pass
    r["id"] = t["id"]
    r["subject"] = t["subject"]
    results.append(r)
    print(f"  {t['id']} [{r.get('urgency','?'):<8}] -> {r.get('team','?'):<12} | {t['subject'][:40]}")
#%% md
## 6. Queue by Urgency
#%%
URG = {"critical": 0, "high": 1, "medium": 2, "low": 3}
for r in sorted(results, key=lambda x: URG.get(x.get("urgency", "low"), 4)):
    print(f"  [{r.get('urgency','?'):<8}] {r.get('team','?'):<12} SLA={r.get('sla_hours','?')}h | {r['subject'][:45]}")
#%% md
## 7. Team Distribution
#%%
tc = Counter(r.get("team", "?") for r in results)
print("TEAM DISTRIBUTION")
for team, cnt in tc.most_common():
    print(f"  {team:<15} {cnt:>2} {'#' * (cnt * 4)}")

uc = Counter(r.get("urgency", "?") for r in results)
print("\\nURGENCY DISTRIBUTION")
for urg, cnt in uc.most_common():
    print(f"  {urg:<10} {cnt:>2}")
#%% md
## 8. Auto-Reply Candidates
#%%
auto = [r for r in results if r.get("auto_reply_possible")]
print(f"AUTO-REPLY CANDIDATES: {len(auto)}/{len(results)}")
for r in auto:
    print(f"  {r['id']} {r['subject'][:50]}")
#%% md
## 9. Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | LLM triage classifies urgency, team, and category in one pass |
| 2 | SLA-based queuing ensures critical tickets get immediate attention |
| 3 | Auto-reply detection reduces support workload |
| 4 | Multi-channel tickets need consistent routing |

---
*NLP Project 81 of 100*
'''

write("Support Ticket Triage System", "support_ticket_triage.ipynb", NB81)

# --- 83 ---
NB83 = '''#%% md
# Procurement Document Analyzer

## 1. Project Overview
**Task:** Compare vendor proposals side-by-side, extracting pricing, terms, SLAs, and key differentiators.
**Stack:** `LangChain` + `ChatOllama` + `qwen3.5:9b`.
#%% md
## 2. Why This Matters
- Procurement teams review dozens of proposals manually
- Side-by-side comparison accelerates vendor selection
- Standardized extraction ensures nothing is missed
#%% md
## 3. Setup
#%%
''' + SETUP + '''
#%% md
## 4. Sample Vendor Proposals
#%%
PROPOSALS = {
    "VendorA": """CloudStore Enterprise Storage Solution
Pricing: $45,000/year for 10TB, $4,000/TB additional
Contract: 2-year minimum, annual billing
SLA: 99.95% uptime, 4-hour response for critical issues
Migration: Free data migration, estimated 2-week timeline
Security: SOC2 Type II, ISO 27001, encryption at rest and transit
Support: 24/7 phone and email, dedicated account manager
Scalability: Auto-scaling up to 100TB, API access included""",

    "VendorB": """DataVault Pro Cloud Storage
Pricing: $38,000/year for 10TB, $3,500/TB additional
Contract: 1-year minimum, monthly or annual billing
SLA: 99.9% uptime, 8-hour response for critical issues
Migration: Migration support at $5,000 flat fee, 3-week timeline
Security: SOC2 Type II, HIPAA compliant, encryption at rest
Support: Business hours phone, 24/7 email, shared account manager
Scalability: Manual scaling requests, API access at additional cost""",

    "VendorC": """NexaCloud Storage Platform
Pricing: $52,000/year for 10TB, $4,500/TB additional
Contract: 3-year minimum, annual billing, 15% discount for prepay
SLA: 99.99% uptime, 1-hour response for critical issues
Migration: White-glove migration included, 1-week timeline
Security: SOC2 Type II, ISO 27001, HIPAA, FedRAMP moderate
Support: 24/7 phone/email/chat, dedicated team of 3
Scalability: Instant auto-scaling to 500TB, full API suite"""
}
print(f"Proposals: {len(PROPOSALS)} vendors")
#%% md
---
## 5. Proposal Extraction
#%%
EXTRACT_SYS = """Extract structured data from this vendor proposal. Return JSON:
{"vendor": "...", "annual_cost_10tb": 0, "cost_per_additional_tb": 0,
 "min_contract_years": 0, "uptime_sla": "...", "response_time": "...",
 "migration_cost": "...", "migration_timeline": "...",
 "certifications": [], "support_level": "...",
 "strengths": [], "weaknesses": []} /no_think"""

print("PROPOSAL EXTRACTION")
print("=" * 70)
extracted = {}
for vendor, proposal in PROPOSALS.items():
    resp = ask(EXTRACT_SYS, f"Proposal:\\n{proposal}")
''' + PARSE + '''
    r = {"vendor": vendor}
    if s >= 0 and e > s:
        try:
            r = json.loads(text[s:e])
        except Exception:
            pass
    r["vendor"] = vendor
    extracted[vendor] = r
    print(f"\\n  {vendor}: ${r.get('annual_cost_10tb', '?')}/yr, {r.get('uptime_sla', '?')} uptime, {r.get('min_contract_years', '?')}yr min")
#%% md
## 6. Side-by-Side Comparison
#%%
fields = ["annual_cost_10tb", "cost_per_additional_tb", "min_contract_years", "uptime_sla", "response_time", "migration_cost"]
print("COMPARISON TABLE")
print("=" * 80)
header = f"  {'Metric':<25}" + "".join(f"{v:<18}" for v in PROPOSALS.keys())
print(header)
print("-" * 79)
for field in fields:
    row = f"  {field:<25}"
    for v in PROPOSALS.keys():
        val = str(extracted.get(v, {}).get(field, "?"))[:16]
        row += f"{val:<18}"
    print(row)
#%% md
## 7. Recommendation
#%%
REC_SYS = """Based on these vendor proposals, provide a recommendation.
Consider: cost, SLA, security, support, migration ease.
Return JSON: {"recommended": "vendor", "rationale": "brief", "risks": ["risk"], "negotiation_points": ["point"]} /no_think"""
summary = json.dumps(extracted, default=str)
resp = ask(REC_SYS, f"Extracted proposals:\\n{summary}")
''' + PARSE + '''
if s >= 0 and e > s:
    try:
        rec = json.loads(text[s:e])
        print(f"RECOMMENDATION: {rec.get('recommended', '?')}")
        print(f"Rationale: {rec.get('rationale', '?')}")
        print(f"Risks: {', '.join(rec.get('risks', []))}")
    except Exception:
        print(resp[:300])
#%% md
## 8. Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | Structured extraction enables apples-to-apples comparison |
| 2 | LLMs identify strengths/weaknesses beyond raw numbers |
| 3 | Automated comparison tables save hours of manual work |
| 4 | Negotiation points surface from gap analysis |

---
*NLP Project 83 of 100*
'''

write("Procurement Document Analyzer", "procurement_document_analyzer.ipynb", NB83)

# --- 86 ---
NB86 = '''#%% md
# Churn Reason Miner

## 1. Project Overview
**Task:** Extract likely churn causes from customer feedback, exit surveys, and support tickets.
**Stack:** `LangChain` + `ChatOllama` + `qwen3.5:9b`.
#%% md
## 2. Why This Matters
- Understanding churn drivers enables targeted retention
- Pattern mining reveals systemic issues vs one-off complaints
- Quantifying churn reasons justifies product investment
#%% md
## 3. Setup
#%%
from collections import Counter
''' + SETUP + '''
#%% md
## 4. Churned Customer Feedback
#%%
FEEDBACK = [
    {"customer": "C-101", "type": "exit_survey", "text": "Your pricing increased 40% with no new features. We found a cheaper alternative that does everything we need."},
    {"customer": "C-102", "type": "support_ticket", "text": "This is the 5th time this month the dashboard has been down. We can't run our business on unreliable software."},
    {"customer": "C-103", "type": "exit_survey", "text": "The product is great but we're downsizing and can't justify the cost for our smaller team."},
    {"customer": "C-104", "type": "cancellation_note", "text": "Switching to CompetitorY. Their API is much better documented and their integration with Salesforce actually works."},
    {"customer": "C-105", "type": "exit_survey", "text": "Support response times have gotten worse. Last ticket took 5 days. We need a vendor who is responsive."},
    {"customer": "C-106", "type": "support_ticket", "text": "The mobile app has been broken for 3 months. No updates, no communication. We're done waiting."},
    {"customer": "C-107", "type": "exit_survey", "text": "We loved the product but our company was acquired and the new parent company has a different vendor."},
    {"customer": "C-108", "type": "cancellation_note", "text": "Missing critical features: SSO, audit logs, role-based access. These are table stakes for enterprise."},
    {"customer": "C-109", "type": "exit_survey", "text": "Onboarding was terrible. 3 months in and half our team still doesn't know how to use core features."},
    {"customer": "C-110", "type": "exit_survey", "text": "Price increase plus removing features from our tier. Feels like a bait and switch."},
]
print(f"Churned customers: {len(FEEDBACK)}")
#%% md
---
## 5. Churn Reason Extraction
#%%
CHURN_SYS = """Analyze this churned customer feedback. Return JSON:
{"primary_reason": "pricing|reliability|competition|support|features|company_change|onboarding",
 "secondary_reasons": [],
 "preventable": true/false,
 "sentiment_intensity": 1-5,
 "specific_issues": ["issue"],
 "retention_action": "what could have prevented this"} /no_think"""

print("CHURN ANALYSIS")
print("=" * 70)
results = []
for f in FEEDBACK:
    resp = ask(CHURN_SYS, f"[{f['type']}] {f['text']}")
''' + PARSE + '''
    r = {"primary_reason": "?"}
    if s >= 0 and e > s:
        try:
            r = json.loads(text[s:e])
        except Exception:
            pass
    r["customer"] = f["customer"]
    results.append(r)
    prev = "PREVENTABLE" if r.get("preventable") else "external"
    print(f"  {f['customer']} [{r.get('primary_reason','?'):<12}] {prev:<12} {f['text'][:45]}...")
#%% md
## 6. Churn Pattern Analysis
#%%
reasons = Counter(r.get("primary_reason", "?") for r in results)
print("CHURN REASONS")
print("=" * 40)
for reason, cnt in reasons.most_common():
    print(f"  {reason:<15} {cnt:>2} {'#' * (cnt * 4)}")

preventable = sum(1 for r in results if r.get("preventable"))
print(f"\\nPreventable: {preventable}/{len(results)} ({preventable / len(results) * 100:.0f}%)")
#%% md
## 7. Retention Recommendations
#%%
REC_SYS = """Based on these churn patterns, provide top 3 retention recommendations.
Return JSON: {"recommendations": [{"priority": 1, "action": "...", "impact": "...", "effort": "low|medium|high"}]} /no_think"""
pattern_summary = json.dumps(dict(reasons))
resp = ask(REC_SYS, f"Churn reasons: {pattern_summary}\\nPreventable: {preventable}/{len(results)}")
''' + PARSE + '''
if s >= 0 and e > s:
    try:
        recs = json.loads(text[s:e]).get("recommendations", [])
        print("RETENTION RECOMMENDATIONS")
        for r in recs:
            print(f"  #{r.get('priority', '?')} [{r.get('effort', '?')}] {r.get('action', '?')}")
    except Exception:
        pass
#%% md
## 8. Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | Pricing and reliability are top churn drivers |
| 2 | Preventable churn represents recoverable revenue |
| 3 | Pattern mining reveals systemic vs isolated issues |
| 4 | Retention recommendations tie directly to root causes |

---
*NLP Project 86 of 100*
'''

write("Churn Reason Miner", "churn_reason_miner.ipynb", NB86)

# --- 87 ---
NB87 = '''#%% md
# Voice-of-Customer Dashboard Notebook

## 1. Project Overview
**Task:** Summarize customer themes and pain points from multi-channel feedback into an executive-ready VoC report.
**Stack:** `LangChain` + `ChatOllama` + `qwen3.5:9b`.
#%% md
## 2. Why This Matters
- Unifies feedback from surveys, tickets, reviews, and social
- Identifies top themes and sentiment trends
- Provides actionable insights for product and CX teams
#%% md
## 3. Setup
#%%
from collections import Counter
''' + SETUP + '''
#%% md
## 4. Multi-Channel Feedback
#%%
FEEDBACK = [
    {"source": "NPS survey", "text": "Love the product but wish the mobile app was better", "score": 8},
    {"source": "NPS survey", "text": "Support team is amazing, always responsive", "score": 9},
    {"source": "NPS survey", "text": "Too expensive for what it offers", "score": 4},
    {"source": "support_ticket", "text": "Dashboard loading is extremely slow since last update"},
    {"source": "support_ticket", "text": "Can't export reports to PDF, need this urgently"},
    {"source": "app_review", "text": "Great features but crashes on Android frequently", "rating": 3},
    {"source": "app_review", "text": "Best tool for project management, highly recommend", "rating": 5},
    {"source": "social", "text": "@YourProduct the new update broke our workflow, please fix ASAP"},
    {"source": "social", "text": "Just discovered @YourProduct and it's a game changer for our team!"},
    {"source": "NPS survey", "text": "Missing integrations with our existing tools", "score": 5},
    {"source": "support_ticket", "text": "Billing shows wrong amount, need immediate correction"},
    {"source": "app_review", "text": "Onboarding was confusing, took weeks to figure out basics", "rating": 2},
]
print(f"Feedback: {len(FEEDBACK)} items from {len(set(f['source'] for f in FEEDBACK))} channels")
#%% md
---
## 5. Theme Extraction
#%%
VOC_SYS = """Analyze this batch of customer feedback. Identify top themes.
Return JSON: {"themes": [{"name": "...", "count": 0, "sentiment": "positive|negative|mixed",
  "channels": ["source"], "example_quotes": ["quote"], "priority": "high|medium|low"}],
 "overall_sentiment": "positive|negative|mixed",
 "nps_insight": "brief NPS interpretation"} /no_think"""

feedback_text = "\\n".join(f"[{f['source']}] {f['text']}" for f in FEEDBACK)
resp = ask(VOC_SYS, f"Feedback:\\n{feedback_text}")
''' + PARSE + '''
print("VOC THEMES")
print("=" * 60)
voc = {}
if s >= 0 and e > s:
    try:
        voc = json.loads(text[s:e])
        for t in voc.get("themes", []):
            print(f"  [{t.get('priority', '?'):<6}] [{t.get('sentiment', '?'):<8}] {t.get('name', '?')} ({t.get('count', '?')})")
        print(f"\\nOverall: {voc.get('overall_sentiment', '?')}")
    except Exception:
        print(text[:400])
#%% md
## 6. Per-Feedback Classification
#%%
CLS_SYS = """Classify feedback. Return JSON:
{"theme": "product|support|pricing|performance|features|onboarding|billing",
 "sentiment": "positive|negative|neutral", "urgency": "high|medium|low"} /no_think"""

print("PER-FEEDBACK CLASSIFICATION")
classified = []
for f in FEEDBACK:
    resp = ask(CLS_SYS, f"[{f['source']}] {f['text']}")
''' + PARSE + '''
    c = {"theme": "?", "sentiment": "?"}
    if s >= 0 and e > s:
        try:
            c = json.loads(text[s:e])
        except Exception:
            pass
    c["text"] = f["text"]
    classified.append(c)
    print(f"  [{c.get('theme', '?'):<12}] [{c.get('sentiment', '?'):<8}] {f['text'][:50]}...")
#%% md
## 7. Executive VoC Summary
#%%
EXEC_SYS = "Write a concise executive VoC summary. Cover: top themes, sentiment, urgent issues, recommendations. /no_think"
summary_data = json.dumps([{"theme": c.get("theme"), "sentiment": c.get("sentiment"), "text": c["text"][:40]} for c in classified])
exec_summary = ask(EXEC_SYS, summary_data)
print("EXECUTIVE VOC SUMMARY")
print("=" * 60)
print(exec_summary)
#%% md
## 8. Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | Multi-channel aggregation reveals the full customer picture |
| 2 | Theme + sentiment analysis surfaces actionable patterns |
| 3 | Executive summaries make VoC insights accessible to leadership |

---
*NLP Project 87 of 100*
'''

write("Voice-of-Customer Dashboard Notebook", "voice_of_customer_dashboard.ipynb", NB87)

# --- 88 ---
NB88 = '''#%% md
# Analyst Memo Generator

## 1. Project Overview
**Task:** Turn raw analyst notes into structured, professional memos with key findings and recommendations.
**Stack:** `LangChain` + `ChatOllama` + `qwen3.5:9b`.
#%% md
## 2. Why This Matters
- Raw notes are messy and inconsistent
- Structured memos save hours of formatting
- Consistent format improves knowledge sharing
#%% md
## 3. Setup
#%%
''' + SETUP + '''
#%% md
## 4. Raw Analyst Notes
#%%
RAW_NOTES = """
Meeting with CFO re Q3 numbers. Revenue up 12% YoY but margins compressed 200bps due to cloud infra costs.
ARR hit $45M, up from $38M Q2. Net retention 115%. Gross churn 8% annualized - slightly above target.
Sales pipeline strong - $12M in qualified opps for Q4. Enterprise segment growing fastest at 25% QoQ.
Concerns: CAC payback extended to 18 months (was 14). Need to optimize paid channels.
Headcount: 245 employees, plan to hire 30 more by EOY mainly in engineering.
Cash position: $28M, runway ~24 months at current burn. No immediate fundraise needed.
Board wants path to profitability within 18 months. Considering reducing marketing spend.
Product: New analytics module launching Nov. Early beta feedback positive. Expect 15% upsell rate.
Competition: CompetitorZ raised $50M Series C. Aggressive pricing. Need to monitor churn in SMB.
"""
print(f"Raw notes: {len(RAW_NOTES.split())} words")
#%% md
---
## 5. Memo Generation
#%%
MEMO_SYS = """Transform these raw analyst notes into a structured memo. Use this format:
- EXECUTIVE SUMMARY (2-3 sentences)
- KEY METRICS (bullet list with numbers)
- STRENGTHS (bullet list)
- RISKS & CONCERNS (bullet list)
- RECOMMENDATIONS (numbered list)
- OUTLOOK (1-2 sentences)
Be precise, use the actual numbers from the notes. /no_think"""

memo = ask(MEMO_SYS, f"Raw Notes:\\n{RAW_NOTES}")
print("STRUCTURED MEMO")
print("=" * 70)
print(memo)
#%% md
## 6. Key Metrics Extraction
#%%
METRICS_SYS = """Extract all numerical metrics from these notes. Return JSON:
{"metrics": [{"name": "...", "value": "...", "trend": "up|down|flat", "context": "brief"}]} /no_think"""
resp = ask(METRICS_SYS, RAW_NOTES)
''' + PARSE + '''
print("KEY METRICS")
if s >= 0 and e > s:
    try:
        metrics = json.loads(text[s:e]).get("metrics", [])
        for m in metrics:
            print(f"  {m.get('name', '?'):<25} {str(m.get('value', '?')):<15} {m.get('trend', '?')}")
    except Exception:
        pass
#%% md
## 7. Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | Structured memos are generated from messy notes in seconds |
| 2 | Metric extraction ensures no numbers are missed |
| 3 | Consistent format improves cross-team communication |

---
*NLP Project 88 of 100*
'''

write("Analyst Memo Generator", "analyst_memo_generator.ipynb", NB88)

# --- 89 ---
NB89 = '''#%% md
# Knowledge Base Gap Detector

## 1. Project Overview
**Task:** Find unanswered question clusters in support data to identify KB gaps.
**Stack:** `LangChain` + `ChatOllama` + `qwen3.5:9b`.
#%% md
## 2. Why This Matters
- Unanswered questions create support ticket volume
- KB gaps cause repeated escalations
- Filling gaps reduces cost-per-ticket
#%% md
## 3. Setup
#%%
from collections import Counter
''' + SETUP + '''
#%% md
## 4. Support Questions & KB Coverage
#%%
QUESTIONS = [
    {"text": "How do I reset my password?", "answered_by_kb": True},
    {"text": "Can I integrate with Zapier?", "answered_by_kb": False},
    {"text": "What's your refund policy?", "answered_by_kb": True},
    {"text": "How to set up SSO with Okta?", "answered_by_kb": False},
    {"text": "Is there a mobile app?", "answered_by_kb": True},
    {"text": "How to configure SAML authentication?", "answered_by_kb": False},
    {"text": "Can I export data to CSV?", "answered_by_kb": True},
    {"text": "How do webhooks work in your API?", "answered_by_kb": False},
    {"text": "What's the rate limit for the API?", "answered_by_kb": False},
    {"text": "How to set up two-factor authentication?", "answered_by_kb": True},
    {"text": "Can I connect to Azure AD?", "answered_by_kb": False},
    {"text": "How to bulk import users?", "answered_by_kb": False},
    {"text": "What browsers are supported?", "answered_by_kb": True},
    {"text": "How to configure custom roles?", "answered_by_kb": False},
    {"text": "Is there an on-premise deployment option?", "answered_by_kb": False},
]
unanswered = [q for q in QUESTIONS if not q["answered_by_kb"]]
print(f"Total questions: {len(QUESTIONS)}")
print(f"Unanswered by KB: {len(unanswered)} ({len(unanswered) / len(QUESTIONS) * 100:.0f}%)")
#%% md
---
## 5. Gap Clustering
#%%
CLUSTER_SYS = """Group these unanswered support questions into topic clusters.
Return JSON: {"clusters": [{"topic": "...", "questions": ["q1"], "count": 0, "priority": "high|medium|low",
  "suggested_article_title": "..."}]} /no_think"""

q_list = "\\n".join(f"- {q['text']}" for q in unanswered)
resp = ask(CLUSTER_SYS, f"Unanswered questions:\\n{q_list}")
''' + PARSE + '''
print("GAP CLUSTERS")
print("=" * 60)
clusters = []
if s >= 0 and e > s:
    try:
        clusters = json.loads(text[s:e]).get("clusters", [])
        for c in clusters:
            print(f"  [{c.get('priority', '?'):<6}] {c.get('topic', '?')} ({c.get('count', '?')} questions)")
            print(f"    Suggested article: {c.get('suggested_article_title', '?')}")
            for q in c.get("questions", [])[:2]:
                print(f"      - {q}")
            print()
    except Exception:
        print(text[:400])
#%% md
## 6. Article Draft Generation
#%%
if clusters:
    top_gap = clusters[0]
    title = top_gap.get("suggested_article_title", "Help Article")
    questions_str = ", ".join(top_gap.get("questions", [])[:3])
    ARTICLE_SYS = f"Draft a KB article titled '{title}' answering these questions: {questions_str}. Be concise and helpful. /no_think"
    article = ask(ARTICLE_SYS, "Draft the article.")
    print("DRAFT KB ARTICLE")
    print("=" * 60)
    print(article[:500])
else:
    print("No clusters extracted - skipping article draft.")
#%% md
## 7. Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | 60%+ of unanswered questions cluster into a few topics |
| 2 | Filling top 3 gaps can reduce ticket volume significantly |
| 3 | LLM-generated drafts accelerate KB article creation |

---
*NLP Project 89 of 100*
'''

write("Knowledge Base Gap Detector", "knowledge_base_gap_detector.ipynb", NB89)

# --- 90 ---
NB90 = '''#%% md
# Executive Brief Generator

## 1. Project Overview
**Task:** Condense long reports into leadership-ready executive summaries with key metrics, risks, and recommendations.
**Stack:** `LangChain` + `ChatOllama` + `qwen3.5:9b`.
#%% md
## 2. Why This Matters
- Executives need 1-page summaries, not 20-page reports
- Consistent format across departments improves decision-making
- Automated briefs save hours of senior staff time
#%% md
## 3. Setup
#%%
''' + SETUP + '''
#%% md
## 4. Sample Report
#%%
REPORT = """QUARTERLY BUSINESS REVIEW - Q3 2025

FINANCIAL PERFORMANCE
Revenue reached $12.4M in Q3, representing 18% year-over-year growth and 5% quarter-over-quarter growth. Gross margin was 72%, down from 75% in Q2 due to increased cloud infrastructure costs. Operating expenses totaled $10.1M, with R&D accounting for 45%, Sales & Marketing 35%, and G&A 20%. EBITDA was negative $1.2M, an improvement from negative $2.1M in Q2. Cash position stands at $28M with 22 months of runway.

CUSTOMER METRICS
Total customers grew to 1,247 from 1,089 in Q2 (14.5% growth). Enterprise segment (>$50K ACV) grew 25% to 89 customers. Net Revenue Retention was 118%, above the 115% target. Gross churn was 2.1% quarterly (8.4% annualized), slightly above the 8% target. Average deal size increased 12% to $9,850 ACV. Customer acquisition cost (CAC) was $12,400, with payback period of 15 months.

PRODUCT & ENGINEERING
Shipped 3 major features: real-time analytics dashboard, API v3, and mobile app redesign. Sprint velocity improved 15%. Technical debt ratio decreased from 22% to 18%. Platform uptime was 99.97%. Two P1 incidents occurred, both resolved within 2 hours.

RISKS & CHALLENGES
1. CompetitorZ raised $50M and is offering aggressive discounts to our SMB segment.
2. Cloud costs growing faster than revenue (35% vs 18%).
3. Two senior engineers departed; backfill in progress.
4. Enterprise sales cycle lengthening from 45 to 62 days.

OUTLOOK
Q4 pipeline stands at $4.2M in qualified opportunities. Planning to launch AI-powered insights module. Targeting break-even by Q2 2026."""
print(f"Report: {len(REPORT.split())} words")
#%% md
---
## 5. Executive Brief Generation
#%%
BRIEF_SYS = """Condense this report into an executive brief. Format:
1. ONE-LINE HEADLINE (the single most important takeaway)
2. KEY METRICS (5-7 bullet points with numbers)
3. WINS (3 bullets)
4. RISKS (3 bullets, ranked by severity)
5. DECISIONS NEEDED (2-3 specific decisions for leadership)
6. OUTLOOK (2 sentences)

Keep it to ONE PAGE. Use actual numbers. Be direct. /no_think"""

brief = ask(BRIEF_SYS, f"Report:\\n{REPORT}")
print("EXECUTIVE BRIEF")
print("=" * 70)
print(brief)
#%% md
## 6. Metric Extraction
#%%
METRICS_SYS = """Extract the 10 most important metrics. Return JSON:
{"metrics": [{"name": "...", "value": "...", "vs_target": "above|below|on_target|no_target",
  "trend": "improving|declining|stable"}]} /no_think"""

resp = ask(METRICS_SYS, REPORT)
''' + PARSE + '''
print("TOP METRICS")
print("=" * 60)
if s >= 0 and e > s:
    try:
        metrics = json.loads(text[s:e]).get("metrics", [])
        for m in metrics[:10]:
            print(f"  {m.get('name', '?'):<30} {str(m.get('value', '?')):<15} {m.get('trend', '?')}")
    except Exception:
        pass
#%% md
## 7. Decision Brief
#%%
DECISION_SYS = """Based on this report, what are the 3 most important decisions leadership needs to make?
Return JSON: {"decisions": [{"decision": "...", "urgency": "immediate|this_quarter|next_quarter", "options": ["opt1", "opt2"], "recommendation": "..."}]} /no_think"""

resp = ask(DECISION_SYS, REPORT)
''' + PARSE + '''
print("DECISIONS NEEDED")
print("=" * 60)
if s >= 0 and e > s:
    try:
        decisions = json.loads(text[s:e]).get("decisions", [])
        for d in decisions:
            print(f"  [{d.get('urgency', '?')}] {d.get('decision', '?')}")
            print(f"    Recommendation: {d.get('recommendation', '')[:70]}")
            print()
    except Exception:
        pass
#%% md
## 8. Key Takeaways

| # | Takeaway |
|---|----------|
| 1 | Executive briefs distill 20-page reports into 1-page summaries |
| 2 | Metric extraction ensures key numbers aren't buried |
| 3 | Decision framing helps leadership act quickly |
| 4 | Consistent format across departments improves board readiness |

---
*NLP Project 90 of 100*
'''

write("Executive Brief Generator", "executive_brief_generator.ipynb", NB90)

print("\nAll 7 notebooks rewritten cleanly!")

