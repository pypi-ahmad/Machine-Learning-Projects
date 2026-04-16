"""Check state of 10 business NLP notebooks (81-90)."""
import os

BASE = r"E:\Github\Machine-Learning-Projects\NLP"
nbs = [
    ("81", "Support Ticket Triage System", "support_ticket_triage.ipynb"),
    ("82", "Sales Call Note Summarizer", "sales_call_note_summarizer.ipynb"),
    ("83", "Procurement Document Analyzer", "procurement_document_analyzer.ipynb"),
    ("84", "RFP Response Automation", "rfp_response_automation.ipynb"),
    ("85", "Compliance Policy Comparator", "compliance_policy_comparator.ipynb"),
    ("86", "Churn Reason Miner", "churn_reason_miner.ipynb"),
    ("87", "Voice-of-Customer Dashboard Notebook", "voice_of_customer_dashboard.ipynb"),
    ("88", "Analyst Memo Generator", "analyst_memo_generator.ipynb"),
    ("89", "Knowledge Base Gap Detector", "knowledge_base_gap_detector.ipynb"),
    ("90", "Executive Brief Generator", "executive_brief_generator.ipynb"),
]
for num, folder, nb in nbs:
    path = os.path.join(BASE, folder, nb)
    if os.path.exists(path):
        c = open(path, "r", encoding="utf-8").read()
        lines = len(c.splitlines())
        is_pct = c.strip().startswith("#%%")
        has_raw = '"cell_type": "raw"' in c or '"cell_type":"raw"' in c
        has_ollama = "ChatOllama" in c
        has_triple = '\"\"\"' in c  # escaped triple quotes (bad)
        has_good_triple = '"""' in c  # proper triple quotes
        fmt = "percent-OK" if is_pct and not has_raw else ("BROKEN" if has_raw else "other")
        print(f"{num}. {nb:<55} lines={lines:>4} fmt={fmt:<12} ollama={has_ollama} triple_ok={has_good_triple and not has_triple}")
    else:
        print(f"{num}. {nb:<55} NOT FOUND")
