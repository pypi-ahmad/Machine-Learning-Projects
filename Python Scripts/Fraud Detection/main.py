"""Fraud Detection — Streamlit ML demo.

Detect fraudulent transactions using rule-based scoring and
anomaly detection. Interactive transaction analyzer.

Usage:
    streamlit run main.py
"""

import math
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🛡️ Fraud Detection System")


# ── Rule-based fraud scoring ──────────────────────────────────────────────────

RISK_RULES = [
    ("High amount",           lambda t: 1.0 if t["amount"] > 5000 else (0.5 if t["amount"] > 1000 else 0.0)),
    ("Unusual hour",          lambda t: 0.8 if t["hour"] in range(0, 5) else 0.0),
    ("Foreign country",       lambda t: 0.6 if t["country"] != "US" else 0.0),
    ("New merchant",          lambda t: 0.5 if t["merchant_age_days"] < 30 else 0.0),
    ("Multiple declines",     lambda t: 0.9 if t["prior_declines"] >= 3 else 0.0),
    ("Card not present",      lambda t: 0.4 if not t["card_present"] else 0.0),
    ("High velocity",         lambda t: 0.7 if t["txns_last_hour"] >= 5 else 0.0),
    ("Cross-border",          lambda t: 0.5 if t["cross_border"] else 0.0),
    ("Unusual category",      lambda t: 0.4 if t["category"] in ["gambling","crypto","adult"] else 0.0),
    ("Low balance ratio",     lambda t: 0.6 if t["balance_ratio"] < 0.05 else 0.0),
]


def fraud_score(txn: dict) -> dict:
    scores = {}
    total  = 0.0
    for name, fn in RISK_RULES:
        s = fn(txn)
        scores[name] = s
        total += s

    # Normalize to [0, 1]
    max_possible = len(RISK_RULES)
    risk = min(total / max_possible * 2, 1.0)   # scale so common frauds reach ~0.8+

    if   risk >= 0.7: label = "HIGH RISK 🔴"
    elif risk >= 0.4: label = "MEDIUM RISK 🟡"
    else:             label = "LOW RISK 🟢"

    return {"score": risk, "label": label, "rules": scores, "total_flags": total}


def make_transactions(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng    = random.Random(seed)
    cats   = ["retail","food","travel","electronics","gambling","crypto","entertainment","utilities"]
    countries = ["US","US","US","US","US","GB","DE","CN","NG","RU"]
    rows   = []
    base_time = datetime(2024, 1, 1)
    for i in range(n):
        is_fraud = rng.random() < 0.08
        amount   = rng.gauss(2000, 3000) if is_fraud else rng.gauss(150, 200)
        hour     = rng.randint(0, 4) if is_fraud else rng.randint(8, 22)
        country  = rng.choice(["US","US","GB","DE","CN"]) if is_fraud else "US"
        txn = {
            "txn_id":          f"TXN{i:05d}",
            "amount":          round(max(1, amount), 2),
            "hour":            hour,
            "country":         country,
            "merchant_age_days": rng.randint(1, 10) if is_fraud else rng.randint(30, 3000),
            "prior_declines":  rng.randint(2, 5) if is_fraud else rng.randint(0, 1),
            "card_present":    False if is_fraud else rng.choice([True, True, False]),
            "txns_last_hour":  rng.randint(4, 10) if is_fraud else rng.randint(0, 3),
            "cross_border":    country != "US",
            "category":        rng.choice(["gambling","crypto"]) if is_fraud else rng.choice(cats),
            "balance_ratio":   rng.uniform(0.01, 0.1) if is_fraud else rng.uniform(0.1, 0.9),
            "true_fraud":      is_fraud,
        }
        txn["timestamp"] = str(base_time + timedelta(minutes=i*5+rng.randint(0,4)))
        rows.append(txn)
    return pd.DataFrame(rows)


@st.cache_data
def load_data():
    return make_transactions(300)


df_all = load_data()

# ── Apply scores ──────────────────────────────────────────────────────────────
@st.cache_data
def score_all():
    rows = []
    for _, row in df_all.iterrows():
        r = fraud_score(row.to_dict())
        rows.append({
            "txn_id":    row["txn_id"],
            "amount":    row["amount"],
            "country":   row["country"],
            "category":  row["category"],
            "score":     round(r["score"], 3),
            "label":     r["label"],
            "true_fraud": row["true_fraud"],
        })
    return pd.DataFrame(rows)


scored_df = score_all()

# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Analyze Transaction", "Transaction Monitor", "Model Stats"])

with tab1:
    st.subheader("Analyze a Single Transaction")
    c1, c2 = st.columns(2)
    amount    = c1.number_input("Amount ($)", 0.01, 100000.0, 150.0, step=10.0)
    hour      = c2.slider("Transaction Hour (0–23)", 0, 23, 14)
    c1, c2    = st.columns(2)
    country   = c1.selectbox("Country", ["US","GB","DE","CN","NG","RU","FR","JP"])
    category  = c2.selectbox("Category",
                              ["retail","food","travel","electronics","gambling","crypto","utilities"])
    c1, c2    = st.columns(2)
    card_present   = c1.checkbox("Card Present", value=True)
    cross_border   = c2.checkbox("Cross-Border")
    c1, c2, c3     = st.columns(3)
    merchant_age   = c1.number_input("Merchant Age (days)", 0, 5000, 365)
    prior_declines = c2.slider("Prior Declines", 0, 10, 0)
    txns_hour      = c3.slider("Transactions Last Hour", 0, 20, 1)
    balance_ratio  = st.slider("Balance Ratio", 0.0, 1.0, 0.5, step=0.01)

    txn = {
        "amount": amount, "hour": hour, "country": country,
        "merchant_age_days": merchant_age, "prior_declines": prior_declines,
        "card_present": card_present, "txns_last_hour": txns_hour,
        "cross_border": cross_border, "category": category,
        "balance_ratio": balance_ratio,
    }
    result = fraud_score(txn)

    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Risk Score", f"{result['score']:.3f}")
    col2.metric("Classification", result["label"])
    st.progress(result["score"])

    triggered = {k: v for k, v in result["rules"].items() if v > 0}
    if triggered:
        st.warning("Triggered rules: " + ", ".join(f"{k} ({v:.1f})" for k, v in triggered.items()))
    else:
        st.success("No risk rules triggered.")

with tab2:
    st.subheader("Live Transaction Feed")
    risk_filter = st.multiselect("Filter by risk", ["HIGH RISK 🔴","MEDIUM RISK 🟡","LOW RISK 🟢"],
                                  default=["HIGH RISK 🔴","MEDIUM RISK 🟡"])
    filtered = scored_df[scored_df["label"].isin(risk_filter)] if risk_filter else scored_df
    st.dataframe(filtered.head(50), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    high = (scored_df["label"] == "HIGH RISK 🔴").sum()
    med  = (scored_df["label"] == "MEDIUM RISK 🟡").sum()
    low  = (scored_df["label"] == "LOW RISK 🟢").sum()
    c1.metric("High Risk",   high)
    c2.metric("Medium Risk", med)
    c3.metric("Low Risk",    low)

with tab3:
    threshold = st.slider("Score threshold", 0.0, 1.0, 0.4, step=0.05)
    preds     = scored_df["score"] >= threshold
    actual    = scored_df["true_fraud"]
    tp = (preds & actual).sum()
    fp = (preds & ~actual).sum()
    fn = (~preds & actual).sum()
    tn = (~preds & ~actual).sum()
    prec = tp / (tp+fp) if tp+fp else 0
    rec  = tp / (tp+fn) if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    acc  = (tp+tn) / len(preds)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.2%}")
    c2.metric("Precision", f"{prec:.2%}")
    c3.metric("Recall",    f"{rec:.2%}")
    c4.metric("F1 Score",  f"{f1:.2%}")
    st.caption(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
