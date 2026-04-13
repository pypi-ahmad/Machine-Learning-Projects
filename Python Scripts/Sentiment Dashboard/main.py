"""Sentiment Dashboard — Streamlit ML demo.

Analyze sentiment of text, product reviews, or customer feedback.
Uses a lexicon-based approach (no external ML dependencies).

Usage:
    streamlit run main.py
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("💬 Sentiment Dashboard")

# ── Sentiment lexicon ─────────────────────────────────────────────────────────

POSITIVE = {
    "good","great","excellent","amazing","wonderful","fantastic","superb","outstanding",
    "perfect","love","best","awesome","brilliant","happy","pleased","satisfied","enjoyed",
    "delightful","positive","recommend","impressive","quality","nice","helpful","friendly",
    "fast","easy","clean","comfortable","beautiful","reliable","efficient","professional",
    "quick","smooth","fair","honest","clear","effective","useful","innovative","fun",
    "cheap","affordable","value","worth","enjoy","glad","thankful","grateful",
}

NEGATIVE = {
    "bad","terrible","awful","horrible","disgusting","worst","poor","disappointing",
    "frustrated","annoyed","angry","hate","dislike","broken","defective","useless",
    "slow","difficult","confusing","expensive","overpriced","waste","failed","problem",
    "issue","bug","error","crashed","refused","late","missing","wrong","dirty",
    "rude","unprofessional","unreliable","misleading","false","fake","scam","fraud",
    "ugly","cheap","flimsy","uncomfortable","regret","sorry","unhappy","dissatisfied",
}

NEGATION = {"not","no","never","don't","doesn't","didn't","isn't","aren't","won't",
             "can't","couldn't","wouldn't","shouldn't","hardly","barely","neither"}

INTENSIFIERS = {"very":1.5,"extremely":1.8,"absolutely":1.7,"incredibly":1.7,
                "really":1.3,"quite":1.2,"pretty":1.1,"somewhat":0.8,"slightly":0.7}


def analyze(text: str) -> dict:
    words  = re.findall(r"\b\w+\b", text.lower())
    score  = 0.0
    pos_w  = []
    neg_w  = []
    window = 3   # words in negation window

    for i, w in enumerate(words):
        if w in POSITIVE or w in NEGATIVE:
            multiplier = 1.0
            # Check intensifier in window
            for j in range(max(0, i-2), i):
                multiplier *= INTENSIFIERS.get(words[j], 1.0)
            # Check negation
            negated = any(words[k] in NEGATION for k in range(max(0, i-window), i))
            val = 1.0 if w in POSITIVE else -1.0
            if negated: val = -val
            val *= multiplier
            score += val
            if val > 0: pos_w.append(w)
            else:        neg_w.append(w)

    total = len(words) or 1
    norm  = score / (total ** 0.5)   # scale by sqrt(n)

    if   norm >  0.5: label, emoji = "Positive", "😊"
    elif norm < -0.5: label, emoji = "Negative", "😞"
    else:              label, emoji = "Neutral",  "😐"

    confidence = min(abs(norm) / 2, 1.0)

    return {
        "label":      label,
        "emoji":      emoji,
        "score":      round(norm, 3),
        "confidence": confidence,
        "pos_words":  pos_w,
        "neg_words":  neg_w,
        "word_count": len(words),
    }


# ── Sample reviews ─────────────────────────────────────────────────────────────

SAMPLE_REVIEWS = [
    "This product is absolutely amazing! Best purchase I've ever made.",
    "Terrible experience. The item arrived broken and customer service was rude.",
    "It's okay. Not great, not bad. Does what it's supposed to do.",
    "Really happy with this! Fast shipping and great quality. Would recommend.",
    "Very disappointing. The quality is poor and it stopped working after a week.",
    "Decent product for the price. Nothing extraordinary but value for money.",
    "Excellent! Exceeded my expectations. The customer support was friendly and helpful.",
    "Waste of money. Don't buy this. Completely useless and overpriced.",
    "Pretty good product. Minor issues but overall satisfied with the purchase.",
    "Outstanding service! Quick delivery and the product is perfect. Love it!",
]

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Analyze Text", "Batch Reviews", "Dashboard"])

with tab1:
    st.subheader("Single Text Analysis")
    text = st.text_area("Enter text to analyze", height=120,
                         placeholder="Type or paste text here…")
    if text.strip():
        result = analyze(text)
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", f"{result['emoji']} {result['label']}")
        col2.metric("Score",      f"{result['score']:.3f}")
        col3.metric("Confidence", f"{result['confidence']:.0%}")

        if result["pos_words"]:
            st.success(f"Positive signals: {', '.join(set(result['pos_words']))}")
        if result["neg_words"]:
            st.error(f"Negative signals: {', '.join(set(result['neg_words']))}")

with tab2:
    st.subheader("Batch Review Analysis")
    use_sample = st.checkbox("Use sample reviews", value=True)
    if use_sample:
        texts = SAMPLE_REVIEWS
    else:
        raw = st.text_area("Paste reviews (one per line)", height=200)
        texts = [t.strip() for t in raw.splitlines() if t.strip()]

    if texts:
        rows = []
        for i, t in enumerate(texts, 1):
            r = analyze(t)
            rows.append({
                "Review": t[:80] + "…" if len(t) > 80 else t,
                "Sentiment": f"{r['emoji']} {r['label']}",
                "Score":     r["score"],
                "Confidence": f"{r['confidence']:.0%}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        pos = sum(1 for r in rows if "Positive" in r["Sentiment"])
        neg = sum(1 for r in rows if "Negative" in r["Sentiment"])
        neu = len(rows) - pos - neg
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", f"{pos} ({pos/len(rows):.0%})")
        c2.metric("Neutral",  f"{neu} ({neu/len(rows):.0%})")
        c3.metric("Negative", f"{neg} ({neg/len(rows):.0%})")

with tab3:
    st.subheader("Sentiment Overview")
    results = [analyze(t) for t in SAMPLE_REVIEWS]
    labels  = [r["label"] for r in results]
    counts  = Counter(labels)
    st.bar_chart(pd.Series(counts))

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores)
    st.metric("Average Sentiment Score", f"{avg_score:.3f}")

    # Word cloud substitute: top positive/negative words
    all_pos = Counter(w for r in results for w in r["pos_words"])
    all_neg = Counter(w for r in results for w in r["neg_words"])
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Positive Words")
        if all_pos:
            st.bar_chart(pd.Series(dict(all_pos.most_common(10))))
    with col2:
        st.subheader("Top Negative Words")
        if all_neg:
            st.bar_chart(pd.Series(dict(all_neg.most_common(10))))
