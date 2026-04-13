"""Survey Analyzer — Streamlit app.

Upload survey results CSV and get automatic analysis:
response distributions, likert scales, word clouds, and cross-tabs.

Usage:
    streamlit run main.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Survey Analyzer", layout="wide")
st.title("📋 Survey Analyzer")

SAMPLE = Path("sample_survey.csv")

def make_sample() -> pd.DataFrame:
    import random
    random.seed(0)
    n = 120
    return pd.DataFrame({
        "Age Group":         [random.choice(["18-24","25-34","35-44","45-54","55+"]) for _ in range(n)],
        "Gender":            [random.choice(["Male","Female","Non-binary","Prefer not to say"]) for _ in range(n)],
        "Satisfaction (1-5)":[random.choices([1,2,3,4,5], weights=[5,10,20,40,25])[0] for _ in range(n)],
        "Would Recommend":   [random.choice(["Yes","No","Maybe"]) for _ in range(n)],
        "Product Quality":   [random.choice(["Excellent","Good","Average","Poor"]) for _ in range(n)],
        "Support Quality":   [random.choices(["Excellent","Good","Average","Poor"],
                                              weights=[30,40,20,10])[0] for _ in range(n)],
        "Comments":          [random.choice(["Great product","Needs improvement",
                                              "Very happy","Could be better","Love it",
                                              "Fast delivery","Good value","Amazing service","Disappointed",""]) for _ in range(n)],
    })

uploaded = st.sidebar.file_uploader("Upload Survey CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.sidebar.info("Using sample survey data.")
    if not SAMPLE.exists():
        make_sample().to_csv(SAMPLE, index=False)
    df = pd.read_csv(SAMPLE)

st.caption(f"{len(df)} responses  ·  {len(df.columns)} questions")

numeric_cols  = df.select_dtypes(include="number").columns.tolist()
category_cols = df.select_dtypes(exclude="number").columns.tolist()
text_cols     = [c for c in category_cols if df[c].nunique() > 10 or
                  df[c].str.len().mean() > 15 if hasattr(df[c], "str") else False]
cat_cols      = [c for c in category_cols if c not in text_cols]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Cross-Tab", "Comments"])

with tab1:
    st.subheader("Response Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Responses", len(df))
    c2.metric("Questions", len(df.columns))
    c3.metric("Complete Rows", int(df.dropna().shape[0]))

    if numeric_cols:
        st.subheader("Numeric Question Summary")
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

with tab2:
    if cat_cols:
        sel_col = st.selectbox("Select question", cat_cols)
        counts  = df[sel_col].value_counts()
        st.bar_chart(counts)
        pct = (counts / len(df) * 100).round(1).rename("Percent %")
        st.dataframe(pd.concat([counts, pct], axis=1), use_container_width=True)

    if numeric_cols:
        sel_num = st.selectbox("Numeric question", numeric_cols)
        avg     = df[sel_num].mean()
        st.metric(f"Average {sel_num}", f"{avg:.2f}")
        hist = df[sel_num].value_counts().sort_index()
        st.bar_chart(hist)

with tab3:
    if len(cat_cols) >= 2:
        col_a = st.selectbox("Row variable",    cat_cols, index=0)
        col_b = st.selectbox("Column variable", cat_cols, index=min(1, len(cat_cols)-1))
        if col_a != col_b:
            ct = pd.crosstab(df[col_a], df[col_b])
            st.dataframe(ct, use_container_width=True)
            pct_ct = pd.crosstab(df[col_a], df[col_b], normalize="index").round(3) * 100
            st.caption("Row percentages:")
            st.dataframe(pct_ct.round(1), use_container_width=True)
    else:
        st.info("Need at least 2 categorical columns for cross-tab.")

with tab4:
    text_col = st.selectbox("Comments column", category_cols) if category_cols else None
    if text_col:
        comments = df[text_col].dropna()
        comments = comments[comments.str.strip() != ""]
        st.caption(f"{len(comments)} non-empty responses")
        for c in comments.sample(min(20, len(comments)), random_state=1):
            st.markdown(f"- {c}")
        # Word frequency
        from collections import Counter
        import re
        words = re.findall(r'\b\w{3,}\b', " ".join(comments).lower())
        stop  = {"the","and","for","are","this","that","was","with","have","not","but","its"}
        freq  = Counter(w for w in words if w not in stop).most_common(15)
        if freq:
            st.subheader("Most Common Words")
            wdf = pd.DataFrame(freq, columns=["Word","Count"]).set_index("Word")
            st.bar_chart(wdf)
