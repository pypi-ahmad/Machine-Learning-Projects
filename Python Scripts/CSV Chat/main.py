"""CSV Chat — Streamlit demo.

Upload a CSV file and ask natural-language questions about it.
Answers using pattern-matched query generation against pandas,
with no external ML libraries required.

Usage:
    streamlit run main.py
"""

import math
import re

import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Chat", layout="wide")
st.title("💬 CSV Chat")
st.caption("Upload a CSV and ask questions in plain English. Powered by rule-based NL→pandas translation.")


# ── NL query engine ───────────────────────────────────────────────────────────

def find_column(df: pd.DataFrame, hint: str) -> str | None:
    """Best-match column by substring or token overlap."""
    hint_lower = hint.lower().strip()
    # Exact match first
    for c in df.columns:
        if c.lower() == hint_lower:
            return c
    # Substring match
    for c in df.columns:
        if hint_lower in c.lower() or c.lower() in hint_lower:
            return c
    # Token overlap
    h_tokens = set(re.findall(r"[a-z]+", hint_lower))
    best, best_score = None, 0
    for c in df.columns:
        c_tokens = set(re.findall(r"[a-z]+", c.lower()))
        score = len(h_tokens & c_tokens)
        if score > best_score:
            best, best_score = c, score
    return best if best_score > 0 else None


def find_numeric_col(df: pd.DataFrame, hint: str) -> str | None:
    col = find_column(df, hint)
    if col and pd.api.types.is_numeric_dtype(df[col]):
        return col
    # Fallback: first numeric column whose name overlaps
    for c in df.select_dtypes(include="number").columns:
        if any(t in c.lower() for t in re.findall(r"[a-z]+", hint.lower())):
            return c
    return None


def extract_number(text: str) -> int | None:
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None


def answer(df: pd.DataFrame, question: str) -> dict:
    q = question.lower().strip()
    result = {"answer": None, "df": None, "chart": None, "error": None}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(exclude="number").columns.tolist()

    # ── How many rows / shape ────────────────────────────────────────────────
    if re.search(r"how many (rows|records|entries|samples|observations)", q):
        result["answer"] = f"The dataset has **{len(df):,} rows** and **{len(df.columns)} columns**."
        return result

    if re.search(r"(columns|features|fields)", q) and re.search(r"(what|list|show|names?)", q):
        result["answer"] = "Columns: " + ", ".join(f"`{c}`" for c in df.columns)
        return result

    # ── Describe / summary ───────────────────────────────────────────────────
    if re.search(r"(describe|summary|statistics|stats)", q):
        result["df"] = df.describe().round(3)
        result["answer"] = "Statistical summary:"
        return result

    # ── Missing values ────────────────────────────────────────────────────────
    if re.search(r"(missing|null|na\b|nan|empty)", q):
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if miss.empty:
            result["answer"] = "No missing values found."
        else:
            result["df"]    = miss.reset_index().rename(columns={"index": "Column", 0: "Missing"})
            result["answer"] = f"Found missing values in {len(miss)} column(s):"
        return result

    # ── Average / mean ────────────────────────────────────────────────────────
    m = re.search(r"(average|mean|avg)\s+(?:of\s+|the\s+)?(.+?)(?:\s+by\s+(.+))?$", q)
    if m:
        col = find_numeric_col(df, m.group(2))
        grp = find_column(df, m.group(3)) if m.group(3) else None
        if col:
            if grp:
                result["df"]     = df.groupby(grp)[col].mean().reset_index().round(3)
                result["answer"] = f"Average `{col}` grouped by `{grp}`:"
                result["chart"]  = ("bar", grp, col)
            else:
                avg = df[col].mean()
                result["answer"] = f"Average of `{col}`: **{avg:,.3f}**"
        else:
            result["error"] = f"Could not identify a numeric column in: '{m.group(2)}'"
        return result

    # ── Sum / total ────────────────────────────────────────────────────────────
    m = re.search(r"(sum|total)\s+(?:of\s+|the\s+)?(.+?)(?:\s+by\s+(.+))?$", q)
    if m:
        col = find_numeric_col(df, m.group(2))
        grp = find_column(df, m.group(3)) if m.group(3) else None
        if col:
            if grp:
                result["df"]     = df.groupby(grp)[col].sum().reset_index().round(3)
                result["answer"] = f"Sum of `{col}` by `{grp}`:"
                result["chart"]  = ("bar", grp, col)
            else:
                result["answer"] = f"Total `{col}`: **{df[col].sum():,.3f}**"
        else:
            result["error"] = f"Could not identify a numeric column in: '{m.group(2)}'"
        return result

    # ── Max / min ────────────────────────────────────────────────────────────
    for agg_kw, agg_fn in [("max|maximum|highest|largest|top", "max"),
                            ("min|minimum|lowest|smallest|bottom", "min")]:
        m = re.search(rf"({agg_kw})\s+(?:of\s+|the\s+)?(.+)", q)
        if m:
            col = find_numeric_col(df, m.group(2))
            if col:
                val = df[col].max() if agg_fn == "max" else df[col].min()
                idx = df[col].idxmax() if agg_fn == "max" else df[col].idxmin()
                result["answer"] = f"{agg_fn.capitalize()} `{col}`: **{val:,.3f}** (row {idx})"
            else:
                result["error"] = f"Could not identify a numeric column in: '{m.group(2)}'"
            return result

    # ── Count by group ─────────────────────────────────────────────────────────
    m = re.search(r"(count|how many).+\bby\b\s+(.+)", q)
    if not m:
        m = re.search(r"(count|how many)\s+(.+?)(?:\s+per\s+|\s+for each\s+)(.+)", q)
    if m:
        grp_hint = m.group(m.lastindex)
        grp = find_column(df, grp_hint)
        if grp:
            counts = df[grp].value_counts().reset_index()
            counts.columns = [grp, "count"]
            result["df"]    = counts
            result["answer"] = f"Counts by `{grp}`:"
            result["chart"]  = ("bar", grp, "count")
        else:
            result["error"] = f"Could not identify a column for: '{grp_hint}'"
        return result

    # ── Distribution / value counts ────────────────────────────────────────────
    m = re.search(r"(distribution|unique values?|categories|breakdown)\s+(?:of\s+|for\s+)?(.+)", q)
    if m:
        col = find_column(df, m.group(2))
        if col:
            vc = df[col].value_counts().head(20).reset_index()
            vc.columns = [col, "count"]
            result["df"]    = vc
            result["answer"] = f"Value counts for `{col}`:"
            result["chart"]  = ("bar", col, "count")
        else:
            result["error"] = f"Could not find column: '{m.group(2)}'"
        return result

    # ── Correlation ────────────────────────────────────────────────────────────
    if re.search(r"(correlat|relationship|corr)", q):
        if len(numeric_cols) >= 2:
            result["df"]    = df[numeric_cols].corr().round(3)
            result["answer"] = "Correlation matrix:"
        else:
            result["answer"] = "Need at least 2 numeric columns for correlation."
        return result

    # ── Show top / head ────────────────────────────────────────────────────────
    m = re.search(r"(show|display|list|head|first|top)\s+(\d+)?\s*(rows?|records?|entries?)?", q)
    if m:
        n = int(m.group(2)) if m.group(2) else 10
        result["df"]    = df.head(n)
        result["answer"] = f"First {n} rows:"
        return result

    # ── Fallback ───────────────────────────────────────────────────────────────
    result["error"] = (
        "I couldn't understand that question. Try asking:\n"
        "- How many rows?\n"
        "- Average of [column]\n"
        "- Sum of [column] by [group]\n"
        "- Count by [column]\n"
        "- Distribution of [column]\n"
        "- Show top 10 rows\n"
        "- Describe / statistics\n"
        "- Missing values"
    )
    return result


# ── Sample CSV ─────────────────────────────────────────────────────────────────

SAMPLE_CSV = """name,age,salary,department,city,years_exp,performance
Alice,29,72000,Engineering,New York,5,4.2
Bob,34,85000,Engineering,San Francisco,9,3.8
Carol,27,58000,Marketing,Chicago,3,4.5
David,41,95000,Engineering,New York,15,4.1
Eva,32,62000,Sales,Los Angeles,7,3.9
Frank,26,55000,Marketing,Chicago,2,4.0
Grace,38,88000,Engineering,San Francisco,12,4.3
Henry,45,102000,Management,New York,20,3.7
Iris,30,67000,Sales,Chicago,5,4.4
Jack,35,78000,Marketing,Los Angeles,8,3.6
Karen,28,71000,Engineering,Seattle,4,4.1
Leo,43,91000,Management,New York,18,3.8
Maya,31,64000,Sales,San Francisco,6,4.0
Nathan,37,82000,Engineering,Seattle,10,4.2
Olivia,25,51000,Marketing,Chicago,1,3.9
"""


# ── UI ────────────────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload a CSV file", type="csv")
    use_sample = st.checkbox("Use sample employee dataset", value=uploaded is None)

with col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

import io as _io
if uploaded:
    st.session_state.df = pd.read_csv(uploaded)
elif use_sample:
    st.session_state.df = pd.read_csv(_io.StringIO(SAMPLE_CSV))

df = st.session_state.df

if df is not None:
    st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    with st.expander("Preview data"):
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Ask a question")

    SUGGESTED = [
        "How many rows?",
        "Describe the dataset",
        "Average salary by department",
        "Sum of salary by city",
        "Distribution of department",
        "Count by city",
        "Missing values",
        "Show top 5 rows",
        "Correlation",
    ]
    suggestion = st.selectbox("Quick questions", ["(type your own)"] + SUGGESTED)
    question   = st.text_input(
        "Your question",
        value="" if suggestion == "(type your own)" else suggestion,
    )

    if st.button("💬 Ask", type="primary") and question.strip():
        res = answer(df, question)
        st.session_state.chat_history.append({"q": question, "res": res})

    # Chat display
    for item in reversed(st.session_state.chat_history):
        q   = item["q"]
        res = item["res"]
        with st.container():
            st.markdown(f"**Q:** {q}")
            if res["error"]:
                st.warning(res["error"])
            else:
                if res["answer"]:
                    st.markdown(res["answer"])
                if res["df"] is not None:
                    st.dataframe(res["df"], use_container_width=True, hide_index=True)
                if res["chart"]:
                    chart_type, x_col, y_col = res["chart"]
                    chart_df = res["df"].set_index(x_col)[y_col]
                    st.bar_chart(chart_df)
            st.divider()
else:
    st.info("Upload a CSV or enable the sample dataset to start chatting.")
