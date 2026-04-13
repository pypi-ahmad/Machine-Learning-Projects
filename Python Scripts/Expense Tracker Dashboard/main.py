"""Expense Tracker Dashboard — Streamlit app.

Track income and expenses with categories, dates, and notes.
Data is persisted to a local CSV file.  Shows balance, category
breakdown, and monthly trend charts.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Expense Tracker", layout="wide")
st.title("💰 Expense Tracker")

DATA_FILE = Path("expenses.csv")
CATEGORIES = [
    "Food", "Transport", "Housing", "Utilities", "Healthcare",
    "Entertainment", "Shopping", "Education", "Savings", "Income", "Other",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Date", "Description", "Category", "Amount", "Type"])


def save_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# ---------------------------------------------------------------------------
# Sidebar — Add transaction
# ---------------------------------------------------------------------------
st.sidebar.header("Add Transaction")
with st.sidebar.form("add_form"):
    tx_date  = st.date_input("Date", value=date.today())
    tx_desc  = st.text_input("Description")
    tx_cat   = st.selectbox("Category", CATEGORIES)
    tx_type  = st.radio("Type", ["Expense", "Income"], horizontal=True)
    tx_amt   = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f")
    submitted = st.form_submit_button("Add")

if submitted and tx_desc and tx_amt > 0:
    new_row = pd.DataFrame([{
        "Date": pd.Timestamp(tx_date),
        "Description": tx_desc,
        "Category": tx_cat,
        "Amount": tx_amt,
        "Type": tx_type,
    }])
    st.session_state.df = pd.concat([df, new_row], ignore_index=True)
    save_data(st.session_state.df)
    st.sidebar.success("Transaction added!")
    df = st.session_state.df

if df.empty:
    st.info("No transactions yet. Add one using the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
income   = df[df["Type"] == "Income"]["Amount"].sum()
expenses = df[df["Type"] == "Expense"]["Amount"].sum()
balance  = income - expenses

col1, col2, col3 = st.columns(3)
col1.metric("💚 Total Income",   f"${income:,.2f}")
col2.metric("🔴 Total Expenses", f"${expenses:,.2f}")
col3.metric("💎 Balance",        f"${balance:,.2f}",
            delta=f"${balance:,.2f}", delta_color="normal")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Transactions", "By Category", "Monthly Trend", "Delete"])

with tab1:
    # Filters
    c1, c2 = st.columns(2)
    type_filter = c1.multiselect("Type", ["Expense", "Income"], default=["Expense", "Income"])
    cat_filter  = c2.multiselect("Category", CATEGORIES, default=CATEGORIES)
    view = df[df["Type"].isin(type_filter) & df["Category"].isin(cat_filter)]
    view = view.sort_values("Date", ascending=False)
    st.dataframe(view.reset_index(drop=True), use_container_width=True)

with tab2:
    exp_df = df[df["Type"] == "Expense"].copy()
    if exp_df.empty:
        st.info("No expenses yet.")
    else:
        cat_totals = exp_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        st.bar_chart(cat_totals)
        st.dataframe(cat_totals.rename("Total ($)").reset_index(), use_container_width=True)

with tab3:
    df2 = df.copy()
    df2["Month"] = df2["Date"].dt.to_period("M").astype(str)
    monthly = df2.groupby(["Month", "Type"])["Amount"].sum().unstack(fill_value=0)
    if not monthly.empty:
        st.line_chart(monthly)

with tab4:
    st.subheader("Delete Transaction")
    if not df.empty:
        idx = st.number_input("Row index to delete (0-based)", 0, len(df) - 1, 0)
        st.write(df.iloc[idx])
        if st.button("Delete this row"):
            st.session_state.df = df.drop(index=df.index[idx]).reset_index(drop=True)
            save_data(st.session_state.df)
            st.success("Deleted.")
            st.rerun()
