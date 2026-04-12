"""Personal Finance Dashboard — Streamlit app.

Track income, expenses, savings, and investments.
Net worth summary, spending breakdown, and monthly trend charts.
Data stored as CSV.

Usage:
    streamlit run main.py
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Personal Finance", layout="wide")
st.title("💰 Personal Finance Dashboard")

DATA_FILE = Path("finance.csv")

INCOME_CATS  = ["Salary", "Freelance", "Investment", "Rental", "Gift", "Other Income"]
EXPENSE_CATS = ["Housing", "Food", "Transport", "Healthcare", "Education",
                "Entertainment", "Shopping", "Utilities", "Insurance", "Savings", "Other"]


def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "type", "category", "description", "amount"])


def save_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)


if "fin" not in st.session_state:
    st.session_state.fin = load_data()

fin = st.session_state.fin

# ---------------------------------------------------------------------------
# Sidebar — add transaction
# ---------------------------------------------------------------------------
st.sidebar.header("Add Transaction")
with st.sidebar.form("add_txn"):
    t_date  = st.date_input("Date", value=date.today())
    t_type  = st.radio("Type", ["Income", "Expense"], horizontal=True)
    cats    = INCOME_CATS if t_type == "Income" else EXPENSE_CATS
    t_cat   = st.selectbox("Category", cats)
    t_desc  = st.text_input("Description")
    t_amt   = st.number_input("Amount ($)", 0.01, step=0.01, format="%.2f")
    add_btn = st.form_submit_button("Add")

if add_btn and t_amt > 0:
    new_row = pd.DataFrame([{
        "date":        pd.Timestamp(t_date),
        "type":        t_type,
        "category":    t_cat,
        "description": t_desc.strip(),
        "amount":      t_amt,
    }])
    st.session_state.fin = pd.concat([fin, new_row], ignore_index=True)
    save_data(st.session_state.fin)
    st.sidebar.success(f"Added {t_type}: ${t_amt:.2f}")
    st.rerun()

fin = st.session_state.fin

if fin.empty:
    st.info("No transactions yet. Add your first one in the sidebar.")
    st.stop()

fin["date"] = pd.to_datetime(fin["date"])
fin["amount"] = pd.to_numeric(fin["amount"], errors="coerce").fillna(0)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
total_income  = fin[fin["type"] == "Income"]["amount"].sum()
total_expense = fin[fin["type"] == "Expense"]["amount"].sum()
net           = total_income - total_expense
savings_rate  = (net / total_income * 100) if total_income > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Income",    f"${total_income:,.2f}")
c2.metric("Total Expenses",  f"${total_expense:,.2f}")
c3.metric("Net Balance",     f"${net:,.2f}", delta=f"{net:+.2f}")
c4.metric("Savings Rate",    f"{savings_rate:.1f}%")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Transactions", "Spending", "Trends", "Export"])

with tab1:
    st.subheader("All Transactions")
    type_filter = st.multiselect("Type", ["Income", "Expense"], default=["Income", "Expense"])
    month_opts  = sorted(fin["date"].dt.to_period("M").astype(str).unique(), reverse=True)
    month_sel   = st.selectbox("Month", ["All"] + month_opts)

    mask = fin["type"].isin(type_filter)
    if month_sel != "All":
        mask &= fin["date"].dt.to_period("M").astype(str) == month_sel
    view = fin[mask].sort_values("date", ascending=False).copy()
    view["date_str"] = view["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(view[["date_str","type","category","description","amount"]].rename(columns={"date_str":"date"}),
                 use_container_width=True, hide_index=True)

    if st.button("🗑️ Delete last entry"):
        st.session_state.fin = fin.drop(fin.index[-1]).reset_index(drop=True)
        save_data(st.session_state.fin)
        st.rerun()

with tab2:
    st.subheader("Spending by Category")
    expenses = fin[fin["type"] == "Expense"]
    if expenses.empty:
        st.info("No expense data yet.")
    else:
        by_cat = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
        st.bar_chart(by_cat)

        st.write("**Expense Breakdown**")
        df_cat = by_cat.reset_index()
        df_cat.columns = ["Category", "Amount"]
        df_cat["Percentage"] = (df_cat["Amount"] / df_cat["Amount"].sum() * 100).round(1)
        st.dataframe(df_cat, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Monthly Income vs Expenses")
    fin_copy = fin.copy()
    fin_copy["month"] = fin_copy["date"].dt.to_period("M").astype(str)
    monthly = fin_copy.groupby(["month","type"])["amount"].sum().unstack(fill_value=0)
    if "Income" not in monthly:
        monthly["Income"] = 0
    if "Expense" not in monthly:
        monthly["Expense"] = 0
    monthly["Net"] = monthly["Income"] - monthly["Expense"]
    st.line_chart(monthly[["Income","Expense","Net"]])

    st.write("**Monthly Summary Table**")
    st.dataframe(monthly.round(2), use_container_width=True)

with tab4:
    st.subheader("Export Data")
    csv = fin.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "finance_export.csv", "text/csv")
    st.dataframe(fin.sort_values("date", ascending=False), use_container_width=True)
