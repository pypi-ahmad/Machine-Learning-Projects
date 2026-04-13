"""Budget Planner — Streamlit app.

Set a monthly budget by category and track spending against it.
Shows remaining budget, progress bars, and savings projections.
Data persisted to local JSON.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Budget Planner", layout="wide")
st.title("🗓️ Budget Planner")

DATA_FILE = Path("budget.json")

DEFAULT_CATEGORIES = [
    "Housing", "Food & Dining", "Transport", "Utilities", "Healthcare",
    "Entertainment", "Clothing", "Education", "Savings", "Other",
]


def load() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {
        "budgets": {cat: 0 for cat in DEFAULT_CATEGORIES},
        "expenses": [],
        "income": 0,
    }


def save(data: dict) -> None:
    DATA_FILE.write_text(json.dumps(data, indent=2))


if "data" not in st.session_state:
    st.session_state.data = load()

data = st.session_state.data
month = date.today().strftime("%Y-%m")

# Filter expenses for current month
this_month_expenses = [
    e for e in data["expenses"]
    if e["date"].startswith(month)
]

# ---------------------------------------------------------------------------
# Sidebar — income & budgets
# ---------------------------------------------------------------------------
st.sidebar.header(f"Budget for {month}")
income = st.sidebar.number_input("Monthly Income ($)", 0.0,
                                  value=float(data.get("income", 0)), step=50.0)
if income != data.get("income"):
    data["income"] = income
    save(data)

st.sidebar.subheader("Category Budgets")
for cat in DEFAULT_CATEGORIES:
    val = st.sidebar.number_input(cat, 0.0,
                                   value=float(data["budgets"].get(cat, 0)),
                                   step=10.0, key=f"budget_{cat}")
    data["budgets"][cat] = val
save(data)

# ---------------------------------------------------------------------------
# Add expense
# ---------------------------------------------------------------------------
with st.expander("➕ Add Expense"):
    with st.form("add_exp"):
        c1, c2, c3 = st.columns(3)
        exp_cat  = c1.selectbox("Category", DEFAULT_CATEGORIES)
        exp_amt  = c2.number_input("Amount ($)", 0.01, step=0.01, format="%.2f")
        exp_desc = c3.text_input("Description")
        exp_date = st.date_input("Date", date.today())
        submitted = st.form_submit_button("Add")
    if submitted and exp_amt > 0:
        data["expenses"].append({
            "date": exp_date.isoformat(),
            "category": exp_cat,
            "amount": exp_amt,
            "description": exp_desc,
        })
        save(data)
        st.success(f"Added ${exp_amt:.2f} to {exp_cat}")
        st.rerun()

# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------
total_budget = sum(data["budgets"].values())
total_spent  = sum(e["amount"] for e in this_month_expenses)
remaining    = income - total_spent if income > 0 else total_budget - total_spent
savings_rate = ((income - total_spent) / income * 100) if income > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("💵 Income",       f"${income:,.0f}")
col2.metric("📊 Total Budget", f"${total_budget:,.0f}")
col3.metric("🛒 Spent",        f"${total_spent:,.0f}")
col4.metric("💚 Remaining",    f"${remaining:,.0f}",
            delta=f"{savings_rate:.1f}% savings rate")

st.divider()

# ---------------------------------------------------------------------------
# Budget vs Actual
# ---------------------------------------------------------------------------
st.subheader("Budget vs Actual")

spent_by_cat = {cat: 0.0 for cat in DEFAULT_CATEGORIES}
for e in this_month_expenses:
    if e["category"] in spent_by_cat:
        spent_by_cat[e["category"]] += e["amount"]

rows = []
for cat in DEFAULT_CATEGORIES:
    budget = data["budgets"].get(cat, 0)
    spent  = spent_by_cat.get(cat, 0)
    if budget == 0 and spent == 0:
        continue
    pct = spent / budget * 100 if budget > 0 else 0
    status = "✅" if pct <= 80 else ("⚠️" if pct <= 100 else "🔴")
    rows.append({
        "Category": cat,
        "Budget ($)": budget,
        "Spent ($)": spent,
        "Remaining ($)": max(0, budget - spent),
        "% Used": f"{pct:.1f}%",
        "Status": status,
    })

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Progress bars
    st.subheader("Spending Progress")
    for row in rows:
        budget = row["Budget ($)"]
        spent  = row["Spent ($)"]
        pct    = spent / budget if budget > 0 else 0
        st.write(f"**{row['Category']}** — ${spent:.0f} / ${budget:.0f}")
        st.progress(min(pct, 1.0))

# ---------------------------------------------------------------------------
# Recent expenses
# ---------------------------------------------------------------------------
st.subheader(f"Transactions this month ({len(this_month_expenses)})")
if this_month_expenses:
    exp_df = pd.DataFrame(this_month_expenses).sort_values("date", ascending=False)
    st.dataframe(exp_df, use_container_width=True)

    # Chart
    cat_totals = pd.DataFrame(this_month_expenses).groupby("category")["amount"].sum()
    st.bar_chart(cat_totals)
else:
    st.info("No expenses recorded this month yet.")
