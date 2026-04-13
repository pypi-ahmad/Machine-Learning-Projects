"""Subscription Tracker — Streamlit app.

Track recurring subscriptions: cost, billing cycle, renewal dates.
See monthly/yearly totals and upcoming renewals.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Subscription Tracker", layout="wide")
st.title("🔄 Subscription Tracker")

DATA_FILE  = Path("subscriptions.json")
CATEGORIES = ["Streaming", "Software", "News", "Gaming", "Fitness", "Cloud", "Other"]
CYCLES     = {"Monthly": 1, "Quarterly": 3, "Yearly": 12}


def load_subs() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_subs(subs: list[dict]):
    DATA_FILE.write_text(json.dumps(subs, indent=2))


def monthly_cost(sub: dict) -> float:
    return sub["amount"] / CYCLES[sub["cycle"]]


def next_renewal(sub: dict) -> str:
    start = date.fromisoformat(sub["start_date"])
    today = date.today()
    months = CYCLES[sub["cycle"]]
    d = start
    while d <= today:
        # advance by months
        m = d.month - 1 + months
        y = d.year + m // 12
        mo = m % 12 + 1
        day = min(d.day, [31,28+(y%4==0 and (y%100!=0 or y%400==0)),31,30,31,30,
                           31,31,30,31,30,31][mo-1])
        d = date(y, mo, day)
    return str(d)


if "subs" not in st.session_state:
    st.session_state.subs = load_subs()
subs = st.session_state.subs

tab1, tab2, tab3 = st.tabs(["Add Subscription", "Overview", "Upcoming Renewals"])

with tab1:
    with st.form("add_sub"):
        st.subheader("New Subscription")
        c1, c2    = st.columns(2)
        name      = c1.text_input("Service Name", placeholder="Netflix")
        category  = c2.selectbox("Category", CATEGORIES)
        c1, c2    = st.columns(2)
        amount    = c1.number_input("Amount ($)", min_value=0.01, step=0.01, format="%.2f")
        cycle     = c2.selectbox("Billing Cycle", list(CYCLES.keys()))
        start     = st.date_input("Start / Last Billed Date", value=date.today())
        notes     = st.text_input("Notes (optional)")
        submit    = st.form_submit_button("Add Subscription", type="primary")

    if submit and name:
        subs.append({
            "name":       name,
            "category":   category,
            "amount":     round(float(amount), 2),
            "cycle":      cycle,
            "start_date": str(start),
            "notes":      notes,
            "active":     True,
        })
        save_subs(subs)
        st.success(f"Added: {name} — ${amount:.2f}/{cycle.lower()}")

with tab2:
    active = [s for s in subs if s.get("active", True)]
    if not active:
        st.info("No active subscriptions yet.")
    else:
        monthly_total = sum(monthly_cost(s) for s in active)
        yearly_total  = monthly_total * 12

        c1, c2, c3 = st.columns(3)
        c1.metric("Active Subscriptions", len(active))
        c2.metric("Monthly Cost",  f"${monthly_total:,.2f}")
        c3.metric("Yearly Cost",   f"${yearly_total:,.2f}")

        rows = []
        for s in sorted(active, key=lambda x: monthly_cost(x), reverse=True):
            rows.append({
                "Service":       s["name"],
                "Category":      s["category"],
                "Amount":        f"${s['amount']:.2f}",
                "Cycle":         s["cycle"],
                "Monthly Cost":  f"${monthly_cost(s):.2f}",
                "Next Renewal":  next_renewal(s),
                "Notes":         s.get("notes", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # By category breakdown
        st.subheader("Monthly Cost by Category")
        cat_totals: dict[str, float] = {}
        for s in active:
            cat_totals[s["category"]] = cat_totals.get(s["category"], 0) + monthly_cost(s)
        st.bar_chart(pd.Series(cat_totals).sort_values(ascending=False))

        # Cancel / deactivate
        st.subheader("Cancel Subscription")
        names = [s["name"] for s in active]
        cancel_name = st.selectbox("Select to cancel", [""] + names)
        if st.button("Cancel") and cancel_name:
            for s in subs:
                if s["name"] == cancel_name:
                    s["active"] = False
            save_subs(subs)
            st.success(f"Cancelled: {cancel_name}")
            st.rerun()

with tab3:
    active = [s for s in subs if s.get("active", True)]
    if not active:
        st.info("No active subscriptions.")
    else:
        today    = date.today()
        cutoff30 = today + timedelta(days=30)
        upcoming = []
        for s in active:
            nr = date.fromisoformat(next_renewal(s))
            days_left = (nr - today).days
            upcoming.append({"Service": s["name"], "Renewal Date": str(nr),
                             "Days Left": days_left, "Amount": f"${s['amount']:.2f}",
                             "Cycle": s["cycle"]})
        upcoming.sort(key=lambda x: x["Days Left"])
        df = pd.DataFrame(upcoming)
        st.dataframe(df, use_container_width=True, hide_index=True)

        due_soon = [u for u in upcoming if u["Days Left"] <= 7]
        if due_soon:
            st.warning(f"⚠️ {len(due_soon)} subscription(s) renewing within 7 days!")
