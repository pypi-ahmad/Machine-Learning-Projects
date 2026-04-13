"""Attendance Dashboard — Streamlit app.

Mark and track attendance for students or employees.
Daily check-in, streak tracking, and monthly reports.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st.title("✅ Attendance Dashboard")

DATA_FILE = Path("attendance.json")


def load_data() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {"members": [], "records": {}}


def save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, indent=2))


if "data" not in st.session_state:
    st.session_state.data = load_data()

data    = st.session_state.data
members = data["members"]
records = data["records"]   # {name: [date_str, ...]}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Manage Members")
new_member = st.sidebar.text_input("Add member name")
if st.sidebar.button("Add") and new_member.strip():
    name = new_member.strip()
    if name not in members:
        members.append(name)
        save_data(data)
        st.rerun()

del_member = st.sidebar.selectbox("Remove member", [""] + members)
if st.sidebar.button("Remove") and del_member:
    members.remove(del_member)
    records.pop(del_member, None)
    save_data(data)
    st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Mark Attendance", "Reports", "Calendar"])

with tab1:
    if not members:
        st.info("Add members via the sidebar.")
    else:
        sel_date = st.date_input("Date", value=date.today())
        date_str = str(sel_date)
        st.subheader(f"Attendance for {date_str}")

        present = []
        cols = st.columns(min(4, len(members)))
        checks = {}
        for i, name in enumerate(sorted(members)):
            already_present = date_str in records.get(name, [])
            c = cols[i % len(cols)]
            checks[name] = c.checkbox(name, value=already_present, key=f"att_{name}_{date_str}")

        if st.button("💾 Save Attendance", type="primary"):
            for name, checked in checks.items():
                dates_list = records.setdefault(name, [])
                if checked and date_str not in dates_list:
                    dates_list.append(date_str)
                elif not checked and date_str in dates_list:
                    dates_list.remove(date_str)
            save_data(data)
            st.success("Attendance saved!")

        present_count = sum(1 for n in members if date_str in records.get(n, []))
        st.metric("Present today", f"{present_count}/{len(members)}")

with tab2:
    if not members:
        st.info("No members yet.")
    else:
        st.subheader("Attendance Summary")
        rows = []
        for name in sorted(members):
            attended = len(records.get(name, []))
            rows.append({"Name": name, "Days Present": attended})
        df = pd.DataFrame(rows)
        total_days = df["Days Present"].sum()
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Name")["Days Present"])

        st.subheader("Monthly Breakdown")
        month_sel = st.selectbox("Month", [
            (date.today() - timedelta(days=30*i)).strftime("%Y-%m")
            for i in range(12)
        ])
        month_rows = []
        for name in sorted(members):
            month_dates = [d for d in records.get(name, []) if d.startswith(month_sel)]
            month_rows.append({"Name": name, "Days Present": len(month_dates)})
        st.dataframe(pd.DataFrame(month_rows), use_container_width=True, hide_index=True)

with tab3:
    if not members:
        st.info("No members yet.")
    else:
        sel_member = st.selectbox("Member", sorted(members))
        all_dates  = sorted(records.get(sel_member, []))
        if not all_dates:
            st.info(f"No attendance records for {sel_member}.")
        else:
            df_cal = pd.DataFrame({"Date": pd.to_datetime(all_dates), "Present": 1})
            df_cal["Month"] = df_cal["Date"].dt.to_period("M").astype(str)
            monthly = df_cal.groupby("Month")["Present"].sum()
            st.subheader(f"{sel_member} — Monthly Attendance")
            st.bar_chart(monthly)
            st.caption(f"Total days attended: {len(all_dates)}")
            csv = df_cal[["Date","Present"]].to_csv(index=False).encode()
            st.download_button("📥 Export CSV", csv, f"{sel_member}_attendance.csv")
