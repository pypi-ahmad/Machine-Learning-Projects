"""Time Tracker — Streamlit app.

Log time entries by project and task, view daily/weekly totals,
and export timesheets.  Data stored as CSV.

Usage:
    streamlit run main.py
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Time Tracker", layout="wide")
st.title("⏱️ Time Tracker")

DATA_FILE = Path("time_log.csv")

COLS = ["date", "project", "task", "start", "end", "duration_min", "notes"]


def load_log() -> pd.DataFrame:
    if DATA_FILE.exists():
        try:
            return pd.read_csv(DATA_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def save_log(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)


if "log" not in st.session_state:
    st.session_state.log = load_log()

log = st.session_state.log

# ---------------------------------------------------------------------------
# Sidebar — log time entry
# ---------------------------------------------------------------------------
st.sidebar.header("Log Time Entry")
with st.sidebar.form("log_entry"):
    e_date    = st.date_input("Date", value=date.today())
    e_project = st.text_input("Project *", placeholder="My Project")
    e_task    = st.text_input("Task",    placeholder="Design review")
    e_start   = st.time_input("Start time", value=datetime.now().replace(minute=0, second=0, microsecond=0).time())
    e_end     = st.time_input("End time",   value=(datetime.now().replace(second=0, microsecond=0) + timedelta(hours=1)).time())
    e_notes   = st.text_input("Notes (optional)")
    log_btn   = st.form_submit_button("Log Entry")

if log_btn and e_project.strip():
    start_dt = datetime.combine(e_date, e_start)
    end_dt   = datetime.combine(e_date, e_end)
    if end_dt <= start_dt:
        st.sidebar.error("End time must be after start time.")
    else:
        duration = int((end_dt - start_dt).total_seconds() / 60)
        new_row = pd.DataFrame([{
            "date":         str(e_date),
            "project":      e_project.strip(),
            "task":         e_task.strip(),
            "start":        e_start.strftime("%H:%M"),
            "end":          e_end.strftime("%H:%M"),
            "duration_min": duration,
            "notes":        e_notes.strip(),
        }])
        st.session_state.log = pd.concat([log, new_row], ignore_index=True)
        save_log(st.session_state.log)
        st.sidebar.success(f"Logged {duration} min on '{e_project.strip()}'")
        st.rerun()

log = st.session_state.log

if log.empty:
    st.info("No time entries yet. Log your first entry in the sidebar.")
    st.stop()

log["date"] = pd.to_datetime(log["date"])
log["duration_min"] = pd.to_numeric(log["duration_min"], errors="coerce").fillna(0)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Log", "Reports", "Export"])

with tab1:
    st.subheader("Time Log")
    projects   = ["All"] + sorted(log["project"].unique().tolist())
    proj_filter = st.selectbox("Project", projects)
    date_range  = st.date_input("Date range", [log["date"].min().date(), log["date"].max().date()])

    mask = (log["date"].dt.date >= date_range[0]) & (log["date"].dt.date <= date_range[1]) if len(date_range) == 2 else pd.Series([True]*len(log))
    if proj_filter != "All":
        mask &= log["project"] == proj_filter
    view = log[mask].sort_values("date", ascending=False).copy()
    view["duration"] = view["duration_min"].apply(lambda m: f"{int(m)//60}h {int(m)%60}m")
    st.dataframe(view[["date", "project", "task", "start", "end", "duration", "notes"]],
                 use_container_width=True, hide_index=True)

    total_min = int(view["duration_min"].sum())
    st.metric("Total Time", f"{total_min//60}h {total_min%60}m")

    if st.button("🗑️ Delete last entry"):
        st.session_state.log = log.drop(log.index[-1]).reset_index(drop=True)
        save_log(st.session_state.log)
        st.rerun()

with tab2:
    st.subheader("Reports")

    st.write("**Time by Project**")
    by_proj = log.groupby("project")["duration_min"].sum().sort_values(ascending=False)
    by_proj_h = by_proj.apply(lambda m: round(m / 60, 2)).rename("Hours")
    st.bar_chart(by_proj_h)

    st.write("**Daily Total (last 14 days)**")
    log_recent = log[log["date"] >= pd.Timestamp.now() - pd.Timedelta(days=14)]
    daily = log_recent.groupby(log_recent["date"].dt.date)["duration_min"].sum().rename("Minutes")
    st.line_chart(daily)

    st.write("**Weekly Summary**")
    log_copy = log.copy()
    log_copy["week"] = log_copy["date"].dt.to_period("W").astype(str)
    weekly = log_copy.groupby(["week", "project"])["duration_min"].sum().unstack(fill_value=0)
    weekly_h = (weekly / 60).round(2)
    st.dataframe(weekly_h, use_container_width=True)

with tab3:
    st.subheader("Export Timesheet")
    export_df = log.copy()
    export_df["hours"] = (export_df["duration_min"] / 60).round(2)
    csv = export_df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "timesheet.csv", "text/csv")
    st.dataframe(export_df, use_container_width=True)
