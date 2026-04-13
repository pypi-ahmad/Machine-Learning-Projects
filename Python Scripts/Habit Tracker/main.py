"""Habit Tracker — Streamlit app.

Create daily habits and mark them as done each day.
Shows streaks, completion rates, and a monthly heatmap.
Data is saved to a local JSON file.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Habit Tracker", layout="wide")
st.title("✅ Habit Tracker")

DATA_FILE = Path("habits.json")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {"habits": [], "log": {}}  # log: {"YYYY-MM-DD": [habit1, habit2, ...]}


def save(data: dict) -> None:
    DATA_FILE.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = load()

data   = st.session_state.data
habits = data["habits"]
log    = data["log"]

today  = date.today().isoformat()
log.setdefault(today, [])

# ---------------------------------------------------------------------------
# Sidebar — manage habits
# ---------------------------------------------------------------------------
st.sidebar.header("Manage Habits")
new_habit = st.sidebar.text_input("New habit name")
if st.sidebar.button("Add habit") and new_habit.strip():
    if new_habit.strip() not in habits:
        habits.append(new_habit.strip())
        save(data)
        st.sidebar.success(f"Added: {new_habit.strip()}")

if habits:
    del_habit = st.sidebar.selectbox("Delete habit", ["—"] + habits)
    if st.sidebar.button("Delete") and del_habit != "—":
        habits.remove(del_habit)
        save(data)
        st.sidebar.success(f"Deleted: {del_habit}")
        st.rerun()

if not habits:
    st.info("Add habits using the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Today's check-ins
# ---------------------------------------------------------------------------
st.subheader(f"📅 Today — {today}")
cols = st.columns(min(4, len(habits)))
for i, habit in enumerate(habits):
    with cols[i % len(cols)]:
        done = habit in log[today]
        checked = st.checkbox(habit, value=done, key=f"check_{habit}")
        if checked and habit not in log[today]:
            log[today].append(habit)
            save(data)
        elif not checked and habit in log[today]:
            log[today].remove(habit)
            save(data)

st.divider()

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def streak(habit: str) -> int:
    count = 0
    d = date.today()
    while True:
        ds = d.isoformat()
        if ds in log and habit in log[ds]:
            count += 1
            d -= timedelta(days=1)
        else:
            break
    return count


def completion_rate(habit: str, days: int = 30) -> float:
    done = 0
    for i in range(days):
        ds = (date.today() - timedelta(days=i)).isoformat()
        if ds in log and habit in log[ds]:
            done += 1
    return done / days * 100


st.subheader("📊 Statistics (last 30 days)")
stats_data = []
for habit in habits:
    stats_data.append({
        "Habit": habit,
        "Streak (days)": streak(habit),
        "Completion %": f"{completion_rate(habit):.1f}%",
        "Done today": "✅" if habit in log.get(today, []) else "❌",
    })
st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

# ---------------------------------------------------------------------------
# 30-day heatmap (simple table)
# ---------------------------------------------------------------------------
st.subheader("📆 Last 30 Days")
dates_30 = [(date.today() - timedelta(days=i)).isoformat() for i in range(29, -1, -1)]
heatmap_data = {}
for habit in habits:
    heatmap_data[habit] = [
        "✅" if (ds in log and habit in log[ds]) else "⬜"
        for ds in dates_30
    ]
heat_df = pd.DataFrame(heatmap_data, index=[d[5:] for d in dates_30]).T
st.dataframe(heat_df, use_container_width=True)
