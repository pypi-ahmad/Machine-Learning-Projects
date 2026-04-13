"""Fitness Dashboard — Streamlit app.

Log workouts, track weight, calories, and fitness goals.
Weekly summaries, streaks, and progress charts.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fitness Dashboard", layout="wide")
st.title("🏋️ Fitness Dashboard")

DATA_FILE = Path("fitness.json")
WORKOUT_TYPES = ["Running", "Cycling", "Swimming", "Weight Training",
                 "Yoga", "HIIT", "Walking", "Other"]


def load_data() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {"workouts": [], "weight_log": [], "goals": {}}


def save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, indent=2))


if "fit" not in st.session_state:
    st.session_state.fit = load_data()
data     = st.session_state.fit
workouts = data["workouts"]
weight_log = data["weight_log"]
goals    = data["goals"]

tab1, tab2, tab3, tab4 = st.tabs(["Log Workout", "Log Weight", "Dashboard", "Goals"])

with tab1:
    with st.form("log_workout"):
        st.subheader("Log a Workout")
        c1, c2    = st.columns(2)
        wtype     = c1.selectbox("Type", WORKOUT_TYPES)
        wo_date   = c2.date_input("Date", value=date.today())
        c1, c2, c3 = st.columns(3)
        duration  = c1.number_input("Duration (min)", 1, 600, 30)
        calories  = c2.number_input("Calories Burned", 0, 5000, 0)
        distance  = c3.number_input("Distance (km, optional)", 0.0, 1000.0, 0.0, step=0.1)
        notes     = st.text_area("Notes", height=60)
        submit    = st.form_submit_button("Log Workout", type="primary")

    if submit:
        workouts.append({
            "type":     wtype,
            "date":     str(wo_date),
            "duration": int(duration),
            "calories": int(calories),
            "distance": round(float(distance), 2),
            "notes":    notes,
        })
        save_data(data)
        st.success(f"Logged: {wtype} — {duration} min"
                   + (f", {distance:.1f} km" if distance else "")
                   + (f", {calories} kcal" if calories else ""))

with tab2:
    with st.form("log_weight"):
        st.subheader("Log Weight")
        c1, c2 = st.columns(2)
        w_date  = c1.date_input("Date", value=date.today(), key="w_date")
        weight  = c2.number_input("Weight (kg)", 20.0, 300.0, 70.0, step=0.1, format="%.1f")
        submit2 = st.form_submit_button("Log Weight", type="primary")

    if submit2:
        # Update existing entry for same date or add new
        existing = next((w for w in weight_log if w["date"] == str(w_date)), None)
        if existing:
            existing["weight"] = round(float(weight), 1)
        else:
            weight_log.append({"date": str(w_date), "weight": round(float(weight), 1)})
        save_data(data)
        st.success(f"Logged: {weight:.1f} kg on {w_date}")

    if weight_log:
        wdf = pd.DataFrame(sorted(weight_log, key=lambda x: x["date"]))
        st.subheader("Weight History")
        st.line_chart(wdf.set_index("date")["weight"])
        latest = sorted(weight_log, key=lambda x: x["date"])[-1]
        st.metric("Latest Weight", f"{latest['weight']} kg", delta=None)

with tab3:
    if not workouts:
        st.info("No workouts logged yet.")
    else:
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        week_wos   = [w for w in workouts if w["date"] >= str(week_start)]
        month_wos  = [w for w in workouts if w["date"].startswith(today.strftime("%Y-%m"))]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("This Week",        len(week_wos))
        c2.metric("This Month",       len(month_wos))
        c3.metric("Total Workouts",   len(workouts))
        total_cal = sum(w["calories"] for w in workouts if w["calories"])
        c4.metric("Total Calories",   f"{total_cal:,} kcal")

        # Streak
        all_dates = sorted({w["date"] for w in workouts}, reverse=True)
        streak = 0
        check  = today
        for d in all_dates:
            if d == str(check):
                streak += 1
                check -= timedelta(days=1)
            elif d < str(check):
                break
        st.metric("Current Streak", f"{streak} day(s)")

        # Weekly workout minutes
        st.subheader("Daily Duration (last 30 days)")
        cutoff = str(today - timedelta(days=30))
        recent = [w for w in workouts if w["date"] >= cutoff]
        if recent:
            df = pd.DataFrame(recent)
            daily_min = df.groupby("date")["duration"].sum()
            st.bar_chart(daily_min)

        # By type
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("By Workout Type")
            type_counts: dict[str, int] = {}
            for w in workouts: type_counts[w["type"]] = type_counts.get(w["type"], 0) + 1
            st.bar_chart(pd.Series(type_counts).sort_values(ascending=False))

        with col2:
            st.subheader("Calories by Type")
            type_cal: dict[str, int] = {}
            for w in workouts:
                if w["calories"]:
                    type_cal[w["type"]] = type_cal.get(w["type"], 0) + w["calories"]
            if type_cal:
                st.bar_chart(pd.Series(type_cal).sort_values(ascending=False))

        # Recent workouts table
        st.subheader("Recent Workouts")
        rows = [{"Date": w["date"], "Type": w["type"], "Duration": f"{w['duration']} min",
                 "Calories": f"{w['calories']} kcal" if w["calories"] else "—",
                 "Distance": f"{w['distance']} km" if w["distance"] else "—"}
                for w in sorted(workouts, key=lambda x: x["date"], reverse=True)[:20]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Set Fitness Goals")
    with st.form("goals_form"):
        c1, c2 = st.columns(2)
        weekly_target  = c1.number_input("Workouts per week", 0, 14, goals.get("weekly_workouts", 3))
        target_weight  = c2.number_input("Target weight (kg)", 0.0, 300.0,
                                          float(goals.get("target_weight", 70)), step=0.5)
        weekly_minutes = st.number_input("Weekly exercise minutes goal", 0, 3000,
                                         goals.get("weekly_minutes", 150))
        save_g = st.form_submit_button("Save Goals", type="primary")

    if save_g:
        data["goals"] = {
            "weekly_workouts": int(weekly_target),
            "target_weight":   round(float(target_weight), 1),
            "weekly_minutes":  int(weekly_minutes),
        }
        save_data(data)
        st.success("Goals saved!")

    if goals and workouts:
        st.subheader("Goal Progress")
        today      = date.today()
        week_start = today - timedelta(days=today.weekday())
        week_wos   = [w for w in workouts if w["date"] >= str(week_start)]
        week_mins  = sum(w["duration"] for w in week_wos)

        if goals.get("weekly_workouts"):
            pct = min(len(week_wos) / goals["weekly_workouts"], 1.0)
            st.write(f"**Workouts this week:** {len(week_wos)}/{goals['weekly_workouts']}")
            st.progress(pct)

        if goals.get("weekly_minutes"):
            pct2 = min(week_mins / goals["weekly_minutes"], 1.0)
            st.write(f"**Minutes this week:** {week_mins}/{goals['weekly_minutes']}")
            st.progress(pct2)

        if goals.get("target_weight") and weight_log:
            latest_w = sorted(weight_log, key=lambda x: x["date"])[-1]["weight"]
            diff     = round(latest_w - goals["target_weight"], 1)
            st.metric("Target Weight Progress",
                      f"{latest_w} kg",
                      delta=f"{diff:+.1f} kg from goal",
                      delta_color="inverse")
