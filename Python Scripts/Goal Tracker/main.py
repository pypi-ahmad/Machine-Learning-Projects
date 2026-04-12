"""Goal Tracker — Streamlit app.

Set goals with deadlines and milestones, track progress,
and visualise completion.  Data stored as JSON.

Usage:
    streamlit run main.py
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Goal Tracker", layout="wide")
st.title("🎯 Goal Tracker")

DATA_FILE = Path("goals.json")

CATEGORIES = ["Health", "Career", "Finance", "Education", "Personal", "Relationships", "Other"]
PRIORITIES  = ["High", "Medium", "Low"]


def load_goals() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_goals(goals: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(goals, indent=2))


if "goals" not in st.session_state:
    st.session_state.goals = load_goals()

goals = st.session_state.goals

# ---------------------------------------------------------------------------
# Sidebar — add goal
# ---------------------------------------------------------------------------
st.sidebar.header("Add New Goal")
with st.sidebar.form("add_goal"):
    g_title    = st.text_input("Goal title *")
    g_desc     = st.text_area("Description", height=80)
    g_cat      = st.selectbox("Category", CATEGORIES)
    g_priority = st.selectbox("Priority", PRIORITIES)
    g_deadline = st.date_input("Deadline", value=date.today())
    g_milestones = st.text_area("Milestones (one per line)", height=80,
                                 placeholder="Complete outline\nFinish chapter 1")
    add_btn = st.form_submit_button("Add Goal")

if add_btn and g_title.strip():
    milestones = [
        {"text": m.strip(), "done": False}
        for m in g_milestones.splitlines() if m.strip()
    ]
    new_goal = {
        "id":         len(goals),
        "title":      g_title.strip(),
        "description": g_desc.strip(),
        "category":   g_cat,
        "priority":   g_priority,
        "deadline":   str(g_deadline),
        "milestones": milestones,
        "progress":   0,
        "status":     "Active",
    }
    goals.append(new_goal)
    st.session_state.goals = goals
    save_goals(goals)
    st.sidebar.success(f"Added: {g_title.strip()}")
    st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Active Goals", "Completed Goals", "Overview"])

def render_goals(goal_list: list[dict]) -> None:
    for g in goal_list:
        idx = next(i for i, x in enumerate(goals) if x["id"] == g["id"])
        with st.container(border=True):
            h1, h2 = st.columns([4, 1])
            with h1:
                st.write(f"### {g['title']}")
                st.caption(f"📁 {g['category']}  ·  ⚡ {g['priority']}  ·  📅 Deadline: {g['deadline']}")
                if g.get("description"):
                    st.write(g["description"])
            with h2:
                new_prog = st.slider("Progress %", 0, 100, g["progress"],
                                      key=f"prog_{g['id']}")
                if new_prog != g["progress"]:
                    goals[idx]["progress"] = new_prog
                    if new_prog == 100:
                        goals[idx]["status"] = "Completed"
                    save_goals(goals)
                    st.rerun()

            st.progress(g["progress"] / 100)

            if g.get("milestones"):
                st.write("**Milestones:**")
                for mi, ms in enumerate(g["milestones"]):
                    checked = st.checkbox(ms["text"], value=ms["done"],
                                           key=f"ms_{g['id']}_{mi}")
                    if checked != ms["done"]:
                        goals[idx]["milestones"][mi]["done"] = checked
                        done_count = sum(1 for m in goals[idx]["milestones"] if m["done"])
                        total = len(goals[idx]["milestones"])
                        goals[idx]["progress"] = int(done_count / total * 100)
                        if goals[idx]["progress"] == 100:
                            goals[idx]["status"] = "Completed"
                        save_goals(goals)
                        st.rerun()

            col_a, col_b = st.columns(2)
            with col_a:
                if g["status"] == "Active":
                    if st.button("✅ Mark Complete", key=f"done_{g['id']}"):
                        goals[idx]["status"]   = "Completed"
                        goals[idx]["progress"] = 100
                        save_goals(goals)
                        st.rerun()
            with col_b:
                if st.button("🗑️ Delete", key=f"del_{g['id']}"):
                    goals.pop(idx)
                    save_goals(goals)
                    st.rerun()

with tab1:
    active = [g for g in goals if g["status"] == "Active"]
    if not active:
        st.info("No active goals. Add one via the sidebar!")
    else:
        cat_opts = ["All"] + sorted({g["category"] for g in active})
        cat_sel  = st.selectbox("Filter by category", cat_opts)
        shown    = active if cat_sel == "All" else [g for g in active if g["category"] == cat_sel]
        render_goals(shown)

with tab2:
    completed = [g for g in goals if g["status"] == "Completed"]
    if not completed:
        st.info("No completed goals yet. Keep going!")
    else:
        render_goals(completed)

with tab3:
    if not goals:
        st.info("No goals yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Goals",     len(goals))
        c2.metric("Active Goals",    sum(1 for g in goals if g["status"] == "Active"))
        c3.metric("Completed Goals", sum(1 for g in goals if g["status"] == "Completed"))

        st.subheader("Progress by Goal")
        df = pd.DataFrame(goals)[["title", "progress", "category", "priority", "status"]]
        st.dataframe(df, use_container_width=True)

        st.subheader("Goals by Category")
        cat_counts = df["category"].value_counts()
        st.bar_chart(cat_counts)
