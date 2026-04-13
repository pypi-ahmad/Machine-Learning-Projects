"""Calendar Tracker — Streamlit app.

Add, view, and manage calendar events.
Daily/monthly view, reminders, and event categories.

Usage:
    streamlit run main.py
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Calendar Tracker", layout="wide")
st.title("📅 Calendar Tracker")

DATA_FILE = Path("events.json")
CATEGORIES = ["Work", "Personal", "Health", "Social", "Other"]
CAT_COLORS = {"Work": "🔵", "Personal": "🟢", "Health": "🔴", "Social": "🟡", "Other": "⚪"}


def load_events() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_events(events: list[dict]):
    DATA_FILE.write_text(json.dumps(events, indent=2))


if "events" not in st.session_state:
    st.session_state.events = load_events()
events = st.session_state.events

tab1, tab2, tab3 = st.tabs(["Add Event", "Monthly View", "All Events"])

with tab1:
    with st.form("add_event"):
        st.subheader("New Event")
        title    = st.text_input("Title", placeholder="Team meeting")
        c1, c2   = st.columns(2)
        ev_date  = c1.date_input("Date", value=date.today())
        ev_time  = c2.time_input("Time", value=datetime.now().replace(minute=0, second=0).time())
        category = st.selectbox("Category", CATEGORIES)
        duration = st.number_input("Duration (minutes)", 15, 480, 60, step=15)
        notes    = st.text_area("Notes", height=80)
        submit   = st.form_submit_button("Add Event", type="primary")

    if submit and title:
        events.append({
            "title":    title,
            "date":     str(ev_date),
            "time":     ev_time.strftime("%H:%M"),
            "category": category,
            "duration": int(duration),
            "notes":    notes,
        })
        save_events(events)
        st.success(f"Event '{title}' added for {ev_date} at {ev_time.strftime('%H:%M')}.")

with tab2:
    today = date.today()
    # Month navigation
    c1, c2, c3 = st.columns([1, 2, 1])
    if "cal_offset" not in st.session_state:
        st.session_state.cal_offset = 0
    if c1.button("◀ Prev"):
        st.session_state.cal_offset -= 1
    if c3.button("Next ▶"):
        st.session_state.cal_offset += 1
    if c2.button("Today"):
        st.session_state.cal_offset = 0

    offset     = st.session_state.cal_offset
    month_date = (today.replace(day=1) + timedelta(days=32 * offset)).replace(day=1)
    month_str  = month_date.strftime("%Y-%m")
    c2.markdown(f"<h3 style='text-align:center'>{month_date.strftime('%B %Y')}</h3>",
                unsafe_allow_html=True)

    month_events = [e for e in events if e["date"].startswith(month_str)]
    if not month_events:
        st.info(f"No events in {month_date.strftime('%B %Y')}.")
    else:
        by_day: dict[str, list] = {}
        for e in sorted(month_events, key=lambda x: (x["date"], x["time"])):
            by_day.setdefault(e["date"], []).append(e)

        for day_str, day_evs in sorted(by_day.items()):
            d = datetime.strptime(day_str, "%Y-%m-%d").date()
            label = "**Today**" if d == today else d.strftime("%a, %b %d")
            with st.expander(f"{label} — {len(day_evs)} event(s)"):
                for e in day_evs:
                    icon = CAT_COLORS.get(e["category"], "⚪")
                    st.markdown(f"{icon} **{e['time']}** — {e['title']} "
                                f"*({e['category']}, {e['duration']} min)*")
                    if e.get("notes"):
                        st.caption(e["notes"])

with tab3:
    if not events:
        st.info("No events yet.")
    else:
        cat_filter = st.multiselect("Filter by category", CATEGORIES, default=CATEGORIES)
        filtered   = [e for e in events if e["category"] in cat_filter]
        filtered   = sorted(filtered, key=lambda x: (x["date"], x["time"]))

        rows = [{"Date": e["date"], "Time": e["time"], "Title": e["title"],
                 "Category": e["category"], "Duration": f"{e['duration']} min"}
                for e in filtered]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(filtered)} event(s) shown")

        # Upcoming events
        st.subheader("Upcoming (Next 7 Days)")
        cutoff = str(today + timedelta(days=7))
        upcoming = [e for e in filtered if today.isoformat() <= e["date"] <= cutoff]
        if upcoming:
            for e in upcoming:
                icon = CAT_COLORS.get(e["category"], "⚪")
                st.markdown(f"{icon} **{e['date']} {e['time']}** — {e['title']}")
        else:
            st.info("No upcoming events in the next 7 days.")

        if st.button("Delete All Events", type="secondary"):
            st.session_state.events = []
            save_events([])
            st.rerun()
